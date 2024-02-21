#include "common/configs.hpp"
#include "common/complex.hpp"
#include "common/types.hpp"
#include "common/operations.hpp"
#include "common/opt.hpp"
#include "common/logerror.hpp"
#include "geometries/parallel/em.hpp"
#include "geometries/parallel/radon.hpp"
#include "geometries/parallel/bst.hpp"
#include "common10/cufft_utils.h"

#define NITER_MIN_REG 20 // must be greater than 1 so back_cu and recon_cu are all meaningful.

extern "C" {
void get_tEM_FQ(
float *sino_cu, float *recon_cu, float *angles_cu, float *flat_cu,
float *backcounts_cu, float *back_cu, cufftComplex *ft_sino_cu, cufftComplex *ft_recon_cu,
int nrays, int nangles, int blocksize, size_t recon_size, size_t ft_recon_size,
float scale, cudaStream_t stream, cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
cImage& cartesianblock, cImage& polarblock, cImage& realpolar,
cufftHandle plan1d, cufftHandle plan2d, int blocksize_bst, 
int zpad, int interpolation, float dx, float tv_param, int niter)
{
    int sizeimage = nrays;  /* Size of reconstruction */
    int pad0      = zpad+1; /* Padding value for FST (zpad) and BST (pad0)*/

    /* Initialize variables with BST */
    EMFQ_BST(backcounts_cu, sino_cu, angles_cu, nrays, nangles, blocksize, nrays, zpad+1);

    /* Compute (1.0/backcounts_cu) */
    calc_reciprocal_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(backcounts_cu,recon_size);

    // set_value<<<recon_size/NUM_THREADS, NUM_THREADS>>>(recon_cu, 1.0, recon_size);
    // CUDA_RT_CALL(cudaPeekAtLastError());
    // CUDA_RT_CALL(cudaDeviceSynchronize());

    /* Begin EM Frequency iterations */
    for (int k = 0; k < niter; ++k) {
        
        /* FST call */

        /* Set ft_recon_cu to zero */
        CUDA_RT_CALL(cudaMemsetAsync(ft_recon_cu, 0, sizeof(data_type)*ft_recon_size, stream));
        
        /* FST function for counts */
        pst_counts_real(ft_sino_cu, ft_recon_cu, sino_cu, recon_cu,
                        angles_cu, flat_cu,
                        plan_2D_forward, plan_1D_inverse,
                        nrays, nangles, blocksize,
                        zpad, interpolation, scale,
                        stream);

        /* BST call */
        EMFQ_BST_ITER(  back_cu, sino_cu, angles_cu, 
                        cartesianblock, polarblock, realpolar,
                        plan1d, plan2d,
                        nrays, nangles, blocksize, 
                        blocksize_bst, sizeimage, pad0);

        multiply_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(back_cu, backcounts_cu, recon_size);
        
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
        
        /* Call for Total variation regularization (TV)
        Fev 2024 - Memory bugs (or something else) persist! 
        
        'tv_param' is the regularization parameter od TV.

        If 'tv_param == 0', then it does not enter the TV computation
        */
        if (0.0 < tv_param && NITER_MIN_REG < k) {
            
            total_variation_2d<<<recon_size/NUM_THREADS, NUM_THREADS>>>(back_cu, recon_cu, backcounts_cu,
                                                                        recon_size, nrays, nrays, blocksize,tv_param);
            
            CUDA_RT_CALL(cudaPeekAtLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize());
        }
        /* End call for Total variation (TV) */

        multiply_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(recon_cu, back_cu, recon_size);
        
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

}
}    
extern "C" {
void get_tEM_FQ_GPU(CFG configs,
float *sino, float *recon, float *angles, float *flat,
int blocksize, int gpu)
{
    /*
    Definition of variables:
    blocksize:
        - Value of vertical block to be computed (fraction of nslices)
    */

    CUDA_RT_CALL(cudaSetDevice(gpu));

    /* Projection data sizes */
    int nrays     = configs.tomo.size.x;
    int nangles   = configs.tomo.size.y;

    /*zpad:
        - Padding number
        - Integer values: 0, 1, 2, 3
        - total padding = zpad * nrays
    */
    int zpad = configs.tomo.pad.x;  /* Padding value for FST (zpad) */

    /*interpolation:
        - Options: 'bilinear' and 'nearest'
        - 'nearest'  = 0 (see Python function)
        - 'bilinear' = 1 (see Python function)
    */
    int interpolation = configs.interpolation;

    /*dx: 
        - Detector pixel size in [X] units (can be any)
        - My preference is to always use METERS 
        - Be consistent with the units!!
    */
    float dx = configs.geometry.detector_pixel_x;

    /* tv_param:
        - Regularization parameter for total variation (TV) regularization
        - If 'tv_param' =< 0.0, there is no application of TV regularization
    */
    float tv_param = configs.reconstruction_tv;

    /* niter:
        - Number of iterations for EM
    */
    int niter = configs.em_iterations;


    /* Declaration of FST variables */

    /* Cuda kernels parameters */
    cudaStream_t stream, stream_H2D, stream_D2H; // 'stream' variable is for cuFFT and kernels.

    /* cuFFT parameters */
    cufftHandle plan_2D_forward;
    cufftHandle plan_1D_inverse;
    cufftComplex *ft_recon_cu = nullptr; // pointer to 2D Fourier transform of recon on GPU.
    cufftComplex *ft_sino_cu  = nullptr;  // pointer to 1D Fourier transform of sino on GPU.

    /* cuFFT Dimensions */
    std::array<int, FT_PST_RANK_FORWARD> forward_fft_dim = {nrays*(1+zpad), nrays*(1+zpad)};
    std::array<int, FT_PST_RANK_INVERSE> inverse_fft_dim = {nrays*(1+zpad)};

    /* GPU ptr */
    float *recon_cu      = nullptr; // pointer to recon on GPU.
    float *back_cu       = nullptr;  // Auxiliar pointer to iterations backprojection on GPU.
	float *sino_cu       = nullptr;  // pointer to tomogram (sinogram) on GPU.
    float *backcounts_cu = nullptr;  // Auxiliar pointer to iterations backcounts on GPU.
    float *angles_cu     = nullptr;  // pointer to angles list on GPU.
    float *flat_cu       = nullptr;  // pointer to flat on GPU.
    
    /* Datas Dimensions */
    size_t recon_size    = static_cast<size_t>(blocksize * nrays) * nrays;
	size_t sino_size     = static_cast<size_t>(blocksize * nrays) * nangles;
    size_t ft_sino_size  = static_cast<size_t>(blocksize*nangles)*nrays*(1+zpad);
    size_t ft_recon_size = static_cast<size_t>(blocksize*nrays*(1+zpad))*nrays*(1+zpad);

    float scale = (0 < dx)? dx/(float)inverse_fft_dim[0] : 1.0/(float)inverse_fft_dim[0];

    /* Begin of FST initialization */ 

    /* cuFFT parameters */
    CUFFT_CALL(cufftCreate(&plan_2D_forward));
    CUFFT_CALL(cufftPlanMany(
		&plan_2D_forward, FT_PST_RANK_FORWARD, forward_fft_dim.data(),  // *plan, rank, *n,
		nullptr, 1, forward_fft_dim[0] * forward_fft_dim[1],    		// *inembed, istride, idist,
        nullptr, 1, forward_fft_dim[0] * forward_fft_dim[1],    		// *onembed, ostride, odist,
        CUFFT_C2C, blocksize));							                // type, batch.
    
    CUFFT_CALL(cufftCreate(&plan_1D_inverse));
    CUFFT_CALL(cufftPlanMany( 
        &plan_1D_inverse, FT_PST_RANK_INVERSE, inverse_fft_dim.data(), 	// *plan, rank, *n,
        nullptr, 1, inverse_fft_dim[0],  			                    // *inembed, istride, idist,
        nullptr, 1, inverse_fft_dim[0],  			                    // *onembed, ostride, odist,
        CUFFT_C2C, nangles * blocksize));	                            // type, batch.
    
    /* Cuda kernels parameters - too advanced for me right now */
    CUDA_RT_CALL(cudaStreamCreate(&stream_H2D));
    CUDA_RT_CALL(cudaStreamCreate(&stream_D2H));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan_2D_forward, stream));
    CUFFT_CALL(cufftSetStream(plan_1D_inverse, stream));

    /* GPU ptr allocation */
    CUDA_RT_CALL(cudaHostRegister(recon,
        sizeof(real_data_type) * static_cast<size_t>(blocksize) * nrays * nrays,
        cudaHostRegisterDefault));

    CUDA_RT_CALL(cudaHostRegister(sino,
        sizeof(real_data_type) * static_cast<size_t>(blocksize) * nangles * nrays,
        cudaHostRegisterDefault));

    CUDA_RT_CALL(cudaMalloc(&recon_cu     ,sizeof(real_data_type) * recon_size     ));
    CUDA_RT_CALL(cudaMalloc(&back_cu      ,sizeof(real_data_type) * recon_size     ));
    CUDA_RT_CALL(cudaMalloc(&backcounts_cu,sizeof(real_data_type) * recon_size     ));
    CUDA_RT_CALL(cudaMalloc(&sino_cu      ,sizeof(real_data_type) * sino_size      ));
    CUDA_RT_CALL(cudaMalloc(&ft_recon_cu  ,sizeof(data_type)      * ft_recon_size  ));
    CUDA_RT_CALL(cudaMalloc(&ft_sino_cu   ,sizeof(data_type)      * ft_sino_size   ));
    CUDA_RT_CALL(cudaMalloc(&flat_cu      ,sizeof(float)          * nrays*blocksize));
    CUDA_RT_CALL(cudaMalloc(&angles_cu    ,sizeof(float)          * nangles        ));
    
    /* Copy data to GPU */
    CUDA_RT_CALL(cudaMemcpyAsync(angles_cu,angles,sizeof(float) * nangles        ,cudaMemcpyHostToDevice,stream_H2D));
    CUDA_RT_CALL(cudaMemcpyAsync(flat_cu  ,flat  ,sizeof(float) * nrays*blocksize,cudaMemcpyHostToDevice,stream_H2D));
    CUDA_RT_CALL(cudaMemcpyAsync(sino_cu  ,sino  ,sizeof(float) * sino_size      ,cudaMemcpyHostToDevice,stream_H2D));
    CUDA_RT_CALL(cudaMemcpyAsync(recon_cu ,recon ,sizeof(float) * recon_size     ,cudaMemcpyHostToDevice,stream_H2D));
    
    CUDA_RT_CALL(cudaStreamSynchronize(stream_H2D));
    /* End of FST initialization */ 

    /* Begin of BST initialization */ 
    int blocksize_bst = 1;
    int sizeimage     = nrays;
    int pad0          = zpad+1;

    /* CImage complex C++ Giovanni struct pointers 
    for BST implementation - See 'inc/commons/types.hpp'
    Change near future!! */
    /* GPU ptr */
    cImage cartesianblock_bst(sizeimage   , sizeimage * blocksize_bst);
    cImage     polarblock_bst(nrays * pad0, nangles   * blocksize_bst);
    cImage      realpolar_bst(nrays * pad0, nangles   * blocksize_bst);

    /* cuFFT parameters */    
    cufftHandle plan1d_bst;
    cufftHandle plan2d_bst;

    int dimms1d[] = {(int)nrays*pad0/2};
    int dimms2d[] = {(int)sizeimage,(int)sizeimage};
    int beds[]    = {nrays*pad0/2};

    HANDLE_FFTERROR( cufftPlanMany(&plan1d_bst, 1, dimms1d, beds, 1, 
        nrays*pad0/2, beds, 1, nrays*pad0/2, CUFFT_C2C, nangles*blocksize_bst*2) );

    HANDLE_FFTERROR( cufftPlanMany(&plan2d_bst, 2, dimms2d, nullptr, 
        0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize_bst) );

    /* End of BST initialization */  

    /* Calls EM Frequency iterations */
    get_tEM_FQ( sino_cu, recon_cu, angles_cu, flat_cu,
                backcounts_cu, back_cu, ft_sino_cu, ft_recon_cu,
                nrays, nangles, blocksize, recon_size, ft_recon_size,
                scale, stream, plan_2D_forward, plan_1D_inverse,
                cartesianblock_bst, polarblock_bst, realpolar_bst,
                plan1d_bst, plan2d_bst, blocksize_bst, 
                zpad, interpolation, dx, tv_param, niter);

    /* Copy computed reconstruction back to CPU */
    CUDA_RT_CALL(cudaMemcpyAsync(   recon, recon_cu, sizeof(real_data_type) * recon_size,
                                    cudaMemcpyDeviceToHost, stream_D2H));
    
    CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));

    /* Free ptr - Dealocation */    
    CUDA_RT_CALL(cudaHostUnregister(recon));
    CUDA_RT_CALL(cudaHostUnregister(sino));
    
    /* Free and Destroy FST allocations */
    free_cuda_counts_real(
        angles_cu, flat_cu, recon_cu, sino_cu,
        ft_recon_cu, ft_sino_cu,
        plan_2D_forward, plan_1D_inverse,
        stream, stream_H2D, stream_D2H);

    /* Free FST auxiliar allocations */
    CUDA_RT_CALL(cudaFree(back_cu));
    CUDA_RT_CALL(cudaFree(backcounts_cu));
    
    /* Destroy BST plans */
    CUFFT_CALL(cufftDestroy(plan1d_bst));
    CUFFT_CALL(cufftDestroy(plan2d_bst));
    
    
    // Apagar (só coloquei pra ver em qual crasha):
    // cartesianblock_bst.~cImage();
    // polarblock_bst.~cImage();
    // realpolar_bst.~cImage();
    // Apagar (não deveria ser necessário):
    // CUDA_RT_CALL(cudaDeviceReset());
}
}

//----------------------
// EM on frequency Threads Block algorithm
//----------------------

extern "C"{

    void _get_tEM_FQ_GPU(CFG configs,
    float *count, float *recon, float *angles, float *flat, 
    int blockgpu, int gpu)
    {
        /* Projection data sizes */
        int nrays     = configs.tomo.size.x;
        int nangles   = configs.tomo.size.y;
        int pad       = configs.tomo.pad.x;
        
        size_t blocksize_aux = calc_blocksize(blockgpu, nangles, nrays, pad, true); 
        size_t blocksize = min((size_t)blockgpu, blocksize_aux);

        for (size_t b = 0; b < blockgpu; b += blocksize) {

            blocksize = min(size_t(blockgpu) - b, blocksize);

            get_tEM_FQ_GPU( configs,
                            count + (size_t)b*nrays*nangles, 
                            recon + (size_t)b*nrays*nrays,
                            angles, 
                            flat + (size_t)b*nrays,
                            blocksize, gpu);
        }
        cudaDeviceSynchronize();
    }

    void get_tEM_FQ_MultiGPU(int* gpus, int ngpus, 
    float *count, float *recon, float *angles, float *flat,
    float *paramf, int *parami)
    {
        int i, Maxgpudev;
		
		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

        CFG configs; GPU gpu_parameters;

        setEMFQParameters(&configs, paramf, parami);
        printEMFQParameters(&configs);

        /* Projection data sizes */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;
        int nslices  = configs.tomo.size.z;
        
        /* Object (reconstruction) data sizes */
        int nx       = configs.obj.size.x;
        int ny       = configs.obj.size.y;
        int nz       = configs.obj.size.z;
            
        int t;
        int blockgpu = (nslices + ngpus - 1) / ngpus;
        
        std::vector<std::future<void>> threads;
        if (ngpus == 1){

            _get_tEM_FQ_GPU(configs, count, recon, angles, flat, nslices, gpus[0]);

        }else{
            for(t = 0; t < ngpus; t++){ 
                
                blockgpu = min(nslices - blockgpu * t, blockgpu);

                threads.push_back(std::async( std::launch::async, _get_tEM_FQ_GPU, 
                    configs,
                    count + (size_t)t * blockgpu * nrays * nangles, 
                    recon + (size_t)t * blockgpu * nrays * nrays, 
                    angles, 
                    flat + (size_t)t * blockgpu * nrays, 
                    blockgpu, gpus[t]));
            }

            for(auto& t : threads)
                t.get();
        }
    }
}

