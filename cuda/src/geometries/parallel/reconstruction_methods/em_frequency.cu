#include "common/configs.hpp"
#include "common/complex.hpp"
#include "common/types.hpp"
#include "common/operations.hpp"
#include "common/logerror.hpp"
#include "geometries/parallel/em.hpp"
#include "geometries/parallel/radon.hpp"
#include "geometries/parallel/bst.hpp"
#include "common10/cufft_utils.h"

#define NITER_MIN_REG 20 // must be greater than 1 so back_cu and recon_cu are all meaningful.


// extern "C" {
// void emfreq_v0(
//     float *sino, float *recon, float *angles, float *flat,
//     int nrays, int nangles, int blocksize, 
//     int zpad, int interpolation, float dx,
//     int niter, int gpu)
// {
//     cudaStream_t stream, stream_H2D, stream_D2H; // stream is for cuFFT and kernels.
//     cufftHandle plan_2D_forward;
//     cufftHandle plan_1D_inverse;
//     cufftComplex *ft_recon_cu = nullptr; // pointer to 2D Fourier transform of recon on GPU.
//     cufftComplex *ft_sino_cu = nullptr;  // pointer to 1D Fourier transform of sino on GPU.
//     float *recon_cu = nullptr; // pointer to recon on GPU.
//     float *back_cu = nullptr;
// 	float *sino_cu = nullptr;  // pointer to sino on GPU.
//     float *backcounts_cu = nullptr;
//     float *angles_cu;
//     float *flat_cu;
//     std::array<int, FT_PST_RANK_FORWARD> forward_fft_dim = {nrays*(1+zpad), nrays*(1+zpad)};
//     std::array<int, FT_PST_RANK_INVERSE> inverse_fft_dim = {nrays*(1+zpad)};
//     size_t recon_size = static_cast<size_t>(blocksize * nrays) * nrays;
// 	size_t sino_size = static_cast<size_t>(blocksize * nrays) * nangles;
//     size_t ft_sino_size = static_cast<size_t>(blocksize*nangles)*nrays*(1+zpad);
//     size_t ft_recon_size = static_cast<size_t>(blocksize*nrays*(1+zpad))*nrays*(1+zpad);
//     float scale = (0 < dx)? dx/(float)inverse_fft_dim[0] : 1.0/(float)inverse_fft_dim[0];

//     CUDA_RT_CALL(cudaSetDevice(gpu));

//     CUFFT_CALL(cufftCreate(&plan_2D_forward));
//     CUFFT_CALL(cufftPlanMany(
// 		&plan_2D_forward, FT_PST_RANK_FORWARD, forward_fft_dim.data(),    // *plan, rank, *n,
// 		nullptr, 1, forward_fft_dim[0] * forward_fft_dim[1],    		  // *inembed, istride, idist,
//         nullptr, 1, forward_fft_dim[0] * forward_fft_dim[1],    		  // *onembed, ostride, odist,
//         CUFFT_C2C, blocksize));							                  // type, batch.
//     CUFFT_CALL(cufftCreate(&plan_1D_inverse));
//     CUFFT_CALL(cufftPlanMany(
// 		&plan_1D_inverse, FT_PST_RANK_INVERSE, inverse_fft_dim.data(), 	// *plan, rank, *n,
// 		nullptr, 1, inverse_fft_dim[0],  			        // *inembed, istride, idist,
//         nullptr, 1, inverse_fft_dim[0],  			        // *onembed, ostride, odist,
//         CUFFT_C2C, nangles * blocksize));	                // type, batch.

//     CUDA_RT_CALL(cudaStreamCreate(&stream_H2D));
//     CUDA_RT_CALL(cudaStreamCreate(&stream_D2H));
//     CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
//     CUFFT_CALL(cufftSetStream(plan_2D_forward, stream));
//     CUFFT_CALL(cufftSetStream(plan_1D_inverse, stream));

//     CUDA_RT_CALL(cudaHostRegister(
//         recon,
//         sizeof(real_data_type) * static_cast<size_t>(blocksize) * nrays * nrays,
//         cudaHostRegisterDefault));
//     CUDA_RT_CALL(cudaHostRegister(
//         sino,
//         sizeof(real_data_type) * static_cast<size_t>(blocksize) * nangles * nrays,
//         cudaHostRegisterDefault));

//     CUDA_RT_CALL(cudaMalloc(&recon_cu, sizeof(real_data_type) * recon_size));
//     CUDA_RT_CALL(cudaMalloc(&back_cu, sizeof(real_data_type) * recon_size));
//     CUDA_RT_CALL(cudaMalloc(&backcounts_cu, sizeof(real_data_type) * recon_size));
//     CUDA_RT_CALL(cudaMalloc(&sino_cu, sizeof(real_data_type) * sino_size));
//     CUDA_RT_CALL(cudaMalloc(&ft_recon_cu, sizeof(data_type) * ft_recon_size));
//     CUDA_RT_CALL(cudaMalloc(&ft_sino_cu, sizeof(data_type) * ft_sino_size));
//     CUDA_RT_CALL(cudaMalloc(&flat_cu, sizeof(float) * nrays*blocksize));
//     CUDA_RT_CALL(cudaMalloc(&angles_cu, sizeof(float) * nangles));
//     CUDA_RT_CALL(cudaMemcpyAsync(
// 		angles_cu, angles, sizeof(float) * nangles,
//         cudaMemcpyHostToDevice, stream_H2D));
//     CUDA_RT_CALL(cudaMemcpyAsync(
// 		flat_cu, flat, sizeof(float) * nrays*blocksize,
//         cudaMemcpyHostToDevice, stream_H2D));
//     CUDA_RT_CALL(cudaMemcpyAsync(
// 		sino_cu, sino, sizeof(float) * sino_size,
//         cudaMemcpyHostToDevice, stream_H2D));
//     CUDA_RT_CALL(cudaMemcpyAsync(
// 		recon_cu, recon, sizeof(float) * recon_size,
//         cudaMemcpyHostToDevice, stream_H2D));
//     CUDA_RT_CALL(cudaStreamSynchronize(stream_H2D));

//     BST(backcounts_cu, sino_cu, nrays, nangles, blocksize, nrays, zpad+1);
//     calc_reciprocal_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
//         backcounts_cu,
//         recon_size);

// // remover a partir daqui.
// //  CUDA_RT_CALL(cudaMemcpyAsync(
// //     sino, recon_cu, sizeof(real_data_type) * recon_size,
// //     cudaMemcpyDeviceToHost, stream_D2H));
// //  CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
// //  return;
// // parar de remover aqui.

//     // set_value<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
//     //     recon_cu, 1.0, recon_size);
//     // CUDA_RT_CALL(cudaPeekAtLastError());
//     // CUDA_RT_CALL(cudaDeviceSynchronize());

//     for (int k = 0; k < niter; ++k) {
//         // FST:
//         CUDA_RT_CALL(cudaMemsetAsync(ft_recon_cu, 0, sizeof(data_type)*ft_recon_size, stream));
//         pst_counts_real(
//             ft_sino_cu, ft_recon_cu, sino_cu, recon_cu,
//             angles_cu, flat_cu,
//             plan_2D_forward, plan_1D_inverse,
//             nrays, nangles, blocksize,
//             zpad, interpolation, scale,
//             stream);

// // remover a partir daqui.
// // f (k == 0) {
// //   CUDA_RT_CALL(cudaMemcpyAsync(
// //       recon, recon_cu, sizeof(real_data_type) * recon_size,
// //       cudaMemcpyDeviceToHost, stream_D2H));
// //   CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
// //   return;
// // 
// // parar de remover aqui.

//         // BST:
//         BST(back_cu, sino_cu, nrays, nangles, blocksize, nrays, zpad+1);

// // remover a partir daqui.
// // f (k == 0) { 
// //    CUDA_RT_CALL(cudaMemcpyAsync(
// //        sino, sino_cu, sizeof(real_data_type) * recon_size,
// //        cudaMemcpyDeviceToHost, stream_D2H));
// //    CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
// //    // return;
// // 
// // parar de remover aqui.

//         multiply_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
//             back_cu, backcounts_cu, recon_size);
//         CUDA_RT_CALL(cudaPeekAtLastError());
//         CUDA_RT_CALL(cudaDeviceSynchronize());

// // remover a partir daqui.
// // f (k == 0) {
// //    CUDA_RT_CALL(cudaMemcpyAsync(
// //        sino, back_cu, sizeof(real_data_type) * recon_size,
// //        cudaMemcpyDeviceToHost, stream_D2H));
// //    CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
// //    // return;
// // 
// // parar de remover aqui.
// // remover a partir daqui.
// // f (k == 0) {
// //    CUDA_RT_CALL(cudaMemcpyAsync(
// //        recon, backcounts_cu, sizeof(real_data_type) * recon_size,
// //        cudaMemcpyDeviceToHost, stream_D2H));
// //    CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
// //    return;
// // 
// // parar de remover aqui.

//         multiply_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
//             recon_cu, back_cu, recon_size);
//         CUDA_RT_CALL(cudaPeekAtLastError());
//         CUDA_RT_CALL(cudaDeviceSynchronize());

// // remover a partir daqui.
// // f (k == 0) {
// //    CUDA_RT_CALL(cudaMemcpyAsync(
// //        recon, recon_cu, sizeof(real_data_type) * recon_size,
// //        cudaMemcpyDeviceToHost, stream_D2H));
// //    CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
// //    return;
// // 
// // parar de remover aqui.
//     }

//     CUDA_RT_CALL(cudaMemcpyAsync(
//         recon, recon_cu, sizeof(real_data_type) * recon_size,
//         cudaMemcpyDeviceToHost, stream_D2H));
//     CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
        
//     CUDA_RT_CALL(cudaHostUnregister(recon));
//     CUDA_RT_CALL(cudaHostUnregister(sino));
//     free_cuda_counts_real(
//         angles_cu, flat_cu, recon_cu, sino_cu,
//         ft_recon_cu, ft_sino_cu,
//         plan_2D_forward, plan_1D_inverse,
//         stream, stream_H2D, stream_D2H);
// }
// }


extern "C" {
void _get_tEM_FQ_GPU(
float *sino, float *recon, float *angles, float *flat,
int nrays, int nangles, int blocksize,
int zpad, int interpolation, float dx, float tv_param,
int niter, int gpu)
{
    cudaStream_t stream, stream_H2D, stream_D2H; // 'stream' variable is for cuFFT and kernels.
    cufftHandle plan_2D_forward;
    cufftHandle plan_1D_inverse;
    cufftComplex *ft_recon_cu = nullptr; // pointer to 2D Fourier transform of recon on GPU.
    cufftComplex *ft_sino_cu = nullptr;  // pointer to 1D Fourier transform of sino on GPU.
    float *recon_cu = nullptr; // pointer to recon on GPU.
    float *back_cu = nullptr;  // pointer to backprojection of 
	float *sino_cu = nullptr;  // pointer to sino on GPU.
    float *backcounts_cu = nullptr;  // pointer to backcounts on GPU.
    float *angles_cu;
    float *flat_cu;
    std::array<int, FT_PST_RANK_FORWARD> forward_fft_dim = {nrays*(1+zpad), nrays*(1+zpad)};
    std::array<int, FT_PST_RANK_INVERSE> inverse_fft_dim = {nrays*(1+zpad)};
    size_t recon_size = static_cast<size_t>(blocksize * nrays) * nrays;
	size_t sino_size = static_cast<size_t>(blocksize * nrays) * nangles;
    size_t ft_sino_size = static_cast<size_t>(blocksize*nangles)*nrays*(1+zpad);
    size_t ft_recon_size = static_cast<size_t>(blocksize*nrays*(1+zpad))*nrays*(1+zpad);
    float scale = (0 < dx)? dx/(float)inverse_fft_dim[0] : 1.0/(float)inverse_fft_dim[0];

    CUDA_RT_CALL(cudaSetDevice(gpu));

    // FST initialization:
    CUFFT_CALL(cufftCreate(&plan_2D_forward));
    CUFFT_CALL(cufftPlanMany(
		&plan_2D_forward, FT_PST_RANK_FORWARD, forward_fft_dim.data(),    // *plan, rank, *n,
		nullptr, 1, forward_fft_dim[0] * forward_fft_dim[1],    		  // *inembed, istride, idist,
        nullptr, 1, forward_fft_dim[0] * forward_fft_dim[1],    		  // *onembed, ostride, odist,
        CUFFT_C2C, blocksize));							                  // type, batch.
    CUFFT_CALL(cufftCreate(&plan_1D_inverse));
    CUFFT_CALL(cufftPlanMany(
		&plan_1D_inverse, FT_PST_RANK_INVERSE, inverse_fft_dim.data(), 	// *plan, rank, *n,
		nullptr, 1, inverse_fft_dim[0],  			        // *inembed, istride, idist,
        nullptr, 1, inverse_fft_dim[0],  			        // *onembed, ostride, odist,
        CUFFT_C2C, nangles * blocksize));	                // type, batch.

    CUDA_RT_CALL(cudaStreamCreate(&stream_H2D));
    CUDA_RT_CALL(cudaStreamCreate(&stream_D2H));
    CUDA_RT_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUFFT_CALL(cufftSetStream(plan_2D_forward, stream));
    CUFFT_CALL(cufftSetStream(plan_1D_inverse, stream));

    CUDA_RT_CALL(cudaHostRegister(
        recon,
        sizeof(real_data_type) * static_cast<size_t>(blocksize) * nrays * nrays,
        cudaHostRegisterDefault));
    CUDA_RT_CALL(cudaHostRegister(
        sino,
        sizeof(real_data_type) * static_cast<size_t>(blocksize) * nangles * nrays,
        cudaHostRegisterDefault));

    CUDA_RT_CALL(cudaMalloc(&recon_cu, sizeof(real_data_type) * recon_size));
    CUDA_RT_CALL(cudaMalloc(&back_cu, sizeof(real_data_type) * recon_size));
    CUDA_RT_CALL(cudaMalloc(&backcounts_cu, sizeof(real_data_type) * recon_size));
    CUDA_RT_CALL(cudaMalloc(&sino_cu, sizeof(real_data_type) * sino_size));
    CUDA_RT_CALL(cudaMalloc(&ft_recon_cu, sizeof(data_type) * ft_recon_size));
    CUDA_RT_CALL(cudaMalloc(&ft_sino_cu, sizeof(data_type) * ft_sino_size));
    CUDA_RT_CALL(cudaMalloc(&flat_cu, sizeof(float) * nrays*blocksize));
    CUDA_RT_CALL(cudaMalloc(&angles_cu, sizeof(float) * nangles));
    CUDA_RT_CALL(cudaMemcpyAsync(
		angles_cu, angles, sizeof(float) * nangles,
        cudaMemcpyHostToDevice, stream_H2D));
    CUDA_RT_CALL(cudaMemcpyAsync(
		flat_cu, flat, sizeof(float) * nrays*blocksize,
        cudaMemcpyHostToDevice, stream_H2D));
    CUDA_RT_CALL(cudaMemcpyAsync(
		sino_cu, sino, sizeof(float) * sino_size,
        cudaMemcpyHostToDevice, stream_H2D));
    CUDA_RT_CALL(cudaMemcpyAsync(
		recon_cu, recon, sizeof(float) * recon_size,
        cudaMemcpyHostToDevice, stream_H2D));
    CUDA_RT_CALL(cudaStreamSynchronize(stream_H2D));
    // FST initialization finishes here.

    // BST initialization starts here:
    int blocksize_bst = 1;
    int sizeimage = nrays;
    int pad0 = zpad+1;

    cImage cartesianblock_bst(sizeimage, sizeimage*blocksize_bst);
    cImage polarblock_bst(nrays * pad0, nangles*blocksize_bst);
    cImage realpolar_bst(nrays * pad0, nangles*blocksize_bst);
	    
    cufftHandle plan1d_bst;
    cufftHandle plan2d_bst;

    int dimms1d[] = {(int)nrays*pad0/2};
    int dimms2d[] = {(int)sizeimage,(int)sizeimage};
    int beds[] = {nrays*pad0/2};

    HANDLE_FFTERROR( cufftPlanMany(&plan1d_bst, 1, dimms1d, beds, 1, nrays*pad0/2, beds, 1, nrays*pad0/2, CUFFT_C2C, nangles*blocksize_bst*2) );
    HANDLE_FFTERROR( cufftPlanMany(&plan2d_bst, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize_bst) );
    // BST initialization finishes here. 


    BST(backcounts_cu, sino_cu, angles, nrays, nangles, blocksize, nrays, zpad+1);
    calc_reciprocal_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
        backcounts_cu,
        recon_size);

    // set_value<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
    //     recon_cu, 1.0, recon_size);
    // CUDA_RT_CALL(cudaPeekAtLastError());
    // CUDA_RT_CALL(cudaDeviceSynchronize());

    for (int k = 0; k < niter; ++k) {
        // FST:
        CUDA_RT_CALL(cudaMemsetAsync(ft_recon_cu, 0, sizeof(data_type)*ft_recon_size, stream));
        pst_counts_real(
            ft_sino_cu, ft_recon_cu, sino_cu, recon_cu,
            angles_cu, flat_cu,
            plan_2D_forward, plan_1D_inverse,
            nrays, nangles, blocksize,
            zpad, interpolation, scale,
            stream);

        // BST:
        getBST(back_cu, sino_cu, angles_cu, 
	            cartesianblock_bst, polarblock_bst, realpolar_bst,
	            plan1d_bst, plan2d_bst,
	            nrays, nangles, blocksize, blocksize_bst, sizeimage, pad0);

        multiply_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
            back_cu, backcounts_cu, recon_size);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
        
        if (0.0 < tv_param && NITER_MIN_REG < k) {
            total_variation_2d<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
                back_cu, recon_cu, backcounts_cu,
                recon_size, nrays, nrays, blocksize,
                tv_param);
            CUDA_RT_CALL(cudaPeekAtLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize());
        }

        multiply_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
            recon_cu, back_cu, recon_size);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    CUDA_RT_CALL(cudaMemcpyAsync(
        recon, recon_cu, sizeof(real_data_type) * recon_size,
        cudaMemcpyDeviceToHost, stream_D2H));
    CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
        
    CUDA_RT_CALL(cudaHostUnregister(recon));
    CUDA_RT_CALL(cudaHostUnregister(sino));
    free_cuda_counts_real(
        angles_cu, flat_cu, recon_cu, sino_cu,
        ft_recon_cu, ft_sino_cu,
        plan_2D_forward, plan_1D_inverse,
        stream, stream_H2D, stream_D2H);
    CUDA_RT_CALL(cudaFree(back_cu));
    CUDA_RT_CALL(cudaFree(backcounts_cu));
    
    CUFFT_CALL(cufftDestroy(plan1d_bst));
    CUFFT_CALL(cufftDestroy(plan2d_bst));
    
    
//    // Apagar (só coloquei pra ver em qual crasha):
//    cartesianblock_bst.~cImage();
//    polarblock_bst.~cImage();
//    realpolar_bst.~cImage();
    // Apagar (não deveria ser necessário):
    // CUDA_RT_CALL(cudaDeviceReset());
}
}

//----------------------
// EM on frequency Threads Block algorithm
//----------------------

extern "C"{

    void get_tEM_FQ_GPU(float *count, float *recon, float *angles, float *flat,
    int nrays, int nangles, int blockgpu,
    int zpad, int interpolation, float dx, float tv_param,
    int niter, int gpu)
    {
        size_t blocksize_aux = calc_blocksize(blockgpu, nangles, nrays, zpad, true); 
        size_t blocksize = min((size_t)blockgpu, blocksize_aux);

        for (size_t b = 0; b < blockgpu; b += blocksize) {
            blocksize = min(size_t(blockgpu) - b, blocksize);

            _get_tEM_FQ_GPU(count + (size_t)b*nrays*nangles, 
                            recon + (size_t)b*nrays*nrays,
                            angles, flat + (size_t)b*nrays,
                            nrays, nangles, blocksize, zpad, 
                            interpolation, dx, tv_param, niter, gpu);
        }
        cudaDeviceSynchronize();
    }

    void get_tEM_FQ_MultiGPU(int* gpus, int ngpus, 
    float *count, float *recon, float *angles, float *flat,
    int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx, float tv_param,
    int niter)
    {
        int i, Maxgpudev;
		
		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");
            

        int t;
        int blockgpu = (nslices + ngpus - 1) / ngpus;
        
        std::vector<std::future<void>> threads;

        for(t = 0; t < ngpus; t++){ 
            
            blockgpu = min(nslices - blockgpu * t, blockgpu);

            threads.push_back(std::async( std::launch::async, get_tEM_FQ_GPU, 
                count + (size_t)t * blockgpu * nrays * nangles, 
                recon + (size_t)t * blockgpu * nrays * nrays, 
                angles, flat + (size_t)t * blockgpu * nrays, 
                nrays, nangles, blockgpu, zpad, interpolation, 
                dx, tv_param, niter, gpus[t]));
        }

        for(auto& t : threads)
            t.get();
    }
}

// extern "C"{

//     void fast_tEM_FQ(int* gpus, int ngpus, 
//     float *count, float *recon, float *angles, float *flat,
//     float *paramf, int *parami)
//     {
//         // int *ishape,
//         // char *path,
//         // char *outputPath,
//         // char *volOrder,
//         // char *rank,
//         // char *datasetName,
//         // int ngpus,
//         // int *gpu,
//         // int *shape,
//         // int Init,
//         // int Final,
//         // int blockSize,
//         // int timing,
//         // int saving,
//         // int *ix,
//         // int *iy,
//         // int *xmin,
//         // int *xmax,
//         // int *ymin,
//         // int *ymax,
//         // int *center,
//         // int roi,
//         // float *flat,
//         // float *empty,
//         // float *mask,
//         // float *daxpyimg,
//         // float daxpycon,
//         // int susp,
//         // char *uuid,
//         // float *gaps,
//         // int fill
//         // int nrays, int nangles, int nslices, 
//         // int zpad, int interpolation, float dx, float tv_param,
//         // int niter)
//         int i, Maxgpudev;
		
// 		/* Multiples devices */
// 		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

// 		/* If devices input are larger than actual devices on GPU, exit */
// 		for(i = 0; i < ngpus; i++) 
// 			assert(gpus[i] < Maxgpudev && "Invalid device number.");

// 		CFG configs; GPU gpu_parameters;

//         setEMParameters(&configs, paramf, parami);

//         setGPUParameters(&gpu_parameters, configs.tomo.padsize, ngpus, gpus);

//         int t;
//         int blockgpu = (nslices + ngpus - 1) / ngpus;
        
//         std::vector<std::future<void>> threads;

//         for(t = 0; t < ngpus; t++){ 
            
//             blockgpu = min(nslices - blockgpu * t, blockgpu);

//             threads.push_back(std::async( std::launch::async, get_tEM_FQ_GPU, 
//                 count + (size_t)t * blockgpu * nrays * nangles, 
//                 recon + (size_t)t * blockgpu * nrays * nrays, 
//                 angles, flat + (size_t)t * blockgpu * nrays, 
//                 nrays, nangles, blockgpu, zpad, interpolation, 
//                 dx, tv_param, niter, gpus[t]));
//         }

//         for(auto& t : threads)
//             t.get();
//     }
// }