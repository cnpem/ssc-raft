#include "../../../../inc/sscraft.h"
#include "../../../../inc/geometries/parallel/fst.h"
#include "../../../../inc/geometries/parallel/fst_kernel.h"
#include "../../../../inc/geometries/parallel/bst.h"
#include "../../../../inc/common/cufft_utils.h"


extern "C" {

    void emfreq(float *sino, float *recon, float *angles, float *flat,
	int nrays, int nangles, int blocksize, 
    int zpad, int interpolation, float dx,
    int niter, int gpu) {

    cudaSetDevice(gpu);

    cudaStream_t stream, stream_H2D, stream_D2H; // stream is for cuFFT and kernels.
    cufftHandle plan_2D_forward;
	cufftHandle plan_1D_inverse;
    cufftComplex *ft_recon_cu = nullptr; // pointer to 2D Fourier transform of recon on GPU.
	cufftComplex *ft_sino_cu = nullptr;  // pointer to 1D Fourier transform of sino on GPU.
    float *recon_cu = nullptr; // pointer to recon on GPU.
    float *back_cu = nullptr;
	float *sino_cu = nullptr;  // pointer to sino on GPU.
    float *backcounts_cu = nullptr;
    float *angles_cu;
    float *flat_cu;
    std::array<int, FT_PST_RANK_FORWARD> forward_fft_dim = {nrays*(1+zpad), nrays*(1+zpad)};
    std::array<int, FT_PST_RANK_INVERSE> inverse_fft_dim = {nrays*(1+zpad)};
    size_t recon_size = static_cast<size_t>(blocksize * nrays) * nrays;
	size_t sino_size = static_cast<size_t>(blocksize * nrays) * nangles;
    size_t ft_sino_size = static_cast<size_t>(blocksize*nangles)*nrays*(1+zpad);
    size_t ft_recon_size = static_cast<size_t>(blocksize*nrays*(1+zpad))*nrays*(1+zpad);
    float scale = (0 < dx)? dx/(float)inverse_fft_dim[0] : 1.0/(float)inverse_fft_dim[0];

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
        CUFFT_C2C, nangles * blocksize));	    // type, batch.

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
        cudaMemcpyHostToDevice, stream));

        // BST:
        BST(backcounts_cu, sino_cu, nrays, nangles, blocksize, nrays, zpad+1);

        calc_reciprocal_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
            backcounts_cu,
            recon_size);

        set_value<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
            recon_cu, 1.0, recon_size);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaDeviceSynchronize());

        for (int k = 0; k < niter; k++) {
            // FST:
            pst_counts_real(
                ft_sino_cu, ft_recon_cu, sino_cu, recon_cu,
                angles_cu, flat_cu,
                plan_2D_forward, plan_1D_inverse,
                nrays, nangles, blocksize,
                zpad, interpolation, scale,
                stream);

            // BST:
            BST(back_cu, sino_cu, nrays, nangles, blocksize, nrays, zpad+1);

            multiply_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
                back_cu, backcounts_cu, recon_size);
            CUDA_RT_CALL(cudaPeekAtLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize());
            multiply_element_wise<<<recon_size/NUM_THREADS, NUM_THREADS>>>(
                recon_cu, back_cu, recon_size);
            CUDA_RT_CALL(cudaPeekAtLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize());
        }

        CUDA_RT_CALL(cudaMemcpyAsync(
            recon, recon_cu, sizeof(real_data_type) * sino_size,
            cudaMemcpyDeviceToHost, stream_D2H));
        CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
            
        CUDA_RT_CALL(cudaHostUnregister(recon));
        CUDA_RT_CALL(cudaHostUnregister(sino));
        free_cuda_real(
            angles_cu, recon_cu, sino_cu,
            ft_recon_cu, ft_sino_cu,
            plan_2D_forward, plan_1D_inverse,
            stream, stream_H2D, stream_D2H);

    }
}

//----------------------
// EM on frequency Threads Block algorithm
//----------------------

extern "C"{   

  void emfreqgpu(float *count, float *recon, float *angles, float *flat,
	int nrays, int nangles, int blockgpu, 
    int zpad, int interpolation, float dx,
    int niter, int gpu)
  {

      size_t blocksize = min((size_t)blockgpu,32ul);

      for(size_t b = 0; b < blockgpu; b += blocksize){
          blocksize = min(size_t(blockgpu) - b, blocksize);
          // printf("Nslices: %d, blocksize: %ld, Iter: %ld \n", nslices,blocksize,b);

          emfreq(count + (size_t)b*nrays*nangles, recon + (size_t)b*nrays*nrays,  
          flat + (size_t)b*nrays*nangles, angles, 
          nrays, nangles, blocksize, zpad, interpolation, dx, niter, gpu);          
      }

      cudaDeviceSynchronize();

  }

  void emfreqblock(float *count, float *recon, float *angles, float *flat,
	int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx,
    int niter, int ngpus, int* gpus)
  {
      int t;
      int blockgpu = (nslices + ngpus - 1) / ngpus;
      
      std::vector<std::future<void>> threads;

      for(t = 0; t < ngpus; t++){ 
          
          blockgpu = min(nslices - blockgpu * t, blockgpu);

          threads.push_back(std::async( std::launch::async, emfreqgpu, count + (size_t)t * blockgpu * nrays * nangles, 
          recon + (size_t)t * blockgpu * nrays * nrays, 
          flat + (size_t)t * blockgpu * nrays * nangles, angles, 
          nrays, nangles, blockgpu,  zpad, interpolation, dx, niter, gpus[t]
          ));
      }
  
      for(auto& t : threads)
          t.get();
  }

}