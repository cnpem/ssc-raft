// Author: Otávio Moreira Paiano.

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <complex>
#include <array>
#include <vector>
#include <iostream>

// in case it is compiled with host compiler instead of nvcc:
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufftXt.h>

// our libs:
#include "../../../../inc/geometries/parallel/fst.h"
#include "../../../../inc/geometries/parallel/fst_kernel.h"
#include "../../../../inc/common/cufft_utils.h"

extern "C" {
int fst_gpu(
	float *sino, float *recon, float *angles, 
	int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx,
    float *debug)
{
    int blocksize = calc_blocksize(
        static_cast<float>(nslices), 
        static_cast<float>(nangles), 
        static_cast<float>(nrays), 
        static_cast<float>(zpad), 
        true);
    if (blocksize == 0) {
        return -1;
    }
    int num_of_batches = nslices/blocksize;
    cudaStream_t stream, stream_H2D, stream_D2H; // stream is for cuFFT and kernels.
    cufftHandle plan_2D_forward;
	cufftHandle plan_1D_inverse;
    cufftComplex *ft_recon_cu = nullptr; // pointer to 2D Fourier transform of recon on GPU.
	cufftComplex *ft_sino_cu = nullptr;  // pointer to 1D Fourier transform of sino on GPU.
    cufftComplex *recon_cu = nullptr; // pointer to recon on GPU.
	cufftComplex *sino_cu = nullptr;  // pointer to sino on GPU.
    float *angles_cu;
    std::array<int, FT_PST_RANK_FORWARD> forward_fft_dim = {nrays*(1+zpad), nrays*(1+zpad)};
    std::array<int, FT_PST_RANK_INVERSE> inverse_fft_dim = {nrays*(1+zpad)};
    size_t recon_size = static_cast<size_t>(blocksize * nrays) * nrays;
	size_t sino_size = static_cast<size_t>(blocksize * nrays) * nangles;
    size_t ft_sino_size = static_cast<size_t>(blocksize*nangles)*nrays*(1+zpad);
    size_t ft_recon_size = static_cast<size_t>(blocksize*nrays*(1+zpad))*nrays*(1+zpad);
    std::vector<data_type> recon_complex(recon_size);
    std::vector<data_type> sino_complex(sino_size);
    float scale = (0 < dx)? dx/(float)inverse_fft_dim[0] : 1.0/(float)inverse_fft_dim[0];

#ifdef DEBUG
std::vector<data_type> debug_complex(ft_recon_size);
bool debug_true;
if (-0.5 < debug[0]) {
    debug_true = true;
} else {
    debug_true = false;
}
#endif

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

    CUDA_RT_CALL(cudaMalloc(&recon_cu, sizeof(data_type) * recon_complex.size()));
    CUDA_RT_CALL(cudaMalloc(&sino_cu, sizeof(data_type) * sino_complex.size()));
    CUDA_RT_CALL(cudaMalloc(&ft_recon_cu, sizeof(data_type) * ft_recon_size));
    CUDA_RT_CALL(cudaMalloc(&ft_sino_cu, sizeof(data_type) * ft_sino_size));
    CUDA_RT_CALL(cudaMalloc(&angles_cu, sizeof(float) * nangles));
    CUDA_RT_CALL(cudaMemcpyAsync(
		angles_cu, angles, sizeof(float) * nangles,
        cudaMemcpyHostToDevice, stream));

    std::cout << "Computing sinograms..." << std::endl;
    for (int i = 0; i < num_of_batches; ++i) {

        CUDA_RT_CALL(cudaMemsetAsync(ft_recon_cu, 0, sizeof(data_type)*ft_recon_size, stream));

        for (size_t idx = 0; idx < recon_size; idx++) {
            recon_complex[idx] = data_type(recon[i*recon_size + idx], 0);
        }

        CUDA_RT_CALL(cudaMemcpyAsync(
            recon_cu, recon_complex.data(), sizeof(data_type)*recon_size,
            cudaMemcpyHostToDevice, stream_H2D));

        CUDA_RT_CALL(cudaStreamSynchronize(stream));
        CUDA_RT_CALL(cudaStreamSynchronize(stream_H2D));
        // CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));

        pst(ft_sino_cu, ft_recon_cu, sino_cu, recon_cu,
            angles_cu, plan_2D_forward, plan_1D_inverse,
            nrays, nangles, blocksize, 
            zpad, interpolation, scale,
            stream);
        CUDA_RT_CALL(cudaStreamSynchronize(stream));

#ifdef DEBUG
if (debug_true) {
CUDA_RT_CALL(cudaMemcpyAsync(
    debug_complex.data(), ft_recon_cu, sizeof(data_type) * debug_complex.size(),
    cudaMemcpyDeviceToHost, stream_D2H));
CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
std::cout << "\t  debug copied from device!" <<  std::endl;
for (size_t idx = 0; idx < debug_complex.size(); ++idx) {
    debug[i*debug_complex.size() + idx] = debug_complex[idx].real();
}
}
#endif

        CUDA_RT_CALL(cudaMemcpyAsync(
            sino_complex.data(), sino_cu, sizeof(data_type) * sino_complex.size(),
            cudaMemcpyDeviceToHost, stream_D2H));
        CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
        for (size_t idx = 0; idx < sino_size; ++idx) {
            sino[i*sino_size + idx] = sino_complex[idx].real();
        }

    printf("\t  Progress: %.2f %%.\n", (100.0*(i+1)*blocksize)/nslices);
    }

    free_cuda(
        angles_cu, recon_cu, sino_cu,
        ft_recon_cu, ft_sino_cu,
        plan_2D_forward, plan_1D_inverse,
        stream, stream_H2D, stream_D2H);

    return blocksize;
}
}


// Same as above but for real valued sinogram and recon.
extern "C" {
int real_fst_gpu(
	float *sino, float *recon, float *angles, 
	int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx,
    float *debug)
{
    int blocksize = calc_blocksize(
        static_cast<float>(nslices), 
        static_cast<float>(nangles), 
        static_cast<float>(nrays), 
        static_cast<float>(zpad), 
        true);
    if (blocksize == 0) {
        return -1;
    }
    int num_of_batches = nslices/blocksize;
    cudaStream_t stream, stream_H2D, stream_D2H; // stream is for cuFFT and kernels.
    cufftHandle plan_2D_forward;
	cufftHandle plan_1D_inverse;
    cufftComplex *ft_recon_cu = nullptr; // pointer to 2D Fourier transform of recon on GPU.
	cufftComplex *ft_sino_cu = nullptr;  // pointer to 1D Fourier transform of sino on GPU.
    float *recon_cu = nullptr; // pointer to recon on GPU.
	float *sino_cu = nullptr;  // pointer to sino on GPU.
    float *angles_cu;
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

    CUDA_RT_CALL(cudaMalloc(&recon_cu, sizeof(real_data_type) * recon_size));
    CUDA_RT_CALL(cudaMalloc(&sino_cu, sizeof(real_data_type) * sino_size));
    CUDA_RT_CALL(cudaMalloc(&ft_recon_cu, sizeof(data_type) * ft_recon_size));
    CUDA_RT_CALL(cudaMalloc(&ft_sino_cu, sizeof(data_type) * ft_sino_size));
    CUDA_RT_CALL(cudaMalloc(&angles_cu, sizeof(float) * nangles));
    CUDA_RT_CALL(cudaMemcpyAsync(
		angles_cu, angles, sizeof(float) * nangles,
        cudaMemcpyHostToDevice, stream));

    std::cout << "Computing sinograms..." << std::endl;
    for (int i = 0; i < num_of_batches; ++i) {

        CUDA_RT_CALL(cudaMemsetAsync(ft_recon_cu, 0, sizeof(data_type)*ft_recon_size, stream));

        CUDA_RT_CALL(cudaMemcpyAsync(
            recon_cu, &recon[i*recon_size], sizeof(real_data_type)*recon_size,
            cudaMemcpyHostToDevice, stream_H2D));

        CUDA_RT_CALL(cudaStreamSynchronize(stream));
        CUDA_RT_CALL(cudaStreamSynchronize(stream_H2D));
        // CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));

        pst_real(ft_sino_cu, ft_recon_cu, sino_cu, recon_cu,
            angles_cu, plan_2D_forward, plan_1D_inverse,
            nrays, nangles, blocksize, 
            zpad, interpolation, scale,
            stream);
        CUDA_RT_CALL(cudaStreamSynchronize(stream));

        CUDA_RT_CALL(cudaMemcpyAsync(
            &sino[i*sino_size], sino_cu, sizeof(real_data_type) * sino_size,
            cudaMemcpyDeviceToHost, stream_D2H));
        CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));

    printf("\t  Progress: %.2f %%.\n", (100.0*(i+1)*blocksize)/nslices);
    }

    free_cuda_real(
        angles_cu, recon_cu, sino_cu,
        ft_recon_cu, ft_sino_cu,
        plan_2D_forward, plan_1D_inverse,
        stream, stream_H2D, stream_D2H);

    return blocksize;
}
}


// Same as above but returning counts = flat*exp(-tomo) instead of tomo.
extern "C" {
int counts_real_fst_gpu(
	float *sino, float *recon, float *angles, float *flat,
	int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx,
    float *debug)
{
    int blocksize = calc_blocksize(
        static_cast<float>(nslices), 
        static_cast<float>(nangles), 
        static_cast<float>(nrays), 
        static_cast<float>(zpad), 
        true);
    if (blocksize == 0) {
        return -1;
    }
    int num_of_batches = nslices/blocksize;
    cudaStream_t stream, stream_H2D, stream_D2H; // stream is for cuFFT and kernels.
    cufftHandle plan_2D_forward;
	cufftHandle plan_1D_inverse;
    cufftComplex *ft_recon_cu = nullptr; // pointer to 2D Fourier transform of recon on GPU.
	cufftComplex *ft_sino_cu = nullptr;  // pointer to 1D Fourier transform of sino on GPU.
    float *recon_cu = nullptr; // pointer to recon on GPU.
	float *sino_cu = nullptr;  // pointer to sino on GPU.
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
        sizeof(real_data_type) * static_cast<size_t>(nslices) * nrays * nrays,
        cudaHostRegisterDefault));
    CUDA_RT_CALL(cudaHostRegister(
        sino,
        sizeof(real_data_type) * static_cast<size_t>(nslices) * nangles * nrays,
        cudaHostRegisterDefault));

    CUDA_RT_CALL(cudaMalloc(&recon_cu, sizeof(real_data_type) * recon_size));
    CUDA_RT_CALL(cudaMalloc(&sino_cu, sizeof(real_data_type) * sino_size));
    CUDA_RT_CALL(cudaMalloc(&ft_recon_cu, sizeof(data_type) * ft_recon_size));
    CUDA_RT_CALL(cudaMalloc(&ft_sino_cu, sizeof(data_type) * ft_sino_size));
    CUDA_RT_CALL(cudaMalloc(&flat_cu, sizeof(float) * nrays*nslices));
    CUDA_RT_CALL(cudaMalloc(&angles_cu, sizeof(float) * nangles));
    CUDA_RT_CALL(cudaMemcpyAsync(
		angles_cu, angles, sizeof(float) * nangles,
        cudaMemcpyHostToDevice, stream));
    // CUDA_RT_CALL(cudaMemcpyAsync(
	// 	flat_cu, flat, sizeof(float) * nrays*nslices,
    //     cudaMemcpyHostToDevice, stream));

    std::cout << "Computing sinograms..." << std::endl;
    for (int i = 0; i < num_of_batches; ++i) {

        CUDA_RT_CALL(cudaMemsetAsync(ft_recon_cu, 0, sizeof(data_type)*ft_recon_size, stream));

        CUDA_RT_CALL(cudaMemcpyAsync(
            recon_cu, &recon[i*recon_size], sizeof(real_data_type)*recon_size,
            cudaMemcpyHostToDevice, stream_H2D));

        CUDA_RT_CALL(cudaMemcpyAsync(
            flat_cu, &flat[i*blocksize*nrays], sizeof(real_data_type)*blocksize*nrays,
            cudaMemcpyHostToDevice, stream_H2D));

        CUDA_RT_CALL(cudaStreamSynchronize(stream));
        CUDA_RT_CALL(cudaStreamSynchronize(stream_H2D));
        // CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));

        pst_counts_real(
            ft_sino_cu, ft_recon_cu, sino_cu, recon_cu,
            angles_cu, flat_cu,
            plan_2D_forward, plan_1D_inverse,
            nrays, nangles, blocksize,
            zpad, interpolation, scale,
            stream);
        CUDA_RT_CALL(cudaStreamSynchronize(stream));

        CUDA_RT_CALL(cudaMemcpyAsync(
            &sino[i*sino_size], sino_cu, sizeof(real_data_type) * sino_size,
            cudaMemcpyDeviceToHost, stream_D2H));
        CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));

    printf("\t  Progress: %.2f %%.\n", (100.0*(i+1)*blocksize)/nslices);
    }

    CUDA_RT_CALL(cudaHostUnregister(recon));
    CUDA_RT_CALL(cudaHostUnregister(sino));
    free_cuda_real(
        angles_cu, recon_cu, sino_cu,
        ft_recon_cu, ft_sino_cu,
        plan_2D_forward, plan_1D_inverse,
        stream, stream_H2D, stream_D2H);

    return blocksize;
}
}


void pst(
    cufftComplex *ft_sino_cu, cufftComplex *ft_recon_cu, cufftComplex *sino_cu, cufftComplex *recon_cu, 
    float *angles_cu, cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float scale,
    cudaStream_t stream)
{
    int slice_fft_size = nrays*(1+zpad);    // size of the 2d fft.
    int sinogram_fft_size = nrays*(1+zpad); // size of the 1d fft.
    size_t num_blocks = static_cast<size_t>(nrays*(1+zpad))*nangles / NUM_THREADS;

    fftshift2d<<<static_cast<size_t>(nrays)*nrays*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
        recon_cu,
        nrays, nrays, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    copy_to_fft_workspace<<<static_cast<size_t>(nrays)*nrays*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
	    ft_recon_cu, recon_cu,
	    slice_fft_size, slice_fft_size,
        nrays, nrays, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    CUFFT_CALL(cufftExecC2C(plan_2D_forward, ft_recon_cu, ft_recon_cu, CUFFT_FORWARD));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    fftshift2d<<<static_cast<size_t>(slice_fft_size)*slice_fft_size*nslices/(1*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_recon_cu,
        slice_fft_size, slice_fft_size, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

#ifdef DEBUG
    std::cout << "\t  slice_fft_size: " << slice_fft_size << std::endl;
    std::cout << "\t  nrays: " << nrays << std::endl;
    std::cout << "\t  nangles: " << nangles << std::endl;
    std::cout << "\t  cuda grid size (num of blocks) for 'cartesian2polar_nn' kernel: " << num_blocks << std::endl;
    std::cout << "\t  cuda block size (num of threads) for 'cartesian2polar_nn' kernel: " << NUM_THREADS << std::endl;
#endif

    if (interpolation == 0) {
        cartesian2polar_nn<<<num_blocks, NUM_THREADS, 0, stream>>>(
            ft_recon_cu, ft_sino_cu, angles_cu,
            sinogram_fft_size, nangles, nslices);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    } else if (interpolation == 1) {
        cartesian2polar_bi_v2<<<num_blocks, NUM_THREADS, 0, stream>>>(
            ft_recon_cu, ft_sino_cu, angles_cu,
            sinogram_fft_size, nangles, nslices);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    }

    shift_1d<<<static_cast<size_t>(sinogram_fft_size*nangles)*nslices/(2*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_sino_cu,
        sinogram_fft_size, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

	CUFFT_CALL(cufftExecC2C(plan_1D_inverse, ft_sino_cu, ft_sino_cu, CUFFT_INVERSE));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    shift_1d<<<static_cast<size_t>(sinogram_fft_size)*nangles*nslices/(2*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_sino_cu,
        sinogram_fft_size, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    copy_from_fft_workspace<<<static_cast<size_t>(nrays)*nangles*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
	    ft_sino_cu, sino_cu,
	    sinogram_fft_size, nangles,
        nrays, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    scale_data_real_only<<<static_cast<size_t>(nrays)*nangles*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
        sino_cu, static_cast<size_t>(nrays)*nangles*nslices, scale);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
}

// Same as pst but for real valued sample only:
void pst_real(
    cufftComplex *ft_sino_cu, cufftComplex *ft_recon_cu, float *sino_cu, float *recon_cu, 
    float *angles_cu, cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float scale,
    cudaStream_t stream)
{
    int slice_fft_size = nrays*(1+zpad);    // size of the 2d fft.
    int sinogram_fft_size = nrays*(1+zpad); // size of the 1d fft.
    size_t num_blocks = static_cast<size_t>(nrays*(1+zpad))*nangles / NUM_THREADS;

    fftshift2d_real<<<static_cast<size_t>(nrays)*nrays*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
        recon_cu,
        nrays, nrays, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    copy_to_fft_workspace_R2C<<<static_cast<size_t>(nrays)*nrays*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
	    ft_recon_cu, recon_cu,
	    slice_fft_size, slice_fft_size,
        nrays, nrays, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    CUFFT_CALL(cufftExecC2C(plan_2D_forward, ft_recon_cu, ft_recon_cu, CUFFT_FORWARD));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    fftshift2d<<<static_cast<size_t>(slice_fft_size)*slice_fft_size*nslices/(1*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_recon_cu,
        slice_fft_size, slice_fft_size, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

#ifdef DEBUG
    std::cout << "\t  slice_fft_size: " << slice_fft_size << std::endl;
    std::cout << "\t  nrays: " << nrays << std::endl;
    std::cout << "\t  nangles: " << nangles << std::endl;
    std::cout << "\t  cuda grid size (num of blocks) for 'cartesian2polar_nn' kernel: " << num_blocks << std::endl;
    std::cout << "\t  cuda block size (num of threads) for 'cartesian2polar_nn' kernel: " << NUM_THREADS << std::endl;
#endif

    if (interpolation == 0) {
        cartesian2polar_nn<<<num_blocks, NUM_THREADS, 0, stream>>>(
            ft_recon_cu, ft_sino_cu, angles_cu,
            sinogram_fft_size, nangles, nslices);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    } else if (interpolation == 1) {
        cartesian2polar_bi_v2<<<num_blocks, NUM_THREADS, 0, stream>>>(
            ft_recon_cu, ft_sino_cu, angles_cu,
            sinogram_fft_size, nangles, nslices);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    }

    shift_1d<<<static_cast<size_t>(sinogram_fft_size*nangles)*nslices/(2*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_sino_cu,
        sinogram_fft_size, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

	CUFFT_CALL(cufftExecC2C(plan_1D_inverse, ft_sino_cu, ft_sino_cu, CUFFT_INVERSE));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    shift_1d<<<static_cast<size_t>(sinogram_fft_size)*nangles*nslices/(2*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_sino_cu,
        sinogram_fft_size, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    copy_from_fft_workspace_C2R<<<static_cast<size_t>(nrays)*nangles*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
	    ft_sino_cu, sino_cu,
	    sinogram_fft_size, nangles,
        nrays, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    scale_data<<<static_cast<size_t>(nrays)*nangles*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
        sino_cu, static_cast<size_t>(nrays)*nangles*nslices, scale);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
}


// Same as pst but for counts = flat*exp(-tomo) instead of tomo.
void pst_counts_real(
    cufftComplex *ft_sino_cu, cufftComplex *ft_recon_cu, float *sino_cu, float *recon_cu, 
    float *angles_cu, float *flat_cu, cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float scale,
    cudaStream_t stream)
{
    int slice_fft_size = nrays*(1+zpad);    // size of the 2d fft.
    int sinogram_fft_size = nrays*(1+zpad); // size of the 1d fft.
    size_t num_blocks = static_cast<size_t>(nrays*(1+zpad))*nangles / NUM_THREADS;

    fftshift2d_real<<<static_cast<size_t>(nrays)*nrays*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
        recon_cu,
        nrays, nrays, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    copy_to_fft_workspace_R2C<<<static_cast<size_t>(nrays)*nrays*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
	    ft_recon_cu, recon_cu,
	    slice_fft_size, slice_fft_size,
        nrays, nrays, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    CUFFT_CALL(cufftExecC2C(plan_2D_forward, ft_recon_cu, ft_recon_cu, CUFFT_FORWARD));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    fftshift2d<<<static_cast<size_t>(slice_fft_size)*slice_fft_size*nslices/(1*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_recon_cu,
        slice_fft_size, slice_fft_size, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

#ifdef DEBUG
    std::cout << "\t  slice_fft_size: " << slice_fft_size << std::endl;
    std::cout << "\t  nrays: " << nrays << std::endl;
    std::cout << "\t  nangles: " << nangles << std::endl;
    std::cout << "\t  cuda grid size (num of blocks) for 'cartesian2polar_nn' kernel: " << num_blocks << std::endl;
    std::cout << "\t  cuda block size (num of threads) for 'cartesian2polar_nn' kernel: " << NUM_THREADS << std::endl;
#endif

    if (interpolation == 0) {
        cartesian2polar_nn<<<num_blocks, NUM_THREADS, 0, stream>>>(
            ft_recon_cu, ft_sino_cu, angles_cu,
            sinogram_fft_size, nangles, nslices);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    } else if (interpolation == 1) {
        cartesian2polar_bi_v2<<<num_blocks, NUM_THREADS, 0, stream>>>(
            ft_recon_cu, ft_sino_cu, angles_cu,
            sinogram_fft_size, nangles, nslices);
        CUDA_RT_CALL(cudaPeekAtLastError());
        CUDA_RT_CALL(cudaStreamSynchronize(stream));
    }

    shift_1d<<<static_cast<size_t>(sinogram_fft_size*nangles)*nslices/(2*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_sino_cu,
        sinogram_fft_size, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

	CUFFT_CALL(cufftExecC2C(plan_1D_inverse, ft_sino_cu, ft_sino_cu, CUFFT_INVERSE));
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    shift_1d<<<static_cast<size_t>(sinogram_fft_size)*nangles*nslices/(2*NUM_THREADS), NUM_THREADS, 0, stream>>>(
        ft_sino_cu,
        sinogram_fft_size, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    copy_from_fft_workspace_C2R<<<static_cast<size_t>(nrays)*nangles*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
	    ft_sino_cu, sino_cu,
	    sinogram_fft_size, nangles,
        nrays, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    scale_data<<<static_cast<size_t>(nrays)*nangles*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
        sino_cu, static_cast<size_t>(nrays)*nangles*nslices, scale);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));

    calc_counts<<<static_cast<size_t>(nrays)*nangles*nslices/NUM_THREADS, NUM_THREADS, 0, stream>>>(
        sino_cu, flat_cu,
        nrays, nangles, nslices);
    CUDA_RT_CALL(cudaPeekAtLastError());
    CUDA_RT_CALL(cudaStreamSynchronize(stream));
}


int malloc_cuda() {
    return 0;
}

int plan_ffts_cuda() {
    return 0;
}

void free_cuda(
    float *angles_cu, cufftComplex *recon_cu, cufftComplex *sino_cu,
    cufftComplex *ft_recon_cu, cufftComplex *ft_sino_cu,
    cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    cudaStream_t stream, cudaStream_t stream_H2D, cudaStream_t stream_D2H)
{
    CUDA_RT_CALL(cudaFree(angles_cu));
    CUDA_RT_CALL(cudaFree(recon_cu));
	CUDA_RT_CALL(cudaFree(sino_cu));
    CUDA_RT_CALL(cudaFree(ft_recon_cu));
	CUDA_RT_CALL(cudaFree(ft_sino_cu));
    CUFFT_CALL(cufftDestroy(plan_2D_forward));
	CUFFT_CALL(cufftDestroy(plan_1D_inverse));
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    CUDA_RT_CALL(cudaStreamDestroy(stream_H2D));
    CUDA_RT_CALL(cudaStreamDestroy(stream_D2H));
    // CUDA_RT_CALL(cudaDeviceReset());
}

// Same as above but for real valued sinogram and recon.
void free_cuda_real(
    float *angles_cu, float *recon_cu, float *sino_cu,
    cufftComplex *ft_recon_cu, cufftComplex *ft_sino_cu,
    cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    cudaStream_t stream, cudaStream_t stream_H2D, cudaStream_t stream_D2H)
{
    CUDA_RT_CALL(cudaFree(angles_cu));
    CUDA_RT_CALL(cudaFree(recon_cu));
	CUDA_RT_CALL(cudaFree(sino_cu));
    CUDA_RT_CALL(cudaFree(ft_recon_cu));
	CUDA_RT_CALL(cudaFree(ft_sino_cu));
    CUFFT_CALL(cufftDestroy(plan_2D_forward));
	CUFFT_CALL(cufftDestroy(plan_1D_inverse));
    CUDA_RT_CALL(cudaStreamDestroy(stream));
    CUDA_RT_CALL(cudaStreamDestroy(stream_H2D));
    CUDA_RT_CALL(cudaStreamDestroy(stream_D2H));
    // CUDA_RT_CALL(cudaDeviceReset());
}


int calc_blocksize(float nslices, float nangles, float nrays, float zpad, bool complex_data) {
    float complex_or_real = complex_data? 2.0 : 1.0;
    const float float_size = static_cast<float>(sizeof(float)); // float size, in bytes.
    const float slice_size = float_size*std::pow(nrays, 2.0);                // slice size, in bytes.
    const float sino_size = float_size*nrays*nangles;                        // sinogram size, in bytes.
    const float fft_slice_size = float_size*std::pow(nrays*(zpad+1.0), 2.0); // ft slice size, in bytes.
    const float fft_sino_size = float_size * nrays*(zpad+1.0)*nangles;       // ft sinogram size, in bytes.
    const float total_required_mem_per_slice = complex_or_real * ( // in bytes.
        slice_size     +
        sino_size      +
        fft_slice_size +
        fft_sino_size);
    const float empiric_const = 2.0; // the GPU needs some free memory to perform the FFTs.
    const float epsilon = 0.0;       // how much free memory we want to leave, in GB.
    
    // the values permitted for blocksize are powers of two.
    int raw_blocksize; // biggest blocksize feasible, although not necessarily: 
        // 1) a power of two; and 
        // 2) not a divisor of nslices (i.e., nslices % raw_blocksize != 0).
    int blocksize_exp = 1; // to store which power of 2 will be used. 
    int blocksize;

    std::cout << "Calculating blocksize..." << std::endl;

    raw_blocksize = static_cast<int>(
        -epsilon + (A100_MEM)/(BYTES_TO_GB*total_required_mem_per_slice) );
    raw_blocksize = raw_blocksize/empiric_const;
    std::cout << "\t  Raw blocksize: " << raw_blocksize << std::endl;

    if (nslices < raw_blocksize) {
        blocksize = nslices;
    } else {
        while (raw_blocksize >> blocksize_exp) {
            blocksize_exp++;
        }
        blocksize_exp--;
        blocksize = 1 << blocksize_exp;
    }
    std::cout << "\t  Blocksize: " << blocksize << std::endl;
    return blocksize;
}


extern "C" {
int el_wise_mult(
	float *arr1, float *arr2,
	int n1, int n2, int n3)
{
    size_t size = static_cast<size_t>(n1)*n2*n3;
    float *arr1_cu, *arr2_cu;

    CUDA_RT_CALL(cudaMalloc(&arr1_cu, sizeof(float)*size));
    CUDA_RT_CALL(cudaMalloc(&arr2_cu, sizeof(float)*size));
    CUDA_RT_CALL(cudaMemcpyAsync(
		arr1_cu, arr1, sizeof(float)*size,
        cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpyAsync(
		arr2_cu, arr2, sizeof(float)*size,
        cudaMemcpyHostToDevice));

    multiply_element_wise<<<size/NUM_THREADS, NUM_THREADS>>>(
        arr1_cu, arr2_cu, size);

    CUDA_RT_CALL(cudaMemcpyAsync(
		arr1, arr1_cu, sizeof(float)*size,
        cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpyAsync(
		arr2, arr2_cu, sizeof(float)*size,
        cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaFree(arr1_cu));
    CUDA_RT_CALL(cudaFree(arr2_cu));

    return 0;
}
}