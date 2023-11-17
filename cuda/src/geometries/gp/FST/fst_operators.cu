// Author: Otávio Moreira Paiano.

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <iostream>

// in case it is compiled with host compiler instead of nvcc:
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufftXt.h>

// our libs:
#include "../../../../inc/geometries/gp/fst.h"
#include "../../../../inc/geometries/gp/fst_kernel.h"
#include "../../../../inc/common/cufft_utils.h"


#define BYTES_TO_GB (1.0/(1024.0*1024.0*1024.0))
#define A100_MEM 39.5 // A100 40GB device RAM memory, in GB.


// extern "C" {
// int multiply_by_flat_field(
// 	float *sino, float *flat, 
// 	int nrays, int nslices, int nangles,
//     int device)
// {
//     int blocksize = calc_blocksize(
//         static_cast<float>(nslices), 
//         static_cast<float>(nangles), 
//         static_cast<float>(nrays));
//     if (blocksize == 0) {
//         return -1;
//     }
//     float *sino_cu, *flat_cu;
//     int num_of_batches = nslices/blocksize;
//     if (nslices % blocksize != 0) {
//         num_of_batches += 1;
//     }
//     cudaStream_t stream, stream_H2D, stream_D2H; // stream is for cuFFT and kernels.

//     CUDA_RT_CALL(cudaSetDevice(device));
//     CUDA_RT_CALL(cudaStreamCreate(&stream_H2D));
//     CUDA_RT_CALL(cudaStreamCreate(&stream_D2H));
//     CUDA_RT_CALL(cudaStreamCreate(&stream));

//     CUDA_RT_CALL(cudaMalloc(&sino_cu, sizeof(float) * nangles*nslices*nrays));
//     CUDA_RT_CALL(cudaMalloc(&flat_cu, sizeof(float) * nslices*nrays));
//     CUDA_RT_CALL(cudaMemcpyAsync(
// 		flat_cu, flat, sizeof(float) * nslices*nrays,
//         cudaMemcpyHostToDevice, stream));

//     for (int i = 0; i < num_of_batches; ++i) {

//         CUDA_RT_CALL(cudaMemsetAsync(ft_recon_cu, 0, sizeof(data_type)*ft_recon_size, stream));

//         for (size_t idx = 0; idx < recon_size; idx++) {
//             recon_complex[idx] = data_type(recon[i*recon_size + idx], 0);
//         }

//         CUDA_RT_CALL(cudaMemcpyAsync(
//             recon_cu, recon_complex.data(), sizeof(data_type)*recon_size,
//             cudaMemcpyHostToDevice, stream_H2D));

//         CUDA_RT_CALL(cudaStreamSynchronize(stream));
//         CUDA_RT_CALL(cudaStreamSynchronize(stream_H2D));
//         // CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));

//         pst(ft_sino_cu, ft_recon_cu, sino_cu, recon_cu,
//             angles_cu, plan_2D_forward, plan_1D_inverse,
//             nrays, nangles, blocksize, 
//             zpad, interpolation, scale,
//             stream);
//         CUDA_RT_CALL(cudaStreamSynchronize(stream));

//         CUDA_RT_CALL(cudaMemcpyAsync(
//             sino_complex.data(), sino_cu, sizeof(data_type) * sino_complex.size(),
//             cudaMemcpyDeviceToHost, stream_D2H));
//         CUDA_RT_CALL(cudaStreamSynchronize(stream_D2H));
//         for (size_t idx = 0; idx < sino_size; ++idx) {
//             sino[i*sino_size + idx] = sino_complex[idx].real();
//         }

//     printf("\t  Progress: %.2f %%.\n", (100.0*(i+1)*blocksize)/nslices);
//     }

//     free_cuda(
//         angles_cu, recon_cu, sino_cu,
//         ft_recon_cu, ft_sino_cu,
//         plan_2D_forward, plan_1D_inverse,
//         stream, stream_H2D, stream_D2H);

//     return blocksize;
// }
// }


int calc_blocksize(float nslices, float nangles, float nrays) {
    const float float_size = static_cast<float>(sizeof(float)); // float size, in bytes.
    const float slice_size = float_size*std::pow(nrays, 2.0);   // slice size, in bytes.
    const float sino_size = float_size*nrays*nangles;           // sinogram size, in bytes.
    const float total_required_mem_per_slice = float_size*slice_size;  // ou 2*sino_size?
    int raw_blocksize; // biggest blocksize feasible, although not necessarily:
    int blocksize;

    std::cout << "Calculating blocksize..." << std::endl;

    raw_blocksize = static_cast<int>(
        (A100_MEM)/(BYTES_TO_GB*total_required_mem_per_slice) );
    std::cout << "\t  Raw blocksize: " << raw_blocksize << std::endl;

    if (nslices < raw_blocksize) {
        blocksize = nslices;
    } else {
        blocksize = raw_blocksize;
    }
    std::cout << "\t  Blocksize: " << blocksize << std::endl;
    return blocksize;
}