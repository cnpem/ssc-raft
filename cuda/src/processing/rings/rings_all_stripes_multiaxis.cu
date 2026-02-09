#include "processing/rings_all_stripes.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#include <cassert>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
// #include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

extern "C"{
    void remove_all_stripes_transposed(float *tomo, TomoAttributes attribute, int ndevs, int batch) {
        const int length_blocks = std::min(attribute.n_length / ndevs, std::min(attribute.n_length, batch));
        const int n_process = (attribute.n_length + length_blocks - 1) / length_blocks;

        std::vector<std::thread> threads(ndevs);

        for (int k = 0; k < n_process; ++k) {
            const int tid = k % ndevs;
            const int idx = k * length_blocks;
            const int size = std::min(length_blocks, attribute.n_length - idx);

            fprintf(stderr, "Process %d %d %d %d!\n", length_blocks, k, k * length_blocks, size);

            if (threads[tid].joinable()) threads[tid].join();

            threads[tid] = std::thread(_remove_all_stripes_transposed,tomo, attribute, idx, size, tid);
        }

        for (auto &t : threads)
            if (t.joinable()) t.join();
    }

    void _remove_all_stripes_transposed(float *tomo_, TomoAttributes attribute, int start_x, int n_length, int device) {

        float *tomo;
        int *index_sorting;
        float *sorted_data;
        checkCuda(cudaSetDevice(device));

        const size_t tomo_size = n_length * attribute.n_angle * attribute.n_slice;
        // Allocating Memory for Sorting Arrays
        checkCuda(cudaMalloc((void **)&tomo, tomo_size * sizeof(float)));
        checkCuda(cudaMalloc((void **)&sorted_data, tomo_size * sizeof(float)));
        checkCuda(cudaMalloc((void **)&index_sorting, tomo_size * sizeof(int)));

        // Copying Tomography to GPU
        checkCuda(cudaMemcpy2D(tomo, sizeof(float) * n_length, tomo_ + start_x, sizeof(float) * attribute.n_length, n_length * sizeof(float), attribute.n_slice * attribute.n_angle, cudaMemcpyHostToDevice));
        _zyx_to_xyz(sorted_data, tomo, attribute.n_slice, attribute.n_angle, n_length);
        checkCuda(cudaMemcpy(tomo, sorted_data, tomo_size * sizeof(float), cudaMemcpyDeviceToDevice));

        bool *list_mask;
        float *list_data;
        float *sorted_list;

        const size_t mask_size = n_length * attribute.n_slice;

        // Allocating Memory for Masks
        checkCuda(cudaMalloc((void **)&list_data, mask_size * sizeof(double)));
        checkCuda(cudaMalloc((void **)&list_mask, mask_size * sizeof(bool)));
        checkCuda(cudaMalloc((void **)&sorted_list, mask_size * sizeof(double)));

        TomoAttributes trans_attribute = attribute;
        std::swap(trans_attribute.n_length, trans_attribute.n_slice);

        // Kernel CUDA and Operations
        execute_remove_all_stripes(tomo, sorted_data, index_sorting, list_data, sorted_list, list_mask, trans_attribute, n_length);

        _zyx_to_xyz(sorted_data, tomo, n_length, attribute.n_angle, attribute.n_slice);
        checkCuda(cudaMemcpy(tomo, sorted_data, tomo_size * sizeof(float), cudaMemcpyDeviceToDevice));
        checkCuda(cudaMemcpy2D(tomo_ + start_x, sizeof(float) * attribute.n_length, tomo, sizeof(float) * n_length, n_length * sizeof(float), attribute.n_slice * attribute.n_angle, cudaMemcpyDeviceToHost));

        checkCuda(cudaFree(tomo));
        checkCuda(cudaFree(index_sorting));
        checkCuda(cudaFree(sorted_data));
        checkCuda(cudaFree(list_data));
        checkCuda(cudaFree(sorted_list));
        checkCuda(cudaFree(list_mask));
    }
}
__global__ void k_zyx_to_xyz(float* out, const float* in, size_t sizez, size_t sizey, size_t sizex) {
// idz, idy and idx are swapped for faster write
    const size_t idz = blockDim.x * blockIdx.x + threadIdx.x;

    if (idz >= sizez)
        return;

    for (size_t idy = 0; idy < sizey; ++idy) {
        for (size_t idx = 0; idx < sizex; ++idx) {
            const size_t id_in = idz * sizex * sizey + idy * sizex + idx;
            const size_t id_out = idx * sizez * sizey + idy * sizez + idz;
            out[id_out] = in[id_in];
        }
    }
}

void _zyx_to_xyz(float* out, const float* in, size_t sizez, size_t sizey, size_t sizex) {
    const dim3 blocksize {1024, 1, 1};
    const dim3 nblocks {unsigned(sizez / blocksize.x + (sizez % blocksize.x > 0)), 1, 1};
    k_zyx_to_xyz<<<nblocks, blocksize>>>(out, in, sizez, sizey, sizex);
}


