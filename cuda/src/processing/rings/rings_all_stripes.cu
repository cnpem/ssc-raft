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
    void remove_all_stripes(float *tomo, TomoAttributes attribute, int ndevs, int batch) {

        int slice_blocks = std::min(attribute.n_slice / ndevs, std::min(attribute.n_slice, batch));

        // Limit the number of slices per block so that all device buffers fit in the GPU memory.
        // Per element: tomo (4B) + sorted_data (4B) + index_sorting (4B) + mergesort temporaries (4B + 4B)
        size_t free_mem = 0, total_mem = 0;
        checkCuda(cudaMemGetInfo(&free_mem, &total_mem));
        const size_t bytes_per_slice = (size_t)attribute.n_angle * (size_t)attribute.n_length * (5 * sizeof(float))
                                     + (size_t)attribute.n_length * (2 * sizeof(double) + sizeof(bool));
        const size_t max_slices = std::max((size_t)1, (size_t)((double)free_mem * 0.85 / (double)bytes_per_slice));
        if ((size_t)slice_blocks > max_slices) {
            fprintf(stderr, "Reducing batch from %d to %zu slices to fit in GPU memory (%zu MB free).\n", slice_blocks, max_slices, free_mem >> 20);
            slice_blocks = (int)max_slices;
        }

        int n_process = (attribute.n_slice + slice_blocks - 1) / slice_blocks;

        std::vector<std::thread> threads_all_stripes;
        threads_all_stripes.reserve(ndevs);

        for (int k = 0; k < n_process; ++k) {
            size_t idx = (size_t)k * (size_t)slice_blocks * (size_t)attribute.n_angle * (size_t)attribute.n_length;
            int size = (k == n_process - 1) ? (attribute.n_slice % slice_blocks ? attribute.n_slice % slice_blocks : slice_blocks): slice_blocks;

            fprintf(stderr, "Process %d %d %d %d!\n", slice_blocks, k, k * slice_blocks, size);

            threads_all_stripes.emplace_back(std::thread(_remove_all_stripes, tomo + idx, attribute, size, k % ndevs));

            if (threads_all_stripes.size() == ndevs || k == n_process - 1) {
                for (auto &thread : threads_all_stripes) thread.join();
                threads_all_stripes.clear();
            }
    }
    }

    void _remove_all_stripes(float *tomo_, TomoAttributes attribute, int n_slices, int device) {

        float *tomo;
        int *index_sorting;
        float *sorted_data;
        checkCuda(cudaSetDevice(device));

        size_t n = (size_t)n_slices * (size_t)attribute.n_angle * (size_t)attribute.n_length;
        // Allocating Memory for Sorting Arrays
        checkCuda(cudaMalloc((void **)&tomo, n * sizeof(float)));
        checkCuda(cudaMalloc((void **)&sorted_data, n * sizeof(float)));
        checkCuda(cudaMalloc((void **)&index_sorting, n * sizeof(int)));

        // Copying Tomography to GPU
        checkCuda(cudaMemcpy(tomo, tomo_, n * sizeof(float), cudaMemcpyHostToDevice));
        checkCuda(cudaMemcpy(sorted_data, tomo_, n * sizeof(float), cudaMemcpyHostToDevice));

        bool *list_mask;
        float *list_data;
        float *sorted_list;

        n = (size_t)n_slices * (size_t)attribute.n_length;

        // Allocating Memory for Masks
        checkCuda(cudaMalloc((void **)&list_data, n * sizeof(double)));
        checkCuda(cudaMalloc((void **)&list_mask, n * sizeof(bool)));
        checkCuda(cudaMalloc((void **)&sorted_list, n * sizeof(double)));

        // Kernel CUDA and Operations
        execute_remove_all_stripes(tomo, sorted_data, index_sorting, list_data, sorted_list, list_mask, attribute, n_slices);

        n = (size_t)n_slices * (size_t)attribute.n_angle * (size_t)attribute.n_length;
        checkCuda(cudaMemcpy(tomo_, tomo, n * sizeof(float), cudaMemcpyDeviceToHost));

        checkCuda(cudaFree(tomo));
        checkCuda(cudaFree(index_sorting));
        checkCuda(cudaFree(sorted_data));
        checkCuda(cudaFree(list_data));
        checkCuda(cudaFree(sorted_list));
        checkCuda(cudaFree(list_mask));
    }

    void get_remove_all_stripes(float *tomo, TomoAttributes attribute, int nslices, int nangles, int nrays)
    {
        // tomo is already on the device
        int *index_sorting;
        float *sorted_data;

        const size_t plane = (size_t)nangles * (size_t)nrays;

        // Process the volume in chunks of slices that fit in the free GPU memory.
        // Per element: sorted_data (4B) + index_sorting (4B) + mergesort temporaries (4B + 4B)
        size_t free_mem = 0, total_mem = 0;
        checkCuda(cudaMemGetInfo(&free_mem, &total_mem));
        const size_t bytes_per_slice = plane * (3 * sizeof(float) + sizeof(int)) + (size_t)nrays * (2 * sizeof(double) + sizeof(bool));
        size_t chunk = (size_t)((double)free_mem * 0.7 / (double)bytes_per_slice);
        if (chunk < 1) chunk = 1;
        if (chunk > (size_t)nslices) chunk = (size_t)nslices;

        size_t n = chunk * plane;
        // Allocating Memory for Sorting Arrays
        checkCuda(cudaMalloc((void **)&sorted_data, n * sizeof(float)));
        checkCuda(cudaMalloc((void **)&index_sorting, n * sizeof(int)));

        bool *list_mask;
        float *list_data;
        float *sorted_list;

        n = chunk * (size_t)nrays;

        // Allocating Memory for Masks
        checkCuda(cudaMalloc((void **)&list_data, n * sizeof(double)));
        checkCuda(cudaMalloc((void **)&list_mask, n * sizeof(bool)));
        checkCuda(cudaMalloc((void **)&sorted_list, n * sizeof(double)));

        for (size_t s = 0; s < (size_t)nslices; s += chunk) {
            const int cur = (int)std::min(chunk, (size_t)nslices - s);
            float *tomo_chunk = tomo + s * plane;

            checkCuda(cudaMemcpy(sorted_data, tomo_chunk, (size_t)cur * plane * sizeof(float), cudaMemcpyDeviceToDevice));

            // Kernel CUDA and Operations
            execute_remove_all_stripes(tomo_chunk, sorted_data, index_sorting, list_data, sorted_list, list_mask, attribute, cur);
        }
        checkCuda(cudaDeviceSynchronize());

        checkCuda(cudaFree(index_sorting));
        checkCuda(cudaFree(sorted_data));
        checkCuda(cudaFree(list_data));
        checkCuda(cudaFree(sorted_list));
        checkCuda(cudaFree(list_mask));
    }
}

void execute_remove_all_stripes(float *tomo, float *sorted_data, int *index_sorting, float *list_data, float *sorted_list, bool *list_mask, TomoAttributes attribute, int n_slices) {

    // fprintf(stderr, "Unresponsive and Fluctuating Stripes!\n");
    // Remove unresponsive or fluctuating stripes
    execute_remove_unresponsive_and_fluctuating_stripes(tomo, sorted_data, index_sorting, list_data, sorted_list, list_mask, attribute, n_slices);

    // fprintf(stderr, "Large Stripes!\n");
    // Remove Large Stripes
    execute_remove_large_stripes(tomo, sorted_data, index_sorting, list_data, sorted_list, list_mask, attribute, n_slices);

    // fprintf(stderr, "Sorting Stripes!\n");
    // Remove Stripes Based on Sorting
    execute_remove_based_sorting(tomo, sorted_data, index_sorting, list_data, sorted_list, list_mask, attribute, n_slices);

}

void execute_remove_based_sorting(float *tomo, float *sorted_data, int *index_sorting, float *list_data, float *sorted_list, bool *list_mask, TomoAttributes attribute, int n_slices) {

    batch_mergesort_over_angles(tomo, index_sorting, n_slices, attribute.n_angle, attribute.n_length);

    median_filter_over_length(sorted_data, tomo, attribute.size_median, n_slices, attribute.n_angle, attribute.n_length);

    sino_reorder_from_sort(tomo, sorted_data, index_sorting, n_slices, attribute.n_angle, attribute.n_length);
    
}

void execute_remove_large_stripes(float *tomo, float *sorted_data, int *index_sorting, float *list_data, float *sorted_list, bool *list_mask, TomoAttributes attribute, int n_slices) {

    batch_mergesort_over_angles(tomo, index_sorting, n_slices, attribute.n_angle, attribute.n_length);

    median_filter_over_length(sorted_data, tomo, attribute.large_stripes_size, n_slices, attribute.n_angle, attribute.n_length);

    large_list(list_data, tomo, sorted_data, attribute.drop_ratio, n_slices, attribute.n_angle, attribute.n_length);

    detect_stripe_corrected(list_mask, list_data, sorted_list, n_slices, attribute.n_length, attribute.background_ratio);

    large_normalization_and_mask(sorted_data, tomo, list_data, list_mask, n_slices, attribute.n_angle, attribute.n_length);

    sino_reorder_from_sort(tomo, sorted_data, index_sorting, n_slices, attribute.n_angle, attribute.n_length);
}

void execute_remove_unresponsive_and_fluctuating_stripes(float *tomo, float *sorted_data, int *index_sorting, float *list_data, float *sorted_list, bool *list_mask, TomoAttributes attribute, int n_slices) {


    uniform_filter1d_over_angles(sorted_data, tomo, 10, n_slices, attribute.n_angle, attribute.n_length);

    list_unresponsive_sum(list_data, tomo, sorted_data, n_slices, attribute.n_angle, attribute.n_length);

    median_filter_over_length(sorted_list, list_data, attribute.large_stripes_size, n_slices, 1, attribute.n_length);

    list_unresponsive_divide(sorted_list, list_data, attribute.large_stripes_size, n_slices, attribute.n_length);

    detect_stripe_corrected(list_mask, sorted_list, list_data, n_slices, attribute.n_length, attribute.background_ratio);

    mask_unresponsive(list_mask, n_slices, attribute.n_length);

    swap_and_copy(sorted_data, tomo, n_slices, attribute.n_angle, attribute.n_length);

    interpolate_unresponsive(tomo, sorted_data, list_mask, n_slices, attribute.n_angle, attribute.n_length);
}


