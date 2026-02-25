#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
// #include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
// #include <thrust/device_ptr.h>
// #include <thrust/device_vector.h>
// #include <thrust/sequence.h>
#include "processing/rings_all_stripes.cuh"

__device__ inline int minimum(int x1, int x2) {
    return x1 < x2 ? x1 : x2;
}

__device__ inline void swap(float* x1, float* x2) {
    float temp = *x1;
    *x1 = *x2;
    *x2 = temp;
}

__device__ inline void swap(int* x1, int* x2) {
    int temp = *x1;
    *x1 = *x2;
    *x2 = temp;
}

__device__ inline int heap_left(int id) {
    return 2 * id + 1;
}

__device__ inline int heap_right(int id) {
    return 2 * id + 2;
}

__device__ void strided_max_heapify(float* d_data, int* d_indices,
        size_t stride, int y, int yend) {

    while (true) {
        int top = y;
        const int left  = heap_left(y);
        const int right = heap_right(y);

        if (left < yend && d_data[left * stride] > d_data[top * stride]) {
            top = left;
        }
        if (right < yend && d_data[right * stride] > d_data[top * stride]) {
            top = right;
        }

        if (top == y) break;

        swap(&d_data[top * stride], &d_data[y * stride]);
        if (d_indices != nullptr)
            swap(&d_indices[top * stride], &d_indices[y * stride]);
        y = top;
    }
}

__device__ size_t strided_partition(float *d_data, int* d_indices,
        size_t stride, int l, int h) {
    const size_t offset_p = h * stride;
    const float x = d_data[offset_p];
    int i = l;

    for (int j = l; j < h; j++) {
        if (d_data[j * stride] <= x) {
            swap(&d_data[j * stride], &d_data[i * stride]);
            if (d_indices != nullptr)
                swap(&d_indices[j * stride], &d_indices[i * stride]);
            i++;
        }
    }

    swap(&d_data[i * stride], &d_data[h * stride]);
    if (d_indices != nullptr)
        swap(&d_indices[i * stride], &d_indices[h * stride]);

    return i;
}

__device__ void quicksort_over_angles(float* d_data, int* d_indices,
        int n_slices, int n_angles, int n_length,
        size_t idx, size_t idz) {

    int ylow = 0, yhigh = n_angles - 1;

    const size_t offset = idz * n_length * n_angles + idx;

    const int rec_stack_size = 32;
    int rec_stack[rec_stack_size];
    int top = -1;

    rec_stack[++top] = ylow;
    rec_stack[++top] = yhigh;

    float* offset_data = d_data + offset;
    int* offset_idx = d_indices ? d_indices + offset : nullptr;

    while (top >= 0) {

        yhigh = rec_stack[top--];
        ylow = rec_stack[top--];

        const int p = strided_partition(offset_data, offset_idx, n_length, ylow, yhigh);

        if ((p - 1) > ylow) {
            //assert(top < (rec_stack_size - 2));
            rec_stack[++top] = ylow;
            rec_stack[++top] = p - 1;
        }

        if ((p + 1) < yhigh) {
            //assert(top < (rec_stack_size - 2));
            rec_stack[++top] = p + 1;
            rec_stack[++top] = yhigh;
        }
    }
}

__device__ void heapsort_over_angles(float* d_data, int* d_indices,
        int n_slices, int n_angles, int n_length,
        size_t idx, size_t idz) {
    const size_t offset = idz * n_angles * n_length + idx;
    float* offset_data = d_data + offset;
    int* offset_indices = d_indices ? d_indices + offset : nullptr;
    for (int y = n_angles / 2 - 1; y >= 0; --y) { // build heap
        strided_max_heapify(offset_data, offset_indices,
                n_length, y, n_angles);
    }

    for (int yend = n_angles; yend > 0; --yend) { // heap sort
        strided_max_heapify(offset_data, offset_indices,
                n_length, 0, yend);
        const size_t offset1 = idz * n_length * n_angles + 0 * n_length + idx;
        const size_t offset2 = idz * n_length * n_angles + (yend - 1) * n_length + idx;
        swap(&d_data[offset1], &d_data[offset2]);
        if (d_indices != nullptr)
            swap(&d_indices[offset1], &d_indices[offset2]);
    }
}

__device__ void selectionsort_over_angles(float* d_data, int* d_indices,
        int n_slices, int n_angles, int n_length,
        size_t idx, size_t idz) {

    for (size_t y1 = 0; y1 < n_angles; ++y1) {
        const size_t offset1 = idz * n_length * n_angles + y1 * n_length + idx;
        float min = d_data[offset1];
        size_t min_offset = offset1;
        for (size_t y2 = y1 + 1; y2 < n_angles; ++y2) {
            const size_t offset2 = idz * n_length * n_angles + y2 * n_length + idx;
            if (d_data[offset2] < min) {
                min = d_data[offset2];
                min_offset = offset2;
            }
        }
        d_data[min_offset] = d_data[offset1];
        d_data[offset1] = min;

        if (d_indices != nullptr) {
            int i_min = d_indices[min_offset];
            d_indices[min_offset] = d_indices[offset1];
            d_indices[offset1] = i_min;
        }
    }
}

__device__ void batches_sort_over_angles(float* d_data, int* d_indices,
        int n_slices, int n_angles, int n_length,
        size_t idx, size_t idz,
        float* batch_sort_shmem,
        int* batch_sort_shidx,
        int nthreads, int batch_size) {
    const int nbatches = n_angles / batch_size + (n_angles % batch_size > 0);

    const int tid = threadIdx.x;

    const size_t stride = n_length;
    float *offset_data = d_data + idz * n_length * n_angles + idx;
    int* offset_idx = d_indices ? d_indices + idz * n_length * n_angles + idx : nullptr;
    for (int b = 0; b < nbatches; ++b) { // copy from batch to shmem
        const int cur_batch_size = minimum(batch_size, int(n_angles - b * batch_size));
        for (int y = 0; y < cur_batch_size; ++y) {
            batch_sort_shmem[y * nthreads + tid] = offset_data[(b * batch_size + y) * stride];
            if (offset_idx)
                batch_sort_shidx[y * nthreads + tid] = offset_idx[(b * batch_size + y) * stride];
        }

        //heapsort_over_y(batch_sort_shmem, batch_sort_shidx, nthreads, cur_batch_size, 1, tid, 0);
        heapsort_over_angles(batch_sort_shmem, batch_sort_shidx, 1, cur_batch_size, nthreads, tid, 0);

        for (int y = 0; y < cur_batch_size; ++y) { // copy from shmem back to batch
            offset_data[(b * batch_size + y) * stride] = batch_sort_shmem[y * nthreads + tid];
            if (offset_idx != nullptr)
                offset_idx[(b * batch_size + y) * stride] = batch_sort_shidx[y * nthreads + tid];
        }
    }
}

__global__ void  _sort_over_angles(float* d_data, int* d_indices,
        int n_slices, int n_angles, int n_length) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idz >= n_slices || idx >= n_length) return;

    if (d_indices != nullptr) {
        for (int y = 0; y < n_angles; ++y) {
            const size_t offset = idz * n_length * n_angles + y * n_length + idx;
            d_indices[offset] = y;
        }
        __syncthreads();
    }

    //selectionsort_over_y(d_data, d_indices, xsize, ysize, zsize, idx, idz);
    heapsort_over_angles(d_data, d_indices, n_slices, n_angles, n_length, idx, idz);
    //quicksort_over_y(d_data, d_indices, xsize, ysize, zsize, idx, idz);
}


__device__ void batches_merge_over_angles(float* d_data, int* d_indices,
        float* d_data_out, int* d_indices_out,
        int n_slices, int n_angles, int n_length,
        size_t idx, size_t idz,
        float* batch_sort_shmem, int* batch_sort_shidx,
        int nthreads, int batch_size) {

    const int tid = threadIdx.x;
    const int max_batches = 128;
    const int nbatches = n_angles / batch_size + (n_angles % batch_size > 0);

    assert(n_angles <= (max_batches * batch_size));
    assert(nbatches <= batch_size); // needs to fit in the shared memory for merge
    assert(nbatches <= max_batches);

    const size_t offset = idz * n_angles * n_length + idx;
    const size_t stride = n_length;

    int8_t batches[max_batches];
    for (int b = 0; b < nbatches; ++b) {
        batches[b] = minimum(batch_size, int(n_angles - b * batch_size)) - 1;
        batch_sort_shmem[b * nthreads + tid] = d_data[offset + (b * batch_size + batches[b]) * stride];
        batch_sort_shidx[b * nthreads + tid] = b;
    }

    for (int b = nbatches / 2 - 1; b >= 0; --b) { // build heap
        strided_max_heapify(batch_sort_shmem + tid, batch_sort_shidx + tid, nthreads, b, nbatches);
    }

    for (int y = n_angles - 1; y >= 0; --y) {
        const int batch_max = batch_sort_shidx[tid];
        const float max = batch_sort_shmem[tid];

        const size_t out_idx = offset + y * stride;
        const size_t in_idx = offset + (batch_max * batch_size + batches[batch_max]) * stride;
        d_data_out[out_idx] = max;
        if (d_indices != nullptr) {
            d_indices_out[out_idx] = d_indices[in_idx];
        }

        // get next biggest element on selected batch
        // or fill with -infinity if the batch is empty
        batch_sort_shmem[tid] = (batches[batch_max] > 0) ? d_data[in_idx - stride] : -FLT_MAX;
        strided_max_heapify(batch_sort_shmem + tid, batch_sort_shidx + tid, nthreads, 0, nbatches);
        batches[batch_max]--;
    }
}

/**
 * This function sorts all the columns (y axis) in a volume.
 * The sorting strategy divides the volume in many batches (small enough to fit in shared memory),
 * where each batch will be sorted.
 * All the sorted batches are then merged in order to the output
 * volume (the merge strategy uses a heap stored in shared memory containing
 * the biggest element in each batch).
 **/
__global__ void  _batch_mergesort_over_angles(float* data, int* indices,
        float* d_data_out, int* d_indices_out,
        int n_slices, int n_angles, int n_length) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idz >= n_slices || idx >= n_length) return;

    const int nthreads = 32;
    const int batch_size = 128;

    assert(nthreads == blockDim.x);

    __shared__ float batch_sort_shmem[nthreads * batch_size];
    __shared__ int batch_sort_shidx[nthreads * batch_size];

    if (indices != nullptr) {
        for (size_t y = 0; y < n_angles; ++y) {
            const size_t offset = idz * n_length * n_angles + y * n_length + idx;
            indices[offset] = int(y);
        }
        __syncthreads();
    }

    batches_sort_over_angles(data, indices,
            n_slices, n_angles, n_length, idx, idz,
            batch_sort_shmem, batch_sort_shidx,
            nthreads, batch_size);

    __syncthreads();

    batches_merge_over_angles(data, indices,
            d_data_out, d_indices_out,
            n_slices, n_angles, n_length, idx, idz,
            batch_sort_shmem, batch_sort_shidx,
            nthreads, batch_size);
}

void sort_over_angles(float* d_data, int* d_indices,
        int n_slices, int n_angles, int n_length) {

    const dim3 block_size {1024, 1, 1};
    const size_t nblocksx = n_length / block_size.x + (n_length % block_size.x > 0);
    const dim3 nblocks { (unsigned int)(nblocksx), 1, (unsigned int)(n_slices) };

    _sort_over_angles<<<nblocks, block_size>>>(d_data, d_indices, n_slices, n_angles, n_length);

    // cudaDeviceSynchronize();
}



void batch_mergesort_over_angles(float* d_data, int* d_indices,
    int n_slices, int n_angles, int n_length) {

    const dim3 block_size {32, 1, 1};
    const size_t nblocksx = n_length / block_size.x + (n_length % block_size.x > 0);
    const dim3 nblocks { (unsigned int)(nblocksx), 1, (unsigned int)(n_slices) };


    float* d_data_tmp = nullptr;
    int* d_indices_tmp = nullptr;

    const size_t vol_size = size_t(n_length) * size_t(n_angles) * size_t(n_slices);

    cudaMalloc(&d_data_tmp, vol_size * sizeof(float));
    if (d_indices != nullptr)
        cudaMalloc(&d_indices_tmp, vol_size * sizeof(int));

    _batch_mergesort_over_angles<<<nblocks, block_size>>>(d_data, d_indices,
            d_data_tmp, d_indices_tmp,
            n_slices, n_angles, n_length);

    cudaMemcpy(d_data, d_data_tmp,
            vol_size * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_data_tmp);
    if (d_indices != nullptr) {
        cudaMemcpy(d_indices, d_indices_tmp,
                vol_size * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(d_indices_tmp);
    }

    cudaDeviceSynchronize();
}


// void cub_sort_over_x(float* d_data, int* d_indices,
//         size_t xsize, size_t ysize, size_t zsize) {

//     void* d_tmp_buffer = nullptr;
//     size_t tmp_bytes = 0;

//     thrust::device_vector<size_t> d_offsets(ysize * zsize + 1);
//     thrust::sequence(d_offsets.begin(), d_offsets.end(), 0ul, xsize);
//     if (d_indices != nullptr)
//         thrust::sequence(thrust::device_ptr<int>(d_indices),
//                 thrust::device_ptr<int>(d_indices + xsize * ysize * zsize), 0);

//     float* d_data_out = nullptr;
//     int* d_indices_out = nullptr;

//     cudaMalloc(&d_data_out, xsize * ysize * zsize * sizeof(float));
//     if (d_indices != nullptr)
//         cudaMalloc(&d_indices_out, xsize * ysize * zsize * sizeof(int));

//     if (d_indices != nullptr)
//         cub::DeviceSegmentedSort::SortPairs(
//             d_tmp_buffer, tmp_bytes,
//             d_indices, d_indices_out, d_data, d_data_out,
//             xsize * ysize * zsize, ysize * zsize, d_offsets.begin(), d_offsets.begin() + 1);
//     else
//         cub::DeviceSegmentedSort::SortKeys(d_tmp_buffer, tmp_bytes,
//             d_data, d_data_out, xsize * ysize * zsize, ysize * zsize,
//             d_offsets.begin(), d_offsets.begin() + 1);

//     cudaMalloc(&d_tmp_buffer, tmp_bytes);

//     if (d_indices != nullptr)
//         cub::DeviceSegmentedSort::SortPairs(
//             d_tmp_buffer, tmp_bytes,
//             d_data, d_data_out,
//             d_indices, d_indices_out,
//             xsize * ysize * zsize, ysize * zsize, d_offsets.begin(), d_offsets.begin() + 1);
//     else
//         cub::DeviceSegmentedSort::SortKeys(d_tmp_buffer, tmp_bytes,
//             d_data, d_data_out, xsize * ysize * zsize, ysize * zsize,
//             d_offsets.begin(), d_offsets.begin() + 1);


//     cudaMemcpy(d_data, d_data_out,
//             xsize * ysize * zsize * sizeof(float), cudaMemcpyDeviceToDevice);
//     if (d_indices != nullptr)
//         cudaMemcpy(d_indices, d_indices_out,
//                 xsize * ysize * zsize * sizeof(int), cudaMemcpyDeviceToDevice);

//     cudaFree(d_tmp_buffer);
//     cudaFree(d_data_out);
//     if (d_indices != nullptr)
//         cudaFree(d_indices_out);

//     cudaDeviceSynchronize();
// }


__device__ long kernel_pad(long i, long size) {
    if (i < 0)
        return 0;
    if (i >= size)
        return size - 1;
    return i;
}

__device__ long mirror_kernel_pad(long i, long size) {
    if (i < 0)
        return - ((i + 1) % size);
    else if (i >= size)
        return (size - 1) - (i % size);
    return i;
}

__global__ void _median_filter_over_length(float *output, float *input,
        size_t ksize, size_t zsize, size_t ysize, size_t xsize) {

    extern __shared__ float mf_shmem[];

    const size_t nthreads = blockDim.x;
    const size_t tid = threadIdx.x;

    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t idz = blockIdx.z * blockDim.z + threadIdx.z;

    if (idx >= xsize || idy >= ysize || idz >= zsize)
        return;

    const size_t heap_size = ksize / 2 + 1;
    for (long k = 0; k < heap_size; ++k) {
        const long aux_idx = nthreads * k + tid;
        mf_shmem[aux_idx] = FLT_MAX;
    }

    for (long k = 0; k < ksize; ++k) {
        const long in_idx = idz * xsize * ysize + idy * xsize +
            mirror_kernel_pad(idx + k - ksize/2, xsize);
        const float input_val = input[in_idx];
        // less than max value on heap
        if (input_val < mf_shmem[tid]) {
            mf_shmem[tid] = input_val;
            strided_max_heapify(mf_shmem + tid, nullptr, nthreads, 0, heap_size);
        }
    }

    const size_t output_idx = idz * xsize * ysize + idy * xsize + idx;
    output[output_idx] = mf_shmem[tid];
}

void median_filter_over_length(float* d_output, float* d_input,
        int ksize, int n_slices, int n_angles, int n_length) {

    const dim3 block_size {64, 1, 1};
    const size_t nblocksx = n_length / block_size.x + (n_length % block_size.x > 0);
    const dim3 nblocks { (unsigned int)(nblocksx), (unsigned int)(n_angles), (unsigned int)(n_slices) };
    const size_t shmem_size = block_size.x * (ksize / 2 + 1) * sizeof(float);

    _median_filter_over_length<<<nblocks, block_size, shmem_size>>> (d_output, d_input,
            ksize, n_slices, n_angles, n_length);

    cudaDeviceSynchronize();
}
