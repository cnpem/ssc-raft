#include "processing/rings_all_stripes.cuh"


void sino_reorder_from_sort(float *tomo, float *sorted_tomo, int *indexes, int n_slices, int n_angles, int n_length){
    
    dim3 block(128, 2, 1);  // 256 threads/block
    dim3 grid((n_length + block.x - 1) / block.x, (n_angles + block.y - 1) / block.y, (n_slices + block.z - 1) / block.z);

    _sino_reorder_from_sort<<<grid, block>>>(tomo, sorted_tomo, indexes, n_slices, n_angles, n_length);
    
    checkCuda(cudaDeviceSynchronize());

}

void swap_and_copy(float *output, float *input, int n_slices, int n_angles, int n_length) {

    dim3 block(128, 2, 1);  // 256 threads/block
    dim3 grid( (n_length + block.x - 1) / block.x, (n_angles + block.y - 1) / block.y, (n_slices + block.z - 1) / block.z );

    _swap_and_copy<<<grid, block>>>(output, input, n_slices, n_angles, n_length);
    
    checkCuda(cudaDeviceSynchronize());
}

void large_list(float *list, float *sorted_data, float *smooth_data, float drop_ratio, int n_slices, int n_angles, int n_length){

    int block_size = 256;
    int blocks_slice = ((n_slices * n_length) + block_size - 1) / block_size;

    _large_list<<<blocks_slice, block_size>>>(list, sorted_data, smooth_data, drop_ratio, n_slices, n_angles, n_length);

    checkCuda(cudaDeviceSynchronize());
}

void detect_stripe_corrected(bool *list_mask, float *list_data, float *sorted_list, int n_slices, int n_length, float background_ratio){

    int block_size = 256;
    int blocks_array = ((n_slices) + block_size - 1) / block_size;

    _detect_stripe_corrected<<<blocks_array, block_size>>>(list_mask, list_data, sorted_list, n_slices, n_length, background_ratio);

    checkCuda(cudaDeviceSynchronize());
}

void large_normalization_and_mask(float *output, float *tomo, float *list, bool *mask, int n_slices, int n_angles, int n_length){

    dim3 block(128, 2, 1);  // 256 threads/block
    dim3 grid((n_length + block.x - 1) / block.x, (n_angles + block.y - 1) / block.y, (n_slices + block.z - 1) / block.z );

    _large_normalization_and_mask<<<grid, block>>>(output, tomo, list, mask, n_slices, n_angles, n_length);

    checkCuda(cudaDeviceSynchronize());
}

void uniform_filter1d_over_angles(float*  output, float*  input,
    int size, int n_slices, int n_angles, int n_length){

    dim3 block(128, 2, 1);  // 256 threads/block
    dim3 grid((n_length + block.x - 1) / block.x, (n_angles + block.y - 1) / block.y, (n_slices + block.z - 1) / block.z );
    
    _uniform_filter1d_over_angles<<<grid, block>>>(output, input, size, n_slices, n_angles, n_length);
    
    checkCuda(cudaDeviceSynchronize());
} 

void list_unresponsive_sum(float *list_data, float *tomo, float *tomo_smooth, int n_slices, int n_angles, int n_length){

    int block_size = 256;
    int blocks_slice = ((n_slices * n_length) + block_size - 1) / block_size;
    
    _list_unresponsive_sum<<<blocks_slice, block_size>>>(list_data, tomo, tomo_smooth, n_slices, n_angles, n_length);

    checkCuda(cudaDeviceSynchronize());
}


void list_unresponsive_divide(float *smooth_list, float *list, int size_median, int n_slices,int n_length) {

    int block_size = 256;
    int blocks_slice = ((n_slices * n_length) + block_size - 1) / block_size;
    
    _list_unresponsive_divide<<<blocks_slice, block_size>>>(smooth_list, list, size_median, n_slices, n_length);

    checkCuda(cudaDeviceSynchronize());
}

void mask_unresponsive(bool *list_mask, int n_slices, int n_length){
    int block_size = 256;
    int blocks_array = ((n_slices) + block_size - 1) / block_size;
    
    _mask_unresponsive<<<blocks_array, block_size>>>(list_mask, n_slices, n_length);
    
    checkCuda(cudaDeviceSynchronize());
}

void interpolate_unresponsive(float *tomo_interp, float *tomo,
    bool *list_mask, int n_slices,
    int n_angles, int n_length){

    const size_t lr_elems = (size_t)n_slices * (size_t)n_length;
    const size_t lr_bytes = lr_elems * sizeof(int);

    // 1) Allocate device memory for left/right indices
    int* d_left_idx  = nullptr;
    int* d_right_idx = nullptr;
    cudaMalloc((void**)&d_left_idx,  lr_bytes);
    cudaMalloc((void**)&d_right_idx, lr_bytes);

    // 2) (Optional) initialize to -1: memset bytes to 0xFF => -1 for 2's complement ints
    cudaMemset(d_left_idx,  0xFF, lr_bytes);
    cudaMemset(d_right_idx, 0xFF, lr_bytes);


    // Many rows processed in parallel; x-dim unused here.
    dim3 block(1, 128, 1);
    dim3 grid(1, (n_slices + block.y - 1) / block.y, 1);

    _build_neighbors_interpolation<<<grid, block>>>(d_left_idx, d_right_idx, list_mask, n_slices, n_length);

    checkCuda(cudaDeviceSynchronize());

    // Keep warps contiguous on k; tile lightly on j; z for slices
    dim3 block2(128, 2, 1);  // 256 threads/block; tune 256x1x1 if Nj is tiny
    dim3 grid2( (n_length + block.x - 1) / block.x, (n_angles + block.y - 1) / block.y, (n_slices + block.z - 1) / block.z );

    _interpolate_with_neighbors<<<grid2, block2>>>(tomo_interp, tomo, list_mask, d_left_idx, d_right_idx, n_slices, n_angles, n_length);
    
    checkCuda(cudaDeviceSynchronize());

    cudaFree(d_left_idx);
    cudaFree(d_right_idx);
}

