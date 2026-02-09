#ifndef RINGS_ALL_STRIPES_H
#define RINGS_ALL_STRIPES_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cfloat>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "common/configs.hpp"

// Remove Stripes Algorithms
extern "C"{

  void remove_all_stripes(float *tomo, TomoAttributes attribute, int ndevs, int batch);
  void _remove_all_stripes(float *tomo, TomoAttributes attribute, int n_slices, int device);
  void get_remove_all_stripes(float *tomo, TomoAttributes attribute, int nslices, int nangles, int nrays);

  void remove_all_stripes_transposed(float *tomo, TomoAttributes attribute, int ndevs, int batch);
  void _remove_all_stripes_transposed(float *tomo_, TomoAttributes attribute, int start_x, int n_length, int device);

}

// void remove_unresponsive_and_fluctuating_stripes(float *tomo, TomoAttributes attribute);
// void remove_large_stripe(float *tomo, TomoAttributes attribute);
// void remove_stripe_based_sorting(float *tomo, TomoAttributes attribute);

// Execute Remove All Stripes

void execute_remove_all_stripes(float *tomo, float *sorted_data, int *index_sorting, float *list_data, float *sorted_list, bool *list_mask, TomoAttributes attribute, int n_slices);

void execute_remove_based_sorting(float *tomo, float *sorted_data, int *index_sorting, float *list_data, float *sorted_list, bool *list_mask, TomoAttributes attribute, int n_slices);

void execute_remove_large_stripes(float *tomo, float *sorted_data, int *index_sorting, float *list_data, float *sorted_list, bool *list_mask, TomoAttributes attribute, int n_slices);

void execute_remove_unresponsive_and_fluctuating_stripes(float *tomo, float *sorted_data, int *index_sorting, float *list_data, float *sorted_list, bool *list_mask, TomoAttributes attribute, int n_slices);


// UTILS
void checkCuda(cudaError_t result);
void _zyx_to_xyz(float* out, const float* in, size_t sizez, size_t sizey, size_t sizex);
__global__ void k_zyx_to_xyz(float* out, const float* in, size_t sizez, size_t sizey, size_t sizex); 

// Kernels

__global__ void _sino_reorder_from_sort(float* __restrict__ tomo, const float* __restrict__ sorted_tomo, const int* __restrict__ indexes, int n_slices, int n_angles, int n_length);
void sino_reorder_from_sort(float *tomo, float *sorted_tomo, int *indexes, int n_slices, int n_angles, int n_length);

__global__ void _swap_and_copy(float* __restrict__ output, const float* __restrict__ input,int n_slices, int n_angles, int n_length);
void swap_and_copy(float *output, float *input, int n_slices, int n_angles, int n_length);

__global__ void _large_list( float* __restrict__ list, const float* __restrict__ sorted_data,const float* __restrict__ smooth_data, float drop_ratio, int n_slices, int n_angles, int n_length);
void large_list(float *list, float *sorted_data, float *smooth_data, float drop_ratio, int n_slices, int n_angles, int n_length);

__global__ void _detect_stripe_corrected(  bool* __restrict__ list_mask, const float* __restrict__ list_data, float* __restrict__ sorted_list, int n_slices, int n_length, float background_ratio);
void detect_stripe_corrected(bool *list_mask, float *list_data, float *sorted_list, int n_slices, int n_length, float background_ratio);

__global__ void _large_normalization_and_mask(float* __restrict__ out, float* __restrict__ tomo,const float* __restrict__ list,const bool*  __restrict__ mask, int n_slices, int n_angles, int n_lengths);
void large_normalization_and_mask(float *output, float *tomo, float *list, bool *mask, int n_slices, int n_angles, int n_length);

__global__ void _uniform_filter1d_over_angles(float* __restrict__ output,const float* __restrict__ input,int size, int n_slices, int n_angles, int n_length);
void uniform_filter1d_over_angles(float*  output, float*  input, int size, int n_slices, int n_angles, int n_length);

__global__ void _list_unresponsive_sum(float* __restrict__ list_data, const float* __restrict__ tomo, const float* __restrict__ tomo_smooth, int n_slices, int n_angles, int n_length);
void list_unresponsive_sum(float *list_data, float *tomo, float *tomo_smooth, int n_slices, int n_angles, int n_length);

__global__ void _list_unresponsive_divide(float* __restrict__ smooth_list, const float* __restrict__ list, int size_median, int n_slices, int n_length);
void list_unresponsive_divide(float *smooth_list, float *list, int size_median, int n_slices,int n_length);

__global__ void _mask_unresponsive(bool* __restrict__ list_mask, int n_slices, int n_length);
void mask_unresponsive(bool *list_mask, int n_slices, int n_length);

__global__ void _interpolate_with_neighbors(float* __restrict__ out, const float* __restrict__ tomo, const bool*  __restrict__ mask,const int* __restrict__ left_idx, const int* __restrict__ right_idx, int n_slices, int n_angles, int n_length);
__global__ void _build_neighbors_interpolation(int* __restrict__ left_idx, int* __restrict__ right_idx, const bool* __restrict__ mask,int n_slices, int n_length);
void interpolate_unresponsive(float *tomo_interp, float *tomo, bool *list_mask, int n_slices, int n_angles, int n_length);


// SORTING
// --------

// Device and kernel function declarations from sort.cu

__device__ void quicksort_over_angles(float* d_data, int* d_indices, int xsize, int ysize, int zsize, size_t idx, size_t idz);
__device__ void heapsort_over_angles(float* d_data, int* d_indices, int xsize, int ysize, int zsize, size_t idx, size_t idz);
__device__ void selectionsort_over_angles(float* d_data, int* d_indices, int xsize, int ysize, int zsize, size_t idx, size_t idz);


void sort_over_angles(float* d_data, int* d_indices, int xsize, int ysize, int zsize);
void batch_mergesort_over_angles(float* d_data, int* d_indices, int xsize, int ysize, int zsize);
// void cub_sort_over_x(float* d_data, int* d_indices, size_t xsize, size_t ysize, size_t zsize);
void median_filter_over_length(float* d_output, float* d_input, int ksize, int xsize, int ysize, int zsize);


#endif
