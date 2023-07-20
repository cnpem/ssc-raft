#include <cufft.h>
#include <stddef.h> // size_t. also compiles without this header (weird).

#ifndef KERFST_H
#define KERFST_H

#define NUM_THREADS 1024


__global__ void calc_counts(
	float *sino, float *flat,
    int nrays, int nangles, int nslices);
__global__ void set_value(
	float *arr,
	float val,
	size_t size);
__global__ void multiply_element_wise(
	float *arr1,
	float *arr2,
	size_t size);
__global__ void calc_reciprocal_element_wise(
	float *arr,
	size_t size);
__global__ void scale_data(
	float *data,
	size_t size,
	float scale);
__global__ void scale_data_complex(
	cufftComplex *data,
	size_t size,
	float scale);
__global__ void scale_data_real_only(
	cufftComplex *data,
	size_t size,
	float scale);
__global__ void scale_data_imag_only(
	cufftComplex *data,
	size_t size,
	float scale);

__global__ void copy_to_fft_workspace(
	cufftComplex *workspace, cufftComplex *src,
	int n1, int n2, int n1_src, int n2_src, int blocksize);

__global__ void copy_to_fft_workspace_R2C(
	cufftComplex *workspace, float *src,
	int n1, int n2, int n1_src, int n2_src, int blocksize);

__global__ void copy_from_fft_workspace(
	cufftComplex *workspace, cufftComplex *dst,
	int n1, int n2, int n1_dst, int n2_dst, int blocksize);

__global__ void copy_from_fft_workspace_C2R(
	cufftComplex *workspace, float *dst,
	int n1, int n2, int n1_dst, int n2_dst, int blocksize);

__global__ void shift_1d(
	cufftComplex *c,
	size_t sizex, size_t sizey, size_t sizez);

__global__ void shift_1d_real(
	float *data,
	size_t sizex, size_t sizey, size_t sizez);

// Payola's shift kernel:
__global__ void fftshift2d(cufftComplex *c, size_t sizex, size_t sizey, size_t sizez);

// Payola's 2 dimensional fft shift kernel for real/non-complex float data:
__global__ void fftshift2d_real(float *data, size_t sizex, size_t sizey, size_t sizez);

// Payola's -MODIFIED FOR BETTER PERFORMANCE- 2 dimensional fft shift kernel:
__global__ void fftshift2d_v2(cufftComplex *c, size_t sizex, size_t sizey, size_t sizez);

// Kernel for nearest neighbor interpolation from 2DFT of recon to 1DFT of projections at the sinogram. 
__global__ void cartesian2polar_nn(
	cufftComplex *ft_recon,
	cufftComplex *ft_sino,
	float *angles,
	size_t nrays, size_t nangles, size_t blocksize);

// Kernel for bilinear interpolation from 2DFT of recon to 1DFT of projections at the sinogram.
// DON'T USE: Bilinear interpolation fails when xd = xu (e.g., ang = pi/2) or when yd = yu (e.g., ang = 0).
// Use, instead, 'cartesian2polar_bi_v1' or 'cartesian2polar_bi_v2' bellow.
__global__ void cartesian2polar_bi_v0(
	cufftComplex *ft_recon,
	cufftComplex *ft_sino,
	float *angles,
	size_t nrays, size_t nangles, size_t blocksize);

// Kernel for bilinear interpolation from 2DFT of recon to 1DFT of projections at the sinogram.
// Same kernel as as above but with a correction for when xd = xu and yd = u.
__global__ void cartesian2polar_bi_v1(
	cufftComplex *ft_recon,
	cufftComplex *ft_sino,
	float *angles,
	size_t nrays, size_t nangles, size_t blocksize);

// Kernel for bilinear interpolation from 2DFT of recon to 1DFT of projections at the sinogram.
// Same as 'cartesian2polar_bi_v1' but using another aproach for the correction:
//   Here we don't use 'if's to avoid branch divergence. Instead, the ajustments are performed on the data.
__global__ void cartesian2polar_bi_v2(
	cufftComplex *ft_recon,
	cufftComplex *ft_sino,
	float *angles,
	size_t nrays, size_t nangles, size_t blocksize);

	#endif