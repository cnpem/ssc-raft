// in case it is compiled with host compiler instead of nvcc:
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <complex>

#ifndef RAFT_FST_H
#define RAFT_FST_H

#define DEBUG

#define BYTES_TO_GB (1.0/(1024.0*1024.0*1024.0))
#define A100_MEM 39.5 // A100 40GB device RAM memory, in GB.

#define FT_PST_RANK_FORWARD 2 // Forward Fourier Transform rank for Fourier/projection slice theorem (F/PST).
#define FT_PST_RANK_INVERSE 1 // Inverse Fourier Transform rank for Fourier/projection slice theorem (F/PST).

/* CPU Functions*/

typedef ::std::complex<float> data_type;
typedef float real_data_type;

extern "C" {
int fst_gpu(
	float *sino, float *recon, float *angles,
	int nrays, int nangles, int blocksize,
    int zpad, int interpolation, float dx,
    float *debug);
}

// Same as above but for real valued sinogram and recon.
extern "C" {
int real_fst_gpu(
	float *sino, float *recon, float *angles, 
	int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx,
    float *debug);
}

// Same as above but returning counts = flat*exp(-tomo) instead of tomo.
extern "C" {
int counts_real_fst_gpu(
	float *sino, float *recon, float *angles, float *flat,
	int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx,
    float *debug);
}

void pst(
    cufftComplex *ft_sino_cu, cufftComplex *ft_recon_cu, cufftComplex *sino_cu, cufftComplex *recon_cu, 
    float *angles_cu, cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    int nrays, int nangles, int nslices,
    int zpad, int interpolation, float scale,
    cudaStream_t stream);

// Same as pst but for real valued sample only:
void pst_real(
    cufftComplex *ft_sino_cu, cufftComplex *ft_recon_cu, float *sino_cu, float *recon_cu, 
    float *angles_cu, cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float scale,
    cudaStream_t stream);

// Same as pst but for counts = flat*exp(-tomo) instead of tomo.
void pst_counts_real(
    cufftComplex *ft_sino_cu, cufftComplex *ft_recon_cu, float *sino_cu, float *recon_cu, 
    float *angles_cu, float *flat_cu, cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float scale,
    cudaStream_t stream);

void free_cuda(
    float *angles_cu, cufftComplex *recon_cu, cufftComplex *sino_cu,
    cufftComplex *ft_recon_cu, cufftComplex *ft_sino_cu,
    cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    cudaStream_t stream, cudaStream_t stream_H2D, cudaStream_t stream_D2H);

// Same as above but for real valued sinogram and recon.
void free_cuda_real(
    float *angles_cu, float *recon_cu, float *sino_cu,
    cufftComplex *ft_recon_cu, cufftComplex *ft_sino_cu,
    cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
    cudaStream_t stream, cudaStream_t stream_H2D, cudaStream_t stream_D2H);

int calc_blocksize(float nslices, float nangles, float nrays, float zpad, bool complex_data);


extern "C" {
int el_wise_mult(
	float *arr1, float *arr2,
	int n1, int n2, int n3);
}


#endif