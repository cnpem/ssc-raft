#ifndef RAFT_RADON_PAR_H
#define RAFT_RADON_PAR_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufftXt.h>
#include <complex.h>

#define BYTES_TO_GB (1.0/(1024.0*1024.0*1024.0))
#define A100_MEM 39.5 // A100 40GB device RAM memory, in GB.

#define FT_PST_RANK_FORWARD 2 // Forward Fourier Transform rank for Fourier/projection slice theorem (F/PST).
#define FT_PST_RANK_INVERSE 1 // Inverse Fourier Transform rank for Fourier/projection slice theorem (F/PST).

#define NUM_THREADS 1024

typedef std::complex<float> data_type;
typedef float real_data_type;

/* Radon Ray Tracing */
extern "C"{

	void GRadon(int device, float* _frames, float* _image, int nrays, int nangles, int blocksize);

    __global__ void KRadon_RT(float* restrict frames, const float* image, int nrays, int nangles);


}

/* Radon Fourier Slice Theorem (FST) */
extern "C"{

	int fst_gpu(
    float *sino, float *recon, float *angles,
    int nrays, int nangles, int blocksize,
    int zpad, int interpolation, float dx,
    float *debug);

    int real_fst_gpu(
    float *sino, float *recon, float *angles, 
    int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx,
    float *debug);

    int counts_real_fst_gpu(
    float *sino, float *recon, float *angles, float *flat,
    int nrays, int nangles, int nslices, 
    int zpad, int interpolation, float dx,
    float *debug);

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

    void free_cuda_counts_real(
        float *angles_cu, float *flat_cu, float *recon_cu, float *sino_cu,
        cufftComplex *ft_recon_cu, cufftComplex *ft_sino_cu,
        cufftHandle plan_2D_forward, cufftHandle plan_1D_inverse,
        cudaStream_t stream, cudaStream_t stream_H2D, cudaStream_t stream_D2H);

    int calc_blocksize(float nslices, float nangles, float nrays, float zpad, bool complex_data);

    int el_wise_mult(
    float *arr1, float *arr2,
    int n1, int n2, int n3);

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
    __global__ void total_variation_2d(
        float *back,
        float *recon,
        float *backcounts,
        size_t size,
        int nx, int ny, int nz,
        float tv_param);
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

}


#endif