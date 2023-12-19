#ifndef RAFT_CONE_FDK_H
#define RAFT_CONE_FDK_H

#define NUM_THREADS 128
#define EPSILON_GX 1e-8

#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include <fstream>
#include <future>
#include <thread>
#include <time.h>
#include <vector>
#include <iostream>

#include "../../processing.h"

typedef struct {  
    float x,y,z;
    float dx, dy, dz;
    int nx, ny, nz;
    float h,v;
    float dh, dv;
    int nh, nv;
    float D, Dsd;
    float beta_max;
    float dbeta;
    int nbeta;
    int fourier;
    int filter_type; // Filter Types
    float reg; // Phase Filter (Paganin) regularization
    int is_slice; // (bool) Reconstruct a block of slices or not
    int slice_recon_start, slice_recon_end; // Slices: start slice = slice_recon_start, end slice = slice_recon_end
    int slice_tomo_start, slice_tomo_end; // Slices: start slice = slice_tomo_start, end slice = slice_tomo_end
    int nph, padh;
    float energy;

    /* Filter Types definitions
    enum EType
	{
        none      = 0,
        gaussian  = 1,
        lorentz   = 2,
        cosine    = 3,
        rectangle = 4,
        hann      = 5,
        hamming   = 6,
        ramp      = 7
	};
    */

} Lab;


// typedef struct {  
//     int i, i_gpu, zi, z_filter, z_filter_pad;
//     long long int n_proj, n_recon, n_filter, n_filter_pad;
//     long long int idx_proj, idx_proj_max, idx_recon, idx_filter, idx_filter_pad;
//     float z_ph, z_det;
// } Process;


//FDK Functions

extern "C"{
    void gpu_fdk(Lab lab,  float *recon, float *proj, float *angles, int* gpus, int ndev, double *time);
    void set_process(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs);
    void set_process_slices(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs);
    void set_process_slices_2(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs);
    int memory(Lab lab, int ndev);


    void set_filtering_fft(Lab lab, float* proj, int n_process,  int ndevs, Process* process);
    void set_filtering_conv(Lab lab, float* proj, int n_process,  int ndevs, Process* process);
    void set_backprojection(Lab lab, float* recon, float* proj, float *angles, int n_process,  int ndevs, Process* process);

    
    void copy_to_gpu_back(Lab lab, float* proj, float* recon, float *angles, float** c_proj, float** c_recon, float** c_beta, Process process);
    void copy_to_cpu_back(float* recon, float* c_proj, float* c_recon, float* c_beta, Process process);
    void copy_gpu_filter_fft(Lab lab, float* proj, float** c_proj, cufftComplex** c_signal, float** W, Process process);
    void copy_cpu_filter_fft(float* proj, float* c_proj, cufftComplex* c_signal, float* c_W,  Process process);
    void copy_gpu_filter_conv(Lab lab, float* proj, float** c_proj, float** c_Q, Process process) ;
    void copy_cpu_filter_conv(float* proj, float* c_proj, float* c_Q, Process process);
    
    void backprojection(Lab lab, float* recon, float* proj, float* beta,  Process process);
    __global__ void backproj(float* recon, float* proj, float* beta, Lab lab, Process process);
    __global__ void set_beta(Lab lab, float *dangles, float* beta);
    __device__ void set_recon_idxs(long long int n, int* i, int*j, int* k, Lab lab);
    __device__ void set_filter_idxs_pad(long long int n, int* i, int*j, int* k, Lab lab, Process process);


    __host__ void fft(Lab lab, float* proj, cufftComplex* signal, float* W, Process process);
    __host__ void fft_nopad(Lab lab, float* proj, cufftComplex* signal, float* W, Process process);

    
    __global__ void signal_save(Lab lab, float* proj, cufftComplex* signal, Process process);
    __global__ void signal_save_pad(Lab lab, float* proj, cufftComplex* signal, Process process);
    __global__ void signal_filter(Lab lab, float* W, cufftComplex* signal, Process process);
    __global__ void signal_filter_pad(Lab lab, float* W, cufftComplex* signal, Process process);
    __global__ void filt_W(Lab lab, float* W);
    __global__ void filt_Ramp(Lab lab, float* W);
    __global__ void filt_Gaussian(Lab lab, float* W);
    __global__ void filt_Lorentz(Lab lab, float* W);
    __global__ void filt_Cosine(Lab lab, float* W);
    __global__ void filt_Rectangle(Lab lab, float* W);
    __global__ void filt_Hann(Lab lab, float* W);
    __global__ void filt_Hamming(Lab lab, float* W);
    __global__ void signal_inv(Lab lab, float* Q, cufftComplex* signal, Process process);
    __global__ void signal_inv_pad(Lab lab, float* Q, cufftComplex* signal, Process process);

    __device__ void set_filter_idxs(long long int n, int* i, int*j, int* k, Lab lab, Process process);
    __device__ void set_filter_idxs_pad(long long int n, int* i, int*j, int* k, Lab lab, Process process);

    void filtering_conv(Lab lab, float* proj, float* Q, Process process);
    __global__ void calc_Q(float* Q, float* proj, Lab lab, Process process);
    __device__ float calc_convolution(float* projection, Lab lab, float X, float Z, float w_max);
    __device__ float integral_of_gx(float a, float b, float w_max);
    __forceinline__ __device__ float gx(float x, float w_max);
    __device__ void set_filter_idxs_conv(long long int n, int* i, int*j, int* k, Lab lab, Process process);
}


#endif