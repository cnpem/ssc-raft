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
} Lab;


typedef struct {  
    int i, i_gpu, zi, z_filter;
    long long int n_proj, n_recon, n_filter;
    long long int idx_proj, idx_recon, idx_filter;
    float z_ph, z_det;
} Process;


//FDK Functions

extern "C"{
    void gpu_fdk(Lab lab,  float *recon, float *proj, int* gpus, int ndev,  double *time);
    void set_process(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs);
    int memory(Lab lab, int ndev);


    void set_filtering_fft(Lab lab, float* proj, int n_process,  int ndevs, Process* process);
    void set_filtering_conv(Lab lab, float* proj, int n_process,  int ndevs, Process* process);
    void set_backprojection(Lab lab, float* recon, float* proj, int n_process,  int ndevs, Process* process);

    
    
    void copy_to_gpu_back(Lab lab, float* proj, float* recon, float** c_proj, float** c_recon, float** c_beta, Process process);
    void copy_to_cpu_back(float* recon, float* c_proj, float* c_recon, float* c_beta, Process process);
    void copy_gpu_filter_fft(Lab lab, float* proj, float** c_proj, cufftComplex** c_signal, float** W, Process process);
    void copy_cpu_filter_fft(float* proj, float* c_proj, cufftComplex* c_signal, float* c_W,  Process process);
    void copy_gpu_filter_conv(Lab lab, float* proj, float** c_proj, float** c_Q, Process process) ;
    void copy_cpu_filter_conv(float* proj, float* c_proj, float* c_Q, Process process);
    


    void backprojection(Lab lab, float* recon, float* proj, float* beta,  Process process);
    __global__ void backproj(float* recon, float* proj, float* beta, Lab lab, Process process);
    __global__ void set_beta(Lab lab, float* beta);
    __device__ void set_recon_idxs(long long int n, int* i, int*j, int* k, Lab lab);



    __host__ void fft(Lab lab, float* proj, cufftComplex* signal, float* W, Process process);
    void ringsgpu_fdk( Lab lab, float* data, Process process);
    __global__ void signal_save(Lab lab, float* proj, cufftComplex* signal, Process process);
    __global__ void signal_filter(Lab lab, float* W, cufftComplex* signal, Process process);
    __global__ void filt_W(Lab lab, float* W);
    __global__ void signal_inv(Lab lab, float* Q, cufftComplex* signal, Process process);
    __device__ void set_filter_idxs(long long int n, int* i, int*j, int* k, Lab lab, Process process);



    void filtering_conv(Lab lab, float* proj, float* Q, Process process);
    __global__ void calc_Q(float* Q, float* proj, Lab lab, Process process);
    __device__ float calc_convolution(float* projection, Lab lab, float X, float Z, float w_max);
    __device__ float integral_of_gx(float a, float b, float w_max);
    __forceinline__ __device__ float gx(float x, float w_max);
    __device__ void set_filter_idxs_conv(long long int n, int* i, int*j, int* k, Lab lab, Process process);
}


#endif