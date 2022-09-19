#ifndef RAFT_CONE_FDK_H
#define RAFT_CONE_FDK_H

#define NUM_THREADS 128
#define EPSILON_GX 1e-10

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
    bool fourier;
} Lab;

typedef struct {  
    int i, i_gpu, zi;
    long long int n_proj, n_recon, n_filter;
    long long int idx_proj, idx_recon, idx_filter;
    float z_ph, z_det, z_filter;
} Process;


//FDK Functions

extern "C"{
    void gpu_fdk(Lab lab, float *recon, float *proj, int* gpus, int ndev,  double *time);

    void filtering(Lab lab, float* proj, Process process); 

    void backprojection(Lab lab, float* recon, float* proj, Process process);

    void set_process(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs);

    int memory(Lab lab, int ndev);

    void copy_to_cpu_back(float* recon, float* c_proj, float* c_recon,  Process process);

    void copy_to_gpu_back(Lab lab, float* proj, float* recon, float** c_proj, float** c_sample, Process process); }


// Backprojection
extern "C"{
    void calc_backproj(float* recon, float* proj, Lab lab, Process process);

    __global__ void backproj(float* recon, float* proj, Lab lab, Process process);

    __global__ void backproj_interpol(float* recon, float* proj, Lab lab, Process process);}


// Filter Functions: Fourier Transform

extern "C"{
    __host__ void calc_fft(Lab lab, float* proj, Process process);

    __host__ void signal_fft(Lab lab, float* proj, cufftComplex* signal, int op, Process process);

    __global__ void filt_W(Lab lab, float* W);

    __global__ void signal_filter(Lab lab, float* W, cufftComplex* signal, Process process);

    __global__ void signal_inv(Lab lab, float* Q, cufftComplex* signal, Process process);

    __device__ void set_filter_idxs(long long int n, int* i, int*j, int* k, Lab lab, Process process);

    __global__ void signal_save(Lab lab, float* proj, cufftComplex* signal, Process process);

    void copy_gpu_filter(Lab lab, float* proj, float** c_proj, cufftComplex** c_signal, float** W, Process process);

    __host__ void fft(Lab lab, float* proj, cufftComplex* signal, float* W, Process process);

    void copy_cpu_filter(float* proj, float* c_proj, cufftComplex* c_signal, float* c_W,  Process process);}


// Utilitary Functions
extern "C"{
    __device__ void set_recon_idxs(long long int n, int* i, int*j, int* k, Lab lab);

    __device__ void set_projs_idxs(long long int n, int* i, int* k, int* m, Lab lab);}


#endif