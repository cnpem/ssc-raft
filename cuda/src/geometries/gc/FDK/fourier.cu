#include "../../../../inc/gc/fdk.h"
#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>

extern "C"{
__host__ void fft(Lab lab, float* proj, cufftComplex* signal, float* W, Process process){
    int n = lab.nh;
    long long int batch = process.z_filter*lab.nbeta;
    long long int N = process.n_filter;

    int n_threads = NUM_THREADS;
    long long int n_blocks  = N/n_threads + (N % n_threads == 0 ? 0:1);

    cudaSetDevice(process.i_gpu);

    int v[] = {n};
    cufftHandle plan;
    cufftPlanMany(&plan, 1, v, v, 1, n, v, 1, n, CUFFT_C2C, batch);

    //Calculate Signal
    signal_save<<<n_blocks, n_threads>>>(lab, proj, signal, process);

    //Forward FFT
    printf("Forward FFT...\n");
    cufftExecC2C(plan,(cufftComplex*) signal,(cufftComplex*) signal, CUFFT_FORWARD);

    signal_filter<<<n_blocks, n_threads>>>(lab, W, signal, process);

    printf("Inverse FFT...\n");
    cufftExecC2C(plan,(cufftComplex*) signal,(cufftComplex*) signal, CUFFT_INVERSE);

    signal_inv<<<n_blocks, n_threads>>>(lab, proj, signal, process);

    cufftDestroy(plan); 
}}


extern "C"{
__global__ void signal_save(Lab lab, float* proj, cufftComplex* signal, Process process){
    long long int n = blockDim.x * blockIdx.x + threadIdx.x ;
    int i, j, k; 
    long long int idx;
    float X, Z, aux;

    set_filter_idxs(n, &k, &j, &i, lab, process);
    idx = (long long int) k + j*lab.nh + i*lab.nh*lab.nbeta;

    X = k*lab.dh - lab.h;
    Z = (i + process.i*process.z_filter)*lab.dv - lab.v;
    aux = lab.D/sqrtf(lab.Dsd*lab.Dsd + Z*Z + X*X);

    signal[idx].x = proj[idx]*aux;
    signal[idx].y = 0.0;
}}

extern "C"{
__global__ void signal_filter(Lab lab, float* W, cufftComplex* signal, Process process){
    long long int n = blockDim.x * blockIdx.x + threadIdx.x ;
    int i,j,k;
    long long int idx;

    set_filter_idxs(n, &k, &j, &i, lab, process);
    idx = (long long int) k + j*lab.nh + i*lab.nh*lab.nbeta;

    signal[idx].x = signal[idx].x*W[k];
    signal[idx].y = signal[idx].y*W[k];      
}}

extern "C"{
__global__ void filt_W(Lab lab, float* W){
    int i;
    float wmax = 1.0/(2.0*lab.dh);
    for (i = 0; i < lab.nh/2; i++) W[i] = (wmax)/(lab.nh-1) + (2*i*wmax)/(lab.nh-1);     
    for (i = 0; i < lab.nh/2; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh-1);
}}

extern "C"{
__global__ void signal_inv(Lab lab, float* Q, cufftComplex* signal, Process process){
    long long int n = blockDim.x * blockIdx.x + threadIdx.x ;
    int i,j,k;
    long long int idx;

    set_filter_idxs(n, &k, &j, &i, lab, process);
    idx = (long long int) k + j*lab.nh + i*lab.nh*lab.nbeta;

    Q[idx] = signal[idx].x/lab.nh;
}}

extern "C"{
__device__ void set_filter_idxs(long long int n, int* i, int*j, int* k, Lab lab, Process process) {
    long int nij, rem_ij;
    nij = lab.nh*lab.nbeta;
    *k = (n) / nij;
    rem_ij = (n) % nij;
    *j = rem_ij / lab.nh;
    *i = rem_ij % lab.nh;
}}

extern "C"{
__device__ void set_projs_idxs(long long int n, int* i, int* k, int* m, Lab lab) {
    long int nik, rem_ik;
    nik = lab.nx*lab.nz;
    *m = (n) / nik;
    rem_ik = (n) % nik;
    *k = rem_ik / lab.nz;
    *i = rem_ik % lab.nz;
}}
