#include "../../../../inc/include.h"
#include "../../../../inc/gp/rings.h"
#include "../../../../inc/gc/fdk.h"
#include "../../../../inc/common/types.hpp"



extern "C"{
void copy_gpu_filter_fft(Lab lab, float* proj, float** c_proj, cufftComplex** c_signal, float** W, Process process) {
    long long int N = process.n_filter;

    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    cudaMalloc(c_signal, sizeof(cufftComplex)*N);

    cudaMalloc(c_proj, process.n_filter * sizeof(float));    
    cudaMemcpy(*c_proj, &proj[process.idx_filter], process.n_filter * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(W, lab.nh * sizeof(float));
    filt_W<<< 1, 1>>>(lab, *W);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

 
    printf("GPU memory allocated...\n");

    clock_t end = clock();
    printf("Time copy_to_gpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_cpu_filter_fft(float* proj, float* c_proj, cufftComplex* c_signal, float* c_W,  Process process) {
    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    long long int N = process.n_filter;                                         //lab.nbeta * lab.nv * lab.nh;
    cudaMemcpy(&proj[process.idx_filter], c_proj, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_proj);
    cudaFree(c_signal);
    cudaFree(c_W);

    clock_t end = clock();
    printf("Time copy_to_cpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_gpu_filter_conv(Lab lab, float* proj, float** c_proj, float** c_Q, Process process) {
    long long int N = process.n_filter;

    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    cudaMalloc(c_Q, sizeof(float)*N);

    cudaMalloc(c_proj, process.n_filter * sizeof(float));    
    cudaMemcpy(*c_proj, &proj[process.idx_filter], process.n_filter * sizeof(float), cudaMemcpyHostToDevice);
 
    printf("GPU memory allocated...\n");

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");


    clock_t end = clock();
    printf("Time copy_to_gpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_cpu_filter_conv(float* proj, float* c_proj, float* c_Q, Process process) {
    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    long long int N = process.n_filter;                                         //lab.nbeta * lab.nv * lab.nh;
    cudaMemcpy(&proj[process.idx_filter], c_Q, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_proj);
    cudaFree(c_Q);

    clock_t end = clock();
    printf("Time copy_to_cpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}
