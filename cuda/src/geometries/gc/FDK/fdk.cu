#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include "../../../../inc/gc/fdk.h"
#include <fstream>
#include <future>
#include <thread>
#include <time.h>
#include <vector>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
    return 0;
}


extern "C"{
void gpu_fdk(   Lab lab, Rings_ ring, float *recon, float *proj, 
                int* gpus, int ndevs, double *time){	

    int i, k, n_process;
    
    n_process = memory(lab, ndevs);
    printf("n_process = %d, n_gpus = %d \n", n_process, ndevs);

    Process* process = (Process*) malloc(sizeof(Process)*n_process);
    for(i = 0; i < n_process; i++) set_process(lab, i, &process[i], n_process, gpus, ndevs);


    printf("Filter:\n");
    clock_t f_begin = clock();
    k = 0;
    std::vector<thread> threads_filt;

    float* c_filter[ndevs];
    cufftComplex* c_signal[ndevs];
    float* c_W[ndevs];

    while(k < n_process){

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++) 
                copy_gpu_filter(lab, proj, &c_filter[i], &c_signal[i], &c_W[i], process[k+i]);   
            cudaDeviceSynchronize();
        }

        threads_filt.emplace_back(thread( filtering, ring, lab, c_filter[k%ndevs], c_signal[k%ndevs], c_W[k%ndevs], process[k])) ;
        k = k+1;

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++) threads_filt[i].join();
            threads_filt.clear();
            cudaDeviceSynchronize();

            for(i = 0; i < ndevs; i++) 
                copy_cpu_filter(proj, c_filter[i], c_signal[i], c_W[i], process[k-ndevs+i]);            
        }
    }

    clock_t f_end = clock();
    time[0] = double(f_end - f_begin)/CLOCKS_PER_SEC;


    printf("Backproject:\n");
    clock_t b_begin = clock();
    k = 0;
    std::vector<thread> threads_back;
    float* c_proj[ndevs];
    float* c_recon[ndevs];

    while(k < n_process){

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++)
                copy_to_gpu_back(lab, proj, recon, &c_proj[i], &c_recon[i], process[k+i]);   
            cudaDeviceSynchronize();    
        }

        threads_back.emplace_back(thread( backprojection, lab, c_recon[k%ndevs], c_proj[k%ndevs], process[k])) ;
        k = k+1;

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++) threads_back[i].join();
            threads_back.clear();
            cudaDeviceSynchronize();
            for(i = 0; i < ndevs; i++)
                copy_to_cpu_back(recon, c_proj[i], c_recon[i], process[k-ndevs+i]); 
        }
    }

    clock_t b_end = clock();
    time[1] = double(b_end - b_begin)/CLOCKS_PER_SEC;

	free(process);
}}

extern "C"{
void copy_to_gpu_back(Lab lab, float* proj, float* recon, float** c_proj, float** c_recon, Process process) {
    long long int N,M;	
    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    N = process.n_recon;
    M = process.n_proj;                                        //lab.nx * lab.ny * lab.nz;

    printf("Allocating gpu memory...");
    cudaMalloc(c_recon, N * sizeof(float));

    cudaMalloc(c_proj, M * sizeof(float));     
    cudaMemcpy(*c_proj, &proj[process.idx_proj], M * sizeof(float), cudaMemcpyHostToDevice);
 
    printf("GPU memory allocated...\n");
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    clock_t end = clock();
    printf("Time copy_to_gpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_to_cpu_back(float* recon, float* c_proj, float* c_recon,  Process process) {
    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    long long int N = process.n_recon;                                         //lab.nbeta * lab.nv * lab.nh;
    cudaMemcpy(&recon[process.idx_recon], c_recon, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_proj);
    cudaFree(c_recon);
    clock_t end = clock();
    printf("Time copy_to_cpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void backprojection(Lab lab, float* recon, float* proj, Process process) {
    long long int M;	
    long int n_blocks;
    int n_threads;

    M = process.n_recon;                                        //lab.nx * lab.ny * lab.nz;
    
    n_threads = NUM_THREADS;
    n_blocks  = M/n_threads + (M % n_threads == 0 ? 0:1);   
    
    cudaSetDevice(process.i_gpu);

    clock_t b_begin = clock();

    backproj<<<n_blocks, n_threads>>>(recon, proj, lab, process);

    clock_t b_end = clock();
    printf("Time backproj: Gpu %d ---- %f \n",process.i_gpu, double(b_end - b_begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_gpu_filter(Lab lab, float* proj, float** c_proj, cufftComplex** c_signal, float** W, Process process) {
    int n = lab.nh;
    long long int batch = process.z_filter*lab.nbeta;
    long long int N = process.n_filter;

    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaMalloc(c_signal, sizeof(cufftComplex)*N);

    cudaMalloc(c_proj, process.n_filter * sizeof(float));    
    cudaMemcpy(*c_proj, &proj[process.idx_filter], process.n_filter * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(W, lab.nh * sizeof(float));
    filt_W<<<1, 1>>>(lab, *W);
 
    printf("GPU memory allocated...\n");

    clock_t end = clock();
    printf("Time copy_to_gpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_cpu_filter(float* proj, float* c_proj, cufftComplex* c_signal, float* c_W,  Process process) {
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


void filtering(Rings_ ring, Lab lab, float* proj, cufftComplex* signal, float* W, Process process)
{
    // rings
    ringsgpu_fdk(process.i_gpu, proj, ring.nrays, ring.nangles, process.zi_filter, rings.lambda_rings, rings.ringblocks);

    // fdk()
    fft(lab, proj, signal, W, process)

    return
}



