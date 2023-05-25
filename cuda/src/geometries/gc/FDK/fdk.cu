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
void gpu_fdk(   Lab lab, float *recon, float *proj, 
                int* gpus, int ndevs, double *time){	

    int i, n_process;
    
    n_process = memory(lab, ndevs);
    printf("n_process = %d, n_gpus = %d \n", n_process, ndevs);

    Process* process = (Process*) malloc(sizeof(Process)*n_process);
    for(i = 0; i < n_process; i++) set_process(lab, i, &process[i], n_process, gpus, ndevs);

    printf("Filter:\n");
    clock_t f_begin = clock();

    if(lab.fourier == 1){
        set_filtering_fft(lab, proj, n_process, ndevs, process);
    } else {
        set_filtering_conv(lab, proj,  n_process, ndevs, process);
    }

    clock_t f_end = clock();
    time[0] = double(f_end - f_begin)/CLOCKS_PER_SEC;

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    printf("Backproject:\n");
    clock_t b_begin = clock();

    set_backprojection(lab, recon, proj, n_process, ndevs, process);

    clock_t b_end = clock();
    time[1] = double(b_end - b_begin)/CLOCKS_PER_SEC;

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

	free(process);
}}

extern "C"{
void set_backprojection(Lab lab, float* recon, float* proj, int n_process,  int ndevs, Process* process){

    int i, k = 0;
    std::vector<thread> threads_back;
    float* c_proj[ndevs];
    float* c_recon[ndevs];
    float* c_beta[ndevs];

    while(k < n_process){

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++)
                copy_to_gpu_back(lab, proj, recon, &c_proj[i], &c_recon[i], &c_beta[i], process[k+i]);   
            cudaDeviceSynchronize();    
        }

        threads_back.emplace_back(thread( backprojection, lab, c_recon[k%ndevs], c_proj[k%ndevs], c_beta[k%ndevs], process[k])) ;
        k = k+1;

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++) threads_back[i].join();
            threads_back.clear();
            cudaDeviceSynchronize();

            for(i = 0; i < ndevs; i++)
                copy_to_cpu_back(recon, c_proj[i], c_recon[i], c_beta[i], process[k-ndevs+i]); 
        }
    }
}}

extern "C"{
void set_filtering_fft(Lab lab, float* proj, int n_process,  int ndevs, Process* process){

    int i, k = 0;
    std::vector<thread> threads_filt;

    float* c_filter[ndevs];
    cufftComplex* c_signal[ndevs];
    float* c_W[ndevs];

    while(k < n_process){

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++) 
                copy_gpu_filter_fft( lab, proj, &c_filter[i], &c_signal[i], &c_W[i], process[k+i]);   
            cudaDeviceSynchronize();
        }

        threads_filt.emplace_back(thread( fft, lab, c_filter[k%ndevs], c_signal[k%ndevs], c_W[k%ndevs], process[k])) ;
        k = k+1;

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++) threads_filt[i].join();
            threads_filt.clear();
            cudaDeviceSynchronize();

            for(i = 0; i < ndevs; i++) 
                copy_cpu_filter_fft(proj, c_filter[i], c_signal[i], c_W[i], process[k-ndevs+i]);            
        }
    }
}}

extern "C"{
void set_filtering_conv(Lab lab, float* proj, int n_process,  int ndevs, Process* process){

    int i, k = 0;
    std::vector<thread> threads_filt;

    float* c_filter[ndevs];
    float* c_Q[ndevs];

    while(k < n_process){

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++) 
                copy_gpu_filter_conv( lab, proj, &c_filter[i], &c_Q[i], process[k+i]);   
            cudaDeviceSynchronize();
        }

        threads_filt.emplace_back(thread( filtering_conv, lab, c_filter[k%ndevs], c_Q[k%ndevs], process[k])) ;
        k = k+1;

        if(k % ndevs == 0){
            for(i = 0; i < ndevs; i++) threads_filt[i].join();
            threads_filt.clear();
            cudaDeviceSynchronize();

            for(i = 0; i < ndevs; i++) 
                copy_cpu_filter_conv(proj, c_filter[i], c_Q[i], process[k-ndevs+i]);            
        }
    }
}}

