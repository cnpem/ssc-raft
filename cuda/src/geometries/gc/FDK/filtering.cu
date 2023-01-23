#include "../../../../inc/include.h"
#include "../../../../inc/gp/rings.h"
#include "../../../../inc/gc/fdk.h"
#include "../../../../inc/common/types.hpp"


void filtering(Lab lab, float* proj, cufftComplex* signal, float* W, Process process)
{
    cudaSetDevice(process.i_gpu);
    
    // rings
    if(lab.rings){
        printf("Rings .... %d \n", process.i_gpu);
        ringsgpu_fdk(lab, proj,process);
        cudaDeviceSynchronize();
    }
    
    fft(lab, proj, signal, W, process);

    return;
}


void ringsgpu_fdk( Lab lab, float* data, Process process){   

    Rings(data, lab.nh, lab.nbeta, process.z_filter, -1.0, lab.nh*lab.nbeta);

}


extern "C"{
void copy_gpu_filter(Lab lab, float* proj, float** c_proj, cufftComplex** c_signal, float** W, Process process) {
    long long int N = process.n_filter;

    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaMalloc(c_signal, sizeof(cufftComplex)*N);

    cudaMalloc(c_proj, process.n_filter * sizeof(float));    
    cudaMemcpy(*c_proj, &proj[process.idx_filter], process.n_filter * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(W, lab.nh * sizeof(float));
    filt_W<<< 1, 1>>>(lab, *W);
 
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

