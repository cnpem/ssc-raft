#include "../../../../inc/include.h"
#include "../../../../inc/gp/rings.h"
#include "../../../../inc/gc/fdk.h"
#include "../../../../inc/common/types.hpp"


void filtering(Lab lab, float* proj, cufftComplex* signal, float* W, Process process)
{
    cudaSetDevice(process.i_gpu);
    // rings

    printf("Rings .... %d \n", process.i_gpu);

    // Rings(proj, ring.nrays, ring.nangles, ring.nslices, (float) -1, (size_t) 512*512);
    ringsgpu_fdk(process.i_gpu, proj, lab.nh, lab.nbeta, process.z_filter, lab.lambda_rings, lab.ringblocks);

    // fdk()
    fft(lab, proj, signal, W, process);

    return;
}



void ringsgpu_fdk(int gpu, float* data, int nrays, int nangles, int nslices, float lambda_rings, int ringblocks)
{   
    cudaSetDevice(gpu);

    size_t blocksize = min((size_t)nslices,32ul);

    for(size_t bz=0; bz<nslices; bz+=blocksize){
        blocksize = min(blocksize,size_t(nslices)-bz);
        
        for (int m = 0; m < ringblocks / 2; m++){

            Rings(data, nrays, nangles, blocksize, lambda_rings, nrays*nangles);
            size_t offset = nrays*nangles;
            size_t step = (nangles / ringblocks) * nrays;
            float* tomptr = data;

            for (int n = 0; n < ringblocks - 1; n++){
                Rings(data, nrays, nangles, blocksize, lambda_rings, nrays*nangles);

                tomptr += step;
            }
            Rings(tomptr, nrays, nangles%ringblocks + nangles/ringblocks, blocksize, lambda_rings, offset);
        }
    
    }

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

