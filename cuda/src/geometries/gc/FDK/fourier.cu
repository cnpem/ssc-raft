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

    __global__ void filt_Ramp(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nh/2 ; i++) W[i] = (wmax)/(lab.nh-1) + (2*i*wmax)/(lab.nh-1); 
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh-1);
    }

    __global__ void filt_W(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nh/2 ; i++) W[i] = (2*i*wmax)/(lab.nh); 
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh);
    
        // for (i = 0; i <=lab.nh/2 ; i++) W[i] = W[i]*sinf(M_PI*i/lab.nh)/(M_PI*i/lab.nh);
    
        // for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = W[lab.nh/2 + i]*sinf(M_PI*(lab.nh/2-i)/lab.nh)/(M_PI*(lab.nh/2-i)/lab.nh);
    
        for (i = 0; i <=lab.nh/2 ; i++) W[i] = W[i]*(0.54 + 0.46*cosf(2*M_PI*i/lab.nh));
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = W[lab.nh/2 + i]*(0.54 + 0.46*cosf(2*M_PI*(lab.nh/2 - i)/lab.nh));
    
        // for (i = 0; i <=lab.nh/2 ; i++) W[i] = W[i]*(expf(1)/(1-4*(i/lab.nh)*(i/lab.nh)))*expf(-1/(1-4*(i/lab.nh)*(i/lab.nh)));
    
        // for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = W[lab.nh/2 + i]*(expf(1)/(1-4*((lab.nh/2-i)/lab.nh)*((lab.nh/2-i)/lab.nh)))*expf(-1/(1-4*((lab.nh/2-i)/lab.nh)*((lab.nh/2-i)/lab.nh)));
    
    }

    __global__ void filt_Gaussian(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        float w, c = 0.693f;

        for (i = 0; i <= lab.nh/2 ; i++) W[i] = (2*i*wmax)/(lab.nh); 
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh);

        for (i = 0; i <= lab.nh/2 ; i++) W[i] = W[i]*expf(-c*lab.reg*(i/lab.nh)*(i/lab.nh));

        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = W[lab.nh/2 + i]*expf(-c*lab.reg*((lab.nh/2 - i)/lab.nh)*((lab.nh/2 - i)/lab.nh));

    }

    __global__ void filt_Lorentz(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        float w;
        
        for (i = 0; i <= lab.nh/2 ; i++) W[i] = (2*i*wmax)/(lab.nh); 
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh);

        for (i = 0; i <= lab.nh/2 ; i++) W[i] = W[i]/(1.0 + lab.reg*(i/lab.nh)*(i/lab.nh));

        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = W[lab.nh/2 + i]/(1.0 + lab.reg*((lab.nh/2 - i)/lab.nh)*((lab.nh/2 - i)/lab.nh));

    }

    __global__ void filt_Cosine(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nh/2 ; i++) W[i] = (2*i*wmax)/(lab.nh); 
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh);

        for (i = 0; i <= lab.nh/2 ; i++) W[i] = W[i]*cosf( float(M_PI)*0.5f*(i/lab.nh));

        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = W[lab.nh/2 + i]*cosf( float(M_PI)*0.5f*((lab.nh/2 - i)/lab.nh));
    }

    __global__ void filt_Rectangle(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        float param;
        
        for (i = 0; i <= lab.nh/2 ; i++) W[i] = (2*i*wmax)/(lab.nh); 
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh);

        for (i = 0; i <= lab.nh/2 ; i++) {
            param = fmaxf((i/lab.nh) * lab.reg * float(M_PI) * 0.5f, 1E-4f);
            W[i] = W[i]*sinf(param) / param;
        }

        for (i = 1; i < lab.nh/2 ; i++) {
            param = fmaxf(((lab.nh/2 - i)/lab.nh) * lab.reg * float(M_PI) * 0.5f, 1E-4f);
            W[lab.nh/2 + i] = W[lab.nh/2 + i]*sinf(param) / param;

        }

    }

    __global__ void filt_Hann(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nh/2 ; i++) W[i] = (2*i*wmax)/(lab.nh); 
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh);

        for (i = 0; i <=lab.nh/2 ; i++) W[i] = W[i]*(0.5 + 0.5*cosf(2*M_PI*i/lab.nh));

        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = W[lab.nh/2 + i]*(0.5 + 0.5*cosf(2*M_PI*(lab.nh/2 - i)/lab.nh));

    }

    __global__ void filt_Hamming(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nh/2 ; i++) W[i] = (2*i*wmax)/(lab.nh); 
    
        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nh);

        for (i = 0; i <=lab.nh/2 ; i++) W[i] = W[i]*(0.54 + 0.46*cosf(2*M_PI*i/lab.nh));

        for (i = 1; i < lab.nh/2 ; i++) W[lab.nh/2 + i] = W[lab.nh/2 + i]*(0.54 + 0.46*cosf(2*M_PI*(lab.nh/2 - i)/lab.nh));
    }

}
