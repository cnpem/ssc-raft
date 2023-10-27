#include "../../../../inc/geometries/gc/fdk.h"
#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>

extern "C"{
__host__ void fft(Lab lab, float* proj, cufftComplex* signal, float* W, Process process){
    int n = lab.nh, npad = lab.nph;
    long long int batch = process.z_filter*lab.nbeta;
    long long int batch_pad = process.z_filter_pad*lab.nbeta;
    long long int N = process.n_filter;
    long long int Npad = process.n_filter_pad;

    int n_threads = NUM_THREADS;
    long long int n_blocks  = N/n_threads + (N % n_threads == 0 ? 0:1);
    long long int n_blocks_pad  = Npad/n_threads + (Npad % n_threads == 0 ? 0:1);

    printf("FFT: n_threads = %d, n_blocks = %ld, n_blocks_pad = %ld \n",n_threads,n_blocks,n_blocks_pad);
    printf("FFT: n = %d, npad = %d \n",n,npad);

    cudaSetDevice(process.i_gpu);

    int vpad[] = {npad};
    cufftHandle plan;
    cufftPlanMany(&plan, 1, vpad, vpad, 1, npad, vpad, 1, npad, CUFFT_C2C, batch_pad);

    //Calculate Signal
    signal_save_pad<<<n_blocks_pad, n_threads>>>(lab, proj, signal, process);

    //Forward FFT
    printf("Forward FFT...\n");

    cufftExecC2C(plan,(cufftComplex*) signal,(cufftComplex*) signal, CUFFT_FORWARD);

    signal_filter_pad<<<n_blocks_pad, n_threads>>>(lab, W, signal, process);

    printf("Inverse FFT...\n");
    cufftExecC2C(plan,(cufftComplex*) signal,(cufftComplex*) signal, CUFFT_INVERSE);

    signal_inv_pad<<<n_blocks_pad, n_threads>>>(lab, proj, signal, process);

    cufftDestroy(plan); 
}}

extern "C"{
__host__ void fft_nopad(Lab lab, float* proj, cufftComplex* signal, float* W, Process process){
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
    long long int n = blockDim.x * blockIdx.x + threadIdx.x;
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
__global__ void signal_save_pad(Lab lab, float* proj, cufftComplex* signal, Process process){
    long long int npad = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j, k, kk; 
    long long int idx, idp;
    float X, Z, aux;

    set_filter_idxs_pad(npad, &k, &j, &i, lab, process);
    idp = (long long int) k  + j*lab.nph + i*lab.nph*lab.nbeta;

    kk = ( k - lab.padh );
    idx = (long long int) kk + j*lab.nh + i*lab.nh*lab.nbeta;

    X = kk*lab.dh - lab.h;
    Z = (i + process.i*process.z_filter)*lab.dv - lab.v;
    aux = lab.D/sqrtf(lab.Dsd*lab.Dsd + Z*Z + X*X);

    signal[idp].x = 0.0;
    signal[idp].y = 0.0;

    if ( (kk < 0) || (kk >= lab.nh) ) return;

    signal[idp].x = proj[idx]*aux;
    signal[idp].y = 0.0;
}}

extern "C"{
__global__ void signal_filter_pad(Lab lab, float* W, cufftComplex* signal, Process process){
    long long int n = blockDim.x * blockIdx.x + threadIdx.x ;
    int i,j,k;
    long long int idx;

    set_filter_idxs_pad(n, &k, &j, &i, lab, process);
    idx = (long long int) k + j*lab.nph + i * lab.nph * lab.nbeta;

    signal[idx].x = signal[idx].x*W[k];
    signal[idx].y = signal[idx].y*W[k];      
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
__global__ void signal_inv_pad(Lab lab, float* Q, cufftComplex* signal, Process process){
    long long int npad = blockDim.x * blockIdx.x + threadIdx.x ;
    int i,j,k,kk;
    long long int idx, idp;

    set_filter_idxs_pad(npad, &k, &j, &i, lab, process);
    idp = (long long int) k  + j*lab.nph + i*lab.nph*lab.nbeta;

    kk = ( k - lab.padh );

    idx = (long long int) kk + j*lab.nh + i*lab.nh*lab.nbeta;

    if ( (kk < 0) || (kk >= lab.nh) ) return;

    Q[idx] = signal[idp].x/lab.nph;
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
__device__ void set_filter_idxs_pad(long long int n, int* i, int*j, int* k, Lab lab, Process process) {
    long int nij, rem_ij;
    nij = lab.nph*lab.nbeta;
    *k = (n) / nij;
    rem_ij = (n) % nij;
    *j = rem_ij / lab.nph;
    *i = rem_ij % lab.nph;
}}

extern "C"{

    __global__ void filt_Ramp(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (wmax)/(lab.nph-1) + (2*i*wmax)/(lab.nph-1); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph-1);
    }

    __global__ void filt_W(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);
    
        for (i = 0; i <=lab.nph/2 ; i++) W[i] = W[i]*(0.54 + 0.46*cosf(2*M_PI*i/lab.nph));
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i]*(0.54 + 0.46*cosf(2*M_PI*(lab.nph/2 - i)/lab.nph));
    }

    __global__ void filt_Gaussian(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        float w, c = 0.693f;

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = W[i]*expf(-c*lab.reg*(i/lab.nph)*(i/lab.nph));

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i]*expf(-c*lab.reg*((lab.nph/2 - i)/lab.nph)*((lab.nph/2 - i)/lab.nph));

    }

    __global__ void filt_Lorentz(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        float w;
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = W[i]/(1.0 + lab.reg*(i/lab.nph)*(i/lab.nph));

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i]/(1.0 + lab.reg*((lab.nph/2 - i)/lab.nph)*((lab.nph/2 - i)/lab.nph));

    }

    __global__ void filt_Cosine(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = W[i]*cosf( float(M_PI)*0.5f*(i/lab.nph));

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i]*cosf( float(M_PI)*0.5f*((lab.nph/2 - i)/lab.nph));
    }

    __global__ void filt_Rectangle(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        float param;
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <= lab.nph/2 ; i++) {
            param = fmaxf((i/lab.nph) * lab.reg * float(M_PI) * 0.5f, 1E-4f);
            W[i] = W[i]*sinf(param) / param;
        }

        for (i = 1; i < lab.nph/2 ; i++) {
            param = fmaxf(((lab.nph/2 - i)/lab.nph) * lab.reg * float(M_PI) * 0.5f, 1E-4f);
            W[lab.nph/2 + i] = W[lab.nph/2 + i]*sinf(param) / param;

        }

    }

    __global__ void filt_Hann(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <=lab.nph/2 ; i++) W[i] = W[i]*(0.5 + 0.5*cosf(2*M_PI*i/lab.nph));

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i]*(0.5 + 0.5*cosf(2*M_PI*(lab.nph/2 - i)/lab.nph));

    }

    __global__ void filt_Hamming(Lab lab, float* W){
        int i;
        float wmax = 1.0f / ( 2.0f * lab.dh );

        float magnx = lab.Dsd / lab.D;
        float z2x   = lab.Dsd - lab.D;
        
        float gamma = ( lab.reg == 0.0 ? 0.0:(1.0f / lab.reg) ) ;

        float kernelX = 4.0f * float(M_PI) * float(M_PI) * z2x * gamma  / ( magnx );

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2.0f * i * wmax) / ( lab.nph ); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph / 2 + i] = wmax - ( ( 2.0f * i * wmax ) / lab.nph );

        for (i = 0; i <=lab.nph/2 ; i++) W[i] = ( W[i] * ( 0.54f + 0.46f * cosf( 2.0f * float(M_PI) * i / lab.nph ) ) ) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph / 2 + i] * (0.54f + 0.46f * cosf( 2.0f * M_PI * ( lab.nph / 2 - i ) / lab.nph) ) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );

        // for (i = 0; i <=lab.nph/2 ; i++) W[i] = ( W[i] * ( 0.54f + 0.46f * cosf( 2.0f * float(M_PI) * i / lab.nph ) ) );

        // for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph / 2 + i] * (0.54f + 0.46f * cosf( 2.0f * M_PI * ( lab.nph / 2 - i ) / lab.nph) );
    }

}