#include "geometries/conebeam/fdk.hpp"
#include "common/logerror.hpp"
#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>

extern "C"{
__host__ void fft(Lab lab, float* proj, cufftComplex* signal, float* W, Process process){
    int npad = lab.nph;
    long long int batch_pad = process.z_filter_pad*lab.nbeta;
    long long int Npad = process.n_filter_pad;

    int n_threads = NUM_THREADS;
    long long int n_blocks_pad  = Npad/n_threads + (Npad % n_threads == 0 ? 0:1);

    HANDLE_ERROR(cudaSetDevice(process.i_gpu));

    int vpad[] = {npad};
    cufftHandle plan;
    HANDLE_FFTERROR(cufftPlanMany(&plan, 1, vpad, vpad, 1, npad, vpad, 1, npad, CUFFT_C2C, batch_pad));

    //Calculate Signal
    signal_save<<<n_blocks_pad, n_threads>>>(lab, proj, signal, process);

    HANDLE_FFTERROR(cufftExecC2C(plan,(cufftComplex*) signal,(cufftComplex*) signal, CUFFT_FORWARD));

    signal_filter<<<n_blocks_pad, n_threads>>>(lab, W, signal, process);

    HANDLE_FFTERROR(cufftExecC2C(plan,(cufftComplex*) signal,(cufftComplex*) signal, CUFFT_INVERSE));

    signal_inv<<<n_blocks_pad, n_threads>>>(lab, proj, signal, process);

    HANDLE_FFTERROR(cufftDestroy(plan)); 
}}

extern "C"{
__global__ void signal_save(Lab lab, float* proj, cufftComplex* signal, Process process){
    long long int npad = blockDim.x * blockIdx.x + threadIdx.x;
    int i, j, k; 
    long long int idx;
    float X, Z, aux;

    set_filter_idxs(npad, &k, &j, &i, lab, process);
    idx = (long long int) k  + j*lab.nph + i*lab.nph*lab.nbeta;

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
    idx = (long long int) k + j*lab.nph + i * lab.nph * lab.nbeta;   

    signal[idx].x = signal[idx].x * W[k];
    signal[idx].y = signal[idx].y * W[k];       
}}

extern "C"{
__global__ void signal_inv(Lab lab, float* Q, cufftComplex* signal, Process process){
    long long int npad = blockDim.x * blockIdx.x + threadIdx.x ;
    int i,j,k;
    long long int idx;

    set_filter_idxs(npad, &k, &j, &i, lab, process);
    idx = (long long int) k  + j*lab.nph + i*lab.nph*lab.nbeta;

    Q[idx] = signal[idx].x/lab.nph;
}}

extern "C"{
__device__ void set_filter_idxs(long long int n, int* i, int*j, int* k, Lab lab, Process process) {
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

        float magn = lab.Dsd / lab.D;
        float z2   = lab.Dsd - lab.D;
        float wavelenght = (lab.reg == 0.0 ? 1.0:( ( plank * vc ) / lab.energy ) );

        float kernelX = wavelenght * z2 * float(M_PI) * lab.reg / magn;

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (wmax)/(lab.nph-1) + (2*i*wmax)/(lab.nph-1); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph-1);

        for (i = 0; i <=lab.nph/2 ; i++) W[i] = W[i] * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph / 2 + i] * ( 1.0f / ( 1.0f + kernelX * W[lab.nph / 2 + i] * W[lab.nph / 2 + i] ) );
    }

    __global__ void filt_W(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        
        float magn = lab.Dsd / lab.D;
        float z2 = lab.Dsd - lab.D;
        float wavelenght = (lab.reg == 0.0 ? 1.0:( ( plank * vc ) / lab.energy ) );

        float kernelX = wavelenght * z2 * float(M_PI) * lab.reg / magn;
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);
    
        for (i = 0; i <=lab.nph/2 ; i++) W[i] = W[i]*(0.54 + 0.46*cosf(2*M_PI*i/lab.nph)) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i]*(0.54 + 0.46*cosf(2*M_PI*(lab.nph/2 - i)/lab.nph)) * ( 1.0f / ( 1.0f + kernelX * W[lab.nph / 2 + i] * W[lab.nph / 2 + i] ) );
    }

    __global__ void filt_Gaussian(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        float c = 0.693f;

        float magn = lab.Dsd / lab.D;
        float z2 = lab.Dsd - lab.D;
        float wavelenght = (lab.reg == 0.0 ? 1.0:( ( plank * vc ) / lab.energy ) );

        float kernelX = wavelenght * z2 * float(M_PI) * lab.reg / magn;

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = W[i]*expf(-c*(i/lab.nph)*(i/lab.nph)) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i]*expf(-c*((lab.nph/2 - i)/lab.nph)*((lab.nph/2 - i)/lab.nph)) * ( 1.0f / ( 1.0f + kernelX * W[lab.nph / 2 + i] * W[lab.nph / 2 + i] ) );
    }

    __global__ void filt_Lorentz(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);

        float magn = lab.Dsd / lab.D;
        float z2 = lab.Dsd - lab.D;
        float wavelenght = (lab.reg == 0.0 ? 1.0:( ( plank * vc ) / lab.energy ) );

        float kernelX = wavelenght * z2 * float(M_PI) * lab.reg / magn;
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = ( W[i]/(1.0 + (i/lab.nph)*(i/lab.nph)) ) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = ( W[lab.nph/2 + i]/(1.0 + ((lab.nph/2 - i)/lab.nph)*((lab.nph/2 - i)/lab.nph)) ) * ( 1.0f / ( 1.0f + kernelX * W[lab.nph / 2 + i] * W[lab.nph / 2 + i] ) );

    }

    __global__ void filt_Cosine(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);

        float magn = lab.Dsd / lab.D;
        float z2 = lab.Dsd - lab.D;
        float wavelenght = (lab.reg == 0.0 ? 1.0:( ( plank * vc ) / lab.energy ) );

        float kernelX = wavelenght * z2 * float(M_PI) * lab.reg / magn;
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nh/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = W[i] * cosf( float(M_PI)*0.5f*(i/lab.nph)) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i] * cosf( float(M_PI)*0.5f*((lab.nph/2 - i)/lab.nph)) * ( 1.0f / ( 1.0f + kernelX * W[lab.nph / 2 + i] * W[lab.nph / 2 + i] ) );
    }

    __global__ void filt_Rectangle(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);
        float param;

        float magn = lab.Dsd / lab.D;
        float z2 = lab.Dsd - lab.D;
        float wavelenght = (lab.reg == 0.0 ? 1.0:( ( plank * vc ) / lab.energy ) );

        float kernelX = wavelenght * z2 * float(M_PI) * lab.reg / magn;
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <= lab.nph/2 ; i++) {
            param = fmaxf((i/lab.nph) * float(M_PI) * 0.5f, 1E-4f);
            W[i] = ( W[i] * sinf(param) / param ) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );
        }

        for (i = 1; i < lab.nph/2 ; i++) {
            param = fmaxf(((lab.nph/2 - i)/lab.nph) * float(M_PI) * 0.5f, 1E-4f);
            W[lab.nph/2 + i] = ( W[lab.nph/2 + i] * (param) / param) * ( 1.0f / ( 1.0f + kernelX * W[lab.nph / 2 + i] * W[lab.nph / 2 + i] ) );

        }

    }

    __global__ void filt_Hann(Lab lab, float* W){
        int i;
        float wmax = 1.0/(2.0*lab.dh);

        float magn = lab.Dsd / lab.D;
        float z2 = lab.Dsd - lab.D;
        float wavelenght = (lab.reg == 0.0 ? 1.0:( ( plank * vc ) / lab.energy ) );

        float kernelX = wavelenght * z2 * float(M_PI) * lab.reg / magn; 
        
        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2*i*wmax)/(lab.nph); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = wmax - (2*i*wmax)/(lab.nph);

        for (i = 0; i <=lab.nph/2 ; i++) W[i] = W[i] * (0.5 + 0.5*cosf(2*M_PI*i/lab.nph)) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph/2 + i] * (0.5 + 0.5*cosf(2*M_PI*(lab.nph/2 - i)/lab.nph)) * ( 1.0f / ( 1.0f + kernelX * W[lab.nph / 2 + i] * W[lab.nph / 2 + i] ) );

    }

    __global__ void filt_Hamming(Lab lab, float* W){
        int i;
        float wmax = 1.0f / ( 2.0f * lab.dh );

        float magn = lab.Dsd / lab.D;
        float z2 = lab.Dsd - lab.D;
        float wavelenght = (lab.reg == 0.0 ? 1.0:( ( plank * vc ) / lab.energy ) );

        float kernelX = wavelenght * z2 * float(M_PI) * lab.reg / magn; 

        for (i = 0; i <= lab.nph/2 ; i++) W[i] = (2.0f * i * wmax) / ( lab.nph ); 
    
        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph / 2 + i] = wmax - ( ( 2.0f * i * wmax ) / lab.nph );

        for (i = 0; i <=lab.nph/2 ; i++) W[i] = ( W[i] * ( 0.54f + 0.46f * cosf( 2.0f * float(M_PI) * i / lab.nph ) ) ) * ( 1.0f / ( 1.0f + kernelX * W[i] * W[i] ) );

        for (i = 1; i < lab.nph/2 ; i++) W[lab.nph/2 + i] = W[lab.nph / 2 + i] * (0.54f + 0.46f * cosf( 2.0f * M_PI * ( lab.nph / 2 - i ) / lab.nph) ) * ( 1.0f / ( 1.0f + kernelX * W[lab.nph / 2 + i] * W[lab.nph / 2 + i] ) );
    }

}
