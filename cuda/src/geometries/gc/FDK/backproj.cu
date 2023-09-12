#include "../../../../inc/geometries/gc/fdk.h"
#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>

extern "C"{
__global__ void backproj(float* recon, float* proj, float* beta, Lab lab, Process process){

    long long int n = blockDim.x * blockIdx.x + threadIdx.x ;
    long long int idx;
    int i, j, k, m;

    float    x, y, z;
    float u, v, X, Z;
    float cosb, sinb, Q, L;

    int xi, zk;

    set_recon_idxs(n, &i, &j, &k, lab);
    x = -lab.x + i*lab.dx;
    y = -lab.y + j*lab.dy;
    z = process.z_ph + k*lab.dz;
	
    recon[n] = 0.0;

    L = sqrtf(x*x + y*y);
    // if( L <= lab.x && L <= lab.y){
    for(m = 0; m < lab.nbeta; m++){

        cosb = beta[m];
        sinb = beta[m+lab.nbeta];

        u = x*cosb - y*sinb;
        v = x*sinb + y*cosb;

        X = + lab.Dsd*u/(lab.D + v);
        Z = + lab.Dsd*z/(lab.D + v);    

        xi = (int) ((X + lab.h)/lab.dh);
        zk = (int) ((Z - process.z_det)/lab.dv);
	
        if( xi < 0) continue;             
        if( xi >= lab.nh) continue; 
        if( zk < 0) continue;             
        if( zk + process.zi >= lab.nv) continue; 

        idx = (long long int) zk*lab.nbeta*lab.nh + m*lab.nh + xi; 

        Q = proj[idx];   
        recon[n] = recon[n] + Q*__powf(lab.Dsd/(lab.D + v), 2);
        // recon[n] = recon[n] + Q*__powf(lab.Dsd/(lab.D + x*sinb - y*cosb), 2);
    }
    recon[n] = recon[n]*lab.dbeta / 2.0;
    // }
}}

extern "C"{
void copy_to_gpu_back(Lab lab, float* proj, float* recon, float *angles, float** c_proj, float** c_recon, float** c_beta, Process process) {
    long long int N,M;	
    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    N = process.n_recon;
    M = process.n_proj;     //lab.nx * lab.ny * lab.nz;

    float *dangles;
    cudaMalloc((void **)&dangles, lab.nbeta * sizeof(float));
    cudaMemcpy(dangles, angles, lab.nbeta * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    printf("Allocating gpu memory...");
    cudaMalloc(c_recon, N * sizeof(float));

    cudaMalloc(c_proj, M * sizeof(float));     
    cudaMemcpy(*c_proj, &proj[process.idx_proj], M * sizeof(float), cudaMemcpyHostToDevice);
 
    printf("GPU memory allocated...\n");
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    cudaMalloc(c_beta, 2* lab.nbeta * sizeof(float));
    set_beta<<< 1, 1 >>>(lab,dangles,*c_beta);

    cudaFree(dangles);

    clock_t end = clock();
    printf("Time copy_to_gpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_to_cpu_back(float* recon, float* c_proj, float* c_recon, float* c_beta, Process process) {
    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");


    long long int N = process.n_recon;    //lab.nbeta * lab.nv * lab.nh;
    cudaMemcpy(&recon[process.idx_recon], c_recon, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_proj);
    cudaFree(c_recon);
    cudaFree(c_beta);
    clock_t end = clock();
    printf("Time copy_to_cpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void backprojection(Lab lab, float* recon, float* proj, float* beta,  Process process) {
    long long int M;	
    long int n_blocks;
    int n_threads;

    M = process.n_recon;   //lab.nx * lab.ny * lab.nz;
    
    n_threads = NUM_THREADS;
    n_blocks  = M/n_threads + (M % n_threads == 0 ? 0:1);   
    

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    cudaSetDevice(process.i_gpu);

    printf("\n Starting Backprojection: GPU %d \n", process.i_gpu);

    clock_t b_begin = clock();

    backproj<<<n_blocks, n_threads>>>(recon, proj, beta, lab, process);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    clock_t b_end = clock();
    printf("Time backproj: Gpu %d ---- %f \n",process.i_gpu, double(b_end - b_begin)/CLOCKS_PER_SEC);
}}


extern "C"{
__global__ void set_beta(Lab lab, float *dangles, float* beta){

    for(int m = 0; m < lab.nbeta; m++){
        // beta[m] = cosf(lab.dbeta*m);
        // beta[m + lab.nbeta] = sinf(lab.dbeta*m);
        // printf("beta[%d] = %e \n",m,lab.dbeta*m);
        beta[m] = cosf(dangles[m]);
        beta[m + lab.nbeta] = sinf(dangles[m]);
    }
}}



extern "C"{
__device__ void set_recon_idxs(long long int n, int* i, int*j, int* k, Lab lab) {
    long int nij, rem_ij;
    nij = lab.nx*lab.ny;
    *k = (n) / nij;    
    rem_ij = (n) % nij;
    *j = rem_ij / lab.nx;
    *i = rem_ij % lab.nx;
}}
