#include "geometries/conebeam/fdk.hpp"
#include "common/logerror.hpp"
#include "common/opt.hpp"
#include <stdio.h>
#include <cufft.h>

#include <iostream>
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <string>

extern "C"{
__global__ void backproj(float* recon, float* proj, float* beta, Lab lab, Process process){

    long long int n = blockDim.x * blockIdx.x + threadIdx.x ;
    long long int idx;
    int i, j, k, m;

    float Rxpad = lab.dx * (float)lab.nph / 2.0f;
    float Rypad = lab.dy * (float)lab.nph / 2.0f;
    float Hxpad = lab.dh * (float)lab.nph / 2.0f;
    float    x, y, z;
    float u, v, X, Z;
    float cosb, sinb, Q;

    int xi, zk;

    if ( n >= process.n_recon_pad ) return;

    set_recon_idxs(n, &i, &j, &k, lab);
    x = -Rxpad + i*lab.dx;
    y = -Rypad + j*lab.dy;
    z = process.z_ph + k*lab.dz;

    int block = process.z_proj;
	
    recon[n] = 0.0;

    for(m = 0; m < lab.nbeta; m++){

        cosb = beta[m];
        sinb = beta[m+lab.nbeta];

        u = x*cosb - y*sinb;
        v = x*sinb + y*cosb;

        X = + lab.Dsd*u/(lab.D + v);
        Z = + lab.Dsd*z/(lab.D + v);    

        xi = (int) ((X + Hxpad)/lab.dh);
        zk = (int) ((Z - process.z_det)/lab.dv);
	
        if( xi < 0) continue;             
        if( xi >= lab.nph) continue; 
        if( zk < 0) continue;    
        if( zk >= block) continue;   
        if( zk + process.zi >= lab.nv) continue; 

        idx = (long long int) zk*lab.nbeta*lab.nph + m*lab.nph + xi; 
        
        Q = proj[idx];   
        recon[n] = recon[n] + Q*__powf(lab.Dsd/(lab.D + v), 2);
        // recon[n] = recon[n] + Q*__powf(lab.Dsd/(lab.D + x*sinb - y*cosb), 2);
    }

    recon[n] = recon[n]*lab.dbeta / 2.0;
    // }
}}

extern "C"{
void copy_to_gpu_back(Lab lab, 
float* proj, float* recon, float *angles, 
float** c_proj, float** c_recon, float** c_beta, Process process) 
{
    long long int N,Npad,M,Mpad;	
    // clock_t begin = clock();
    HANDLE_ERROR(cudaSetDevice(process.i_gpu));

    N    = process.n_recon;
    M    = process.n_proj;
    Npad = process.n_recon_pad;
    Mpad = process.n_proj_pad;       

    float *dangles;
    HANDLE_ERROR(cudaMalloc((void **)&dangles, lab.nbeta * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(dangles, angles, lab.nbeta * sizeof(float), cudaMemcpyHostToDevice));

    float *c_tomo = opt::allocGPU<float>((size_t)M);/* Alloc GPU sinograms to receive from CPU (temporary ptr)*/

    HANDLE_ERROR(cudaDeviceSynchronize()); 

    // printf("Allocating gpu memory... c_recon Npad = %lld \n",Npad);
    HANDLE_ERROR(cudaMalloc(c_recon, Npad * sizeof(float)));

    // printf("Allocating gpu memory... c_proj M = %lld \n",M);
    HANDLE_ERROR(cudaMalloc(c_proj, Mpad * sizeof(float)));     
    HANDLE_ERROR(cudaMemcpy(c_tomo, &proj[process.idx_proj], M * sizeof(float), cudaMemcpyHostToDevice));
 
    /* Extended domain padding of sinograms */
    /* Projection GPUs padded Grd and Blocks */
    dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 TomogridBlock( (int)ceil( lab.nph        / TPBX ) + 1,
                        (int)ceil( lab.nbeta      / TPBY ) + 1,
                        (int)ceil( process.z_proj / TPBZ ) + 1);
    
    /* Copy GPU sinograms to padded GPU sinograms *c_proj*/
    opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(c_tomo, *c_proj, 
                                                        dim3(lab.nh, lab.nbeta, process.z_proj),
                                                        dim3(lab.padh, 0, 0));

    HANDLE_ERROR(cudaMalloc(c_beta, 2 * lab.nbeta * sizeof(float)));
    set_beta<<< 1, 1 >>>(lab,dangles,*c_beta);

    HANDLE_ERROR(cudaFree(dangles));
    HANDLE_ERROR(cudaFree(c_tomo));

    // clock_t end = clock();
    // printf("Time copy_to_gpu: Gpu %d/%d ---- %f \n",process.i_gpu,process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_to_cpu_back(Lab lab, float* recon, float* c_proj, float* c_recon, float* c_beta, Process process) {
    // clock_t begin = clock();

    HANDLE_ERROR(cudaSetDevice(process.i_gpu));
    HANDLE_ERROR(cudaDeviceSynchronize()); 

    long long int N = process.n_recon;    

    float *c_rec = opt::allocGPU<float>((size_t)N);/* Alloc GPU (temporary ptr)*/

    /* Extended domain padding of sinograms */
    /* Projection GPUs padded Grd and Blocks */
    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlock( (int)ceil( lab.nph         / TPBX ) + 1,
                    (int)ceil( lab.nph         / TPBY ) + 1,
                    (int)ceil( process.z_recon / TPBZ ) + 1);
    
    /* Copy GPU sinograms to padded GPU sinograms *c_proj*/
    opt::remove_paddR2R<<<gridBlock,threadsPerBlock>>>(c_recon, c_rec,
                                                        dim3(lab.nx, lab.ny, process.z_recon),
                                                        dim3(lab.padh, lab.padh, 0));

    // printf("gpu = %d, idx recon = %lld. Bloco = %lld \n",process.i_gpu, process.idx_recon,N);
    HANDLE_ERROR(cudaMemcpy(&recon[process.idx_recon], c_rec, N*sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(c_proj));
    HANDLE_ERROR(cudaFree(c_recon));
    HANDLE_ERROR(cudaFree(c_beta));
    HANDLE_ERROR(cudaFree(c_rec));
    
    // clock_t end = clock();
    // printf("Time copy_to_cpu: Gpu %d/%d ---- %f \n",process.i_gpu,process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void backprojection(Lab lab, float* recon, float* proj, float* beta,  Process process) {
    long long int M;	
    long int n_blocks;
    int n_threads;

    M = process.n_recon_pad;   //lab.nx * lab.ny * lab.nz;
    
    n_threads = NUM_THREADS;
    n_blocks  = M/n_threads + (M % n_threads == 0 ? 0:1);   
    
    HANDLE_ERROR(cudaDeviceSynchronize()); 

    HANDLE_ERROR(cudaSetDevice(process.i_gpu));

    // printf("\n Starting Backprojection: GPU %d \n", process.i_gpu);

    // clock_t b_begin = clock();

    backproj<<<n_blocks, n_threads>>>(recon, proj, beta, lab, process);

    HANDLE_ERROR(cudaDeviceSynchronize()); 

    clock_t b_end = clock();
    // printf("Time backproj: Gpu %d ---- %f \n",process.i_gpu, double(b_end - b_begin)/CLOCKS_PER_SEC);
}}


extern "C"{
__global__ void set_beta(Lab lab, float *dangles, float* beta){

    for(int m = 0; m < lab.nbeta; m++){
        beta[m] = cosf(dangles[m]);
        beta[m + lab.nbeta] = sinf(dangles[m]);
    }
}}

extern "C"{
__device__ void set_recon_idxs(long long int n, int* i, int*j, int* k, Lab lab) {
    long int nij, rem_ij;
    nij = lab.nph*lab.nph;
    *k = (n) / nij;    
    rem_ij = (n) % nij;
    *j = rem_ij / lab.nph;
    *i = rem_ij % lab.nph;
}}
