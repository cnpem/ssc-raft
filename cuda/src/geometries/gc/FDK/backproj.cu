#include "../../../../inc/gc/fdk.h"
#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>

extern "C"{
__global__ void backproj(float* recon, float* proj, Lab lab, Process process){

    long long int n = blockDim.x * blockIdx.x + threadIdx.x ;
    long long int idx;
    int i, j, k, m;

    float b, x, y, z;
    float u, v, X, Z;
    float cosb, sinb, Q;

    int xi, zk;

    set_recon_idxs(n, &i, &j, &k, lab);
    x = -lab.x + i*lab.dx;
    y = -lab.y + j*lab.dy;
    z = process.z_ph + k*lab.dz;
	
    recon[n] = 0.0;

    for(m = 0; m < lab.nbeta; m++){
        b = lab.dbeta*m;
        cosb = __cosf(b);
        sinb = __sinf(b);
        
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
    }
    recon[n] = recon[n]*lab.dbeta/sqrtf(2*M_PI);
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
