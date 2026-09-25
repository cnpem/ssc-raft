#ifndef RAFT_FDK_CONE_H
#define RAFT_FDK_CONE_H

#define NUM_THREADS 128
#define EPSILON_GX 1e-8

#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include <fstream>
#include <future>
#include <thread>
#include <time.h>
#include <vector>
#include <iostream>

#include "common/configs.hpp"

typedef struct {  
    float x,y,z;
    float dx, dy, dz;
    int nx, ny, nz;
    float h,v;
    float dh, dv;
    int nh, nv;
    float D, Dsd;
    float beta_max;
    float dbeta;
    int nbeta;
    int filter_type; // Filter Types
    float paganin_slices; // Paganin Filter regularization
    float reg; // Filter general regularization
    int nph, padh; // Padding for backprojection
    float energy; 
    float rotation_axis_offset;
    int blocksize;
    int padmode; // Padding for backprojection
    int npfh, padfh, padfmode; // Padding for filter


    /* Filter Types definitions
    enum EType
	{
        none      = 0,
        gaussian  = 1,
        lorentz   = 2,
        cosine    = 3,
        rectangle = 4,
        hann      = 5,
        hamming   = 6,
        ramp      = 7
	};
    */

} Lab;


// FDK Functions

extern "C"{
    void gpu_fdk(Lab lab,  float *recon, float *proj, float *angles, int* gpus, int ndev);
    void set_process(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs);
    int memory(Lab lab, int ndev);

    void set_backprojection(Lab lab, float* recon, float* proj, float *angles, int n_process,  int ndevs, Process* process);
    
    void copy_to_gpu_back(Lab lab, float* proj, float* recon, float *angles, float** c_proj, float** c_recon, float** c_beta, Process process);
    void copy_to_cpu_back(Lab lab, float* recon, float* c_proj, float* c_recon, float* c_beta, Process process);
    
    void backprojection(Lab lab, float* recon, float* proj, float* beta,  Process process);
    void backprojection_metrology(Lab lab, float* recon, float* proj, float* beta,  Process process, float* error_x, float* error_y, float* error_z);

    __global__ void backproj(float* recon, float* proj, float* beta, Lab lab, Process process);
    __global__ void backproj_metrology(float* recon, float* proj, float* beta, float* error_x, float* error_y, float* error_z, Lab lab, Process process);
    __global__ void set_beta(Lab lab, float *dangles, float* beta);
    __device__ void set_recon_idxs(long long int n, int* i, int*j, int* k, Lab lab);

}


#endif