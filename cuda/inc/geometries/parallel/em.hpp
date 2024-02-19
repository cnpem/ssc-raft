#ifndef RAFT_EM_PAR_H
#define RAFT_EM_PAR_H

#include "common/configs.hpp"

/* EM Parallel */

extern "C"{

    void setEMParameters(CFG *configs, float *parameters_float, int *parameters_int);
    void printEMParameters(CFG *configs);

    void get_tEM_RT_MultiGPU(int* gpus, int ngpus,
    float* recon, float* count, float *flat, float* angles, 
    float *paramf, int *parami);

    void get_eEM_RT_MultiGPU(int* gpus, int ngpus, 
    float* recon, float* tomogram, float* angles, 
    float *paramf, int *parami);

    void get_tEM_RT_GPU(CFG configs, GPU gpus, float *obj, float *count, float *flat, float *angles, 
    int sizez, int ngpu);

    void get_eEM_RT_GPU(CFG configs, GPU gpus, float *obj, float *tomogram, float *angles, 
    int sizez, int ngpu);

    void get_tEM_RT(CFG configs, GPU gpus, 
    float *output, float *count, float *flat, float *angles, 
    float *backcounts, float *temp, float *back,
    int blockSize);

    void get_eEM_RT(CFG configs, GPU gpus, float *output, float *tomo, float *angles, int blockSize);

}

/* CUDA KERNELS */
extern "C"{

    __global__ void kernel_ones(float *output, int sizeImage, int nrays, int nangles,  int blockSize);
    __global__ void kernel_flatTimesExp(float *tmp, float *flat, int sizeImage, int nrays, int nangles,  int blockSize);
    __global__ void kernel_update(float *output, float *back, float *backcounts, int sizeImage, int nrays, int nangles,  int blockSize);
    __global__ void kernel_difference_F(float *y, float *x, int sizeImage, int blockSize);
    __global__ void kernel_backprojection(float *image, float *blocksino, float *angles, int sizeImage, int nrays, int nangles,  int blockSize);
    __global__ void kernel_radon(float *output, float *input, float *angles, int sizeImage, int nrays, int nangles, int blockSize, float a);
    __global__ void kernel_radonWithDivision(float *output, float *input, float *sino,  float *angles, int sizeImage, int nrays, int nangles, int blockSize, float a);
    __global__ void kernel_backprojectionWithUpdate(float *image, float *blocksino, float *backones, float *angles, int sizeImage, int nrays, int nangles,  int blockSize);
    __global__ void kernel_backprojectionOfOnes(float *backones, float *angles, int sizeImage, int nrays, int nangles,  int blockSize);
    __global__ void kernel_updateTV(float *y, float *x, float *backones, int sizeImage, int blockSize, float reg, float epsilon);

}

#endif