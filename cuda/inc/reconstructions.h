#ifndef RAFT_RECON_H
#define RAFT_RECON_H

#include "filters.h"

/* FBP */
extern "C"{

    void getFBPMultiGPU(float* recon, float* tomogram, float* angles, float *paramf, int *parami, int* gpus, int ngpus);

    void getFBPGPU(CFG configs, GPU gpus, float* recon, float* tomogram, float* angles, int sizez, int ngpu);

    void getFBP(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, dim3 tomo_size, dim3 tomo_pad, dim3 recon_size);

    void getFBP_thresh(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, dim3 tomo_size, dim3 tomo_pad, dim3 recon_size);

    __global__ void BackProjection_RT(float* recon, const float *tomo, const float* sintable, const float* costable, dim3 recon_size, dim3 tomo_size);

    __global__ void BackProjection_RT_thresh(float* recon, const float *tomo, const float* sintable, const float* costable, dim3 recon_size, dim3 tomo_size, float threshold, EType datatype); 

    __global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles);

}

/* BST */
extern "C" {
    void BST(float* blockRecon, float *wholesinoblock, int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0);
}

/* EM Parallel */

extern "C"{

  void get_tEM_RT_MultiGPU(float* recon, float* count, float *flat, float* angles, 
  float *paramf, int *parami, int* gpus, int ngpus);

  void get_eEM_RT_MultiGPU(float* recon, float* tomogram, float* angles, 
  float *paramf, int *parami, int* gpus, int ngpus);
  
  void get_tEM_RT_GPU(CFG configs, GPU gpus, float *recon, float *count, float *flat, float *angles, 
  int sizez, int ngpu);

  void get_eEM_RT_GPU(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, 
  int sizez, int ngpu);

  void get_tEM_RT(CFG configs, GPU gpus, float *output, float *count, float *flat, float *angles, int blockSize);

  void get_eEM_RT(CFG configs, GPU gpus, float *output, float *tomo, float *angles, int blockSize);

  void getError_F(float *error, float *x, float *y, int N, int blockSize, int device);
  
  void iterEM( float *em, float *sino, float *sinotmp, float *backones, float *angles,
	       int sizeImage, int nrays, int nangles, int blockSize, int device );
  void iterTV( float *y, float *x, float *backones,
	       int sizeImage, int blockSize, int device, float reg, float epsilon);
  void EMTV(float *output, float *sino, float *angles, 
	    int sizeImage, int nrays, int nangles, int blockSize, int device, int niter,
	    int niter_em, int niter_tv, float reg, float epsilon);

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