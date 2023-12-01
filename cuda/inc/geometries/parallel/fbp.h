#ifndef RAFT_FBP_H
#define RAFT_FBP_H

#include <string.h>
#include "../../common/structs.h"
#include "../../common/operations.hpp"
#include "../../processing/filter.h"


extern "C"{

    void getFBPMultiGPU(float* recon, float* tomogram, float* angles, float *paramf, int *parami, int* gpus, int ngpus);

    void getFBPGPU(CFG configs, GPU gpus, float* recon, float* tomogram, float* angles, int sizez, int ngpu);

    void getFBP(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, dim3 tomo_size, dim3 tomo_pad, dim3 recon_size);

    void getFBP_thresh(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, dim3 tomo_size, dim3 tomo_pad, dim3 recon_size);

    __global__ void BackProjection_RT(float* recon, const float *tomo, const float* sintable, const float* costable,
    dim3 recon_size, dim3 tomo_size);

    __global__ void BackProjection_RT_thresh(float* recon, const float *tomo, const float* sintable, const float* costable,
    dim3 recon_size, dim3 tomo_size, float threshold, EType datatype); 

    __global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles);

}


#endif