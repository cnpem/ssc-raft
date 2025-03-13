#ifndef RAFT_FBP_PAR_H
#define RAFT_FBP_PAR_H

#include "common/configs.hpp"
#include "processing/filters.hpp"

extern "C"{

    void setFBPParameters(CFG *configs, float *parameters_float, int *parameters_int);
    void printFBPParameters(CFG *configs);

    void getFBPMultiGPU(int* gpus, int ngpus, float* recon, float* tomogram, float* angles, float *paramf, int *parami);

    void getFBPGPU(CFG configs, GPU gpus, float* recon, float* tomogram, float* angles, int sizez, int ngpu);

    void getFBP(CFG configs, GPU gpus, float *recon, float *tomogram, float *dataPadded, float *angles, dim3 tomo_size, dim3 tomo_pad, dim3 recon_size);

}


#endif