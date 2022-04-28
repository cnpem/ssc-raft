#ifndef REB_H
#define REB_H

#include "../inc/include.h"
    
extern "C"{
    void GPUrebinning(float *conetomo, float *tomo, float *parameters, size_t *volumesize, int *gpus, int ngpus);
}

extern "C"{
    void CPUrebinning(float *conetomo, float *tomo, float *parameters, size_t *volumesize);
}

void cone_rebinning(RDAT *reb, PAR param);
void _GPUrebinning(PAR param, PROF *prof, float *conetomo, float *tomo, int ngpus);

void set_rebinning_parameters_gpu(PAR *param, float *parameters, size_t *volumesize, int *gpus);
void set_rebinning_parameters_cpu(PAR *param, float *parameters, size_t *volumesize);
RDAT *allocate_gpu_rebinning(RDAT *workspace, PAR param);
void free_gpu_rebinning(RDAT *reb);

void cone_rebinning_cpu(float *tomo, float *conetomo, PAR param);

__global__ void cone_rebinning_kernel(float *dtomo, float *dctomo, PAR param);


#endif 