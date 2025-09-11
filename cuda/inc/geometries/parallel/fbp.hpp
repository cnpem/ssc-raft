#ifndef RAFT_FBP_PAR_H
#define RAFT_FBP_PAR_H

#include "common/configs.hpp"
#include "processing/filters.hpp"

extern "C"{
    void getFBPMultiGPU(DIM tomo, DIM obj, GEO geometry, REC ReconParam,
        int* gpus, int ngpus, float* object, float* tomogram, float* angles);

    void getFBPGPU(DIM tomo, DIM obj, GEO geometry, REC ReconParam, 
    float *object, float *tomogram, float *angles, 
    int sizez, int ngpu);

    void getFBP(REC ReconParam, 
    float *obj, float *tomogram, float *angles, 
    dim3 tomo_size, dim3 obj_size, float pixel);
}


#endif