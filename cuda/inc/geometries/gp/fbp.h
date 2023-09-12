#ifndef RAFT_FBP_H
#define RAFT_FBP_H

#include <string.h>
#include "../../common/structs.h"
#include "../../common/operations.hpp"
#include "../../GeneralOperators/filter.h"


extern "C"{
    void GPUFBP(char* blockRecon, float* sinoblock, int nrays, int nangles, int isizez, int sizeimage, 
                    int csino, struct CFilter reg, struct EType datatype, float threshold, float* angs, bool bShiftCenter);
    
    void fbpsingleGPU(int gpu, float* blockRecon, float* sinoblock, int nrays, 
            int nangles, int isizez, int sizeimage, int csino, float reg_val, int FilterType, float* angs, int bShiftCenter);  

    void fbpgpu(int gpu, float* recon, float* tomogram, int nrays, int nangles, int nslices, int reconsize, int centersino,
        float reg_val, float* angles, float threshold, int reconPrecision, int FilterType, int bShiftCenter); 

    void fbpblock(int* gpus, int ngpus, float* recon, float* tomogram, int nrays, int nangles, int nslices, int reconsize, int centersino,
        float reg_val, float* angles, float threshold, int reconPrecision, int FilterType, int bShiftCenter); 
}


#endif