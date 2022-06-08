#ifndef RAFT_FBP_H
#define RAFT_FBP_H

#include <string.h>
#include "common/structs.h"
#include "common/operations.hpp"
#include "filter.h"


extern "C"{
    void GPUFBP(char* blockRecon, float* sinoblock, int nrays, int nangles, int isizez, int sizeimage, 
                    int csino, struct CFilter reg, struct EType datatype, float threshold, float* angs, bool bShiftCenter);
    
    void CPUFBP(int* devv, int ndevs, float* blockRecon, float* sinoblock, int nrays, 
            int nangles, int isizez, int sizeimage, int csino, float reg_val, int FilterType, float* angs, int bShiftCenter);
        
}


#endif