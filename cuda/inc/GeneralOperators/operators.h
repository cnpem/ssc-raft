#ifndef RAFT_OP_H
#define RAFT_OP_H

#include <string.h>
#include "../common/structs.h"


extern "C"{

    __global__ void padding(float *in, cufftComplex *inpadded, float value, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey);

    __global__ void recuperate_padding(cufftComplex *inpadded, float *in, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey);

    __global__ void padding_phase_filters(float *in, cufftComplex *inpadded, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey);

}

#endif