#ifndef RAFT_OP_H
#define RAFT_OP_H

#include <string.h>
#include "../common/structs.h"


extern "C"{

    __global__ void zeropadding(float *in, cufftComplex *inpadded, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey);

    __global__ void recuperate_zeropadding(cufftComplex *inpadded, float *in, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey);

}

#endif