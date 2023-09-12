#include "../../inc/include.h"
#include "../../inc/common/types.hpp"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{

    __global__ void zeropadding(float *in, cufftComplex *inpadded, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey)
    {
        size_t Npadx = sizex + 2 * padsizex;
        size_t Npady = sizey + 2 * padsizey;

        size_t i      = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j      = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k      = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ii     = ( i - padsizex );
        size_t jj     = ( j - padsizey );
        size_t index  = sizex * k * sizey + sizex * jj + ii;
        size_t indpad = Npadx * k * Npady + Npadx * j + i;
        
        if ( (i >= Npadx) || (j >= Npady) || (k >= sizez) ) return;

        inpadded[indpad].x = 0.0;
        inpadded[indpad].y = 0.0;

        if ( (ii < 0) || (ii >= sizex) || (jj < 0) || (jj >= sizey) ) return;

        inpadded[indpad].x = in[index];

    }

    __global__ void recuperate_zeropadding(cufftComplex *inpadded, float *in, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey)
    {
        size_t Npadx = sizex + 2 * padsizex;
        size_t Npady = sizey + 2 * padsizey;

        size_t i      = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j      = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k      = blockIdx.z*blockDim.z + threadIdx.z;
        
        size_t ii     = ( i - padsizex );
        size_t jj     = ( j - padsizey );
        size_t index  = sizex * k * sizey + sizex * jj + ii;        
        size_t indpad = Npadx * k * Npady + Npadx * j + i;

        if ( (ii < 0) || (ii >= sizex) || (jj < 0) || (jj >= sizey) || (k >= sizez) ) return;

        in[index] = inpadded[indpad].x;
        
    }

}