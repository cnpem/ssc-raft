#include "../../inc/include.h"
#include "../../inc/common/types.hpp"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{

    __global__ void padding(float *in, cufftComplex *inpadded, float value, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey)
    {
        size_t Npadx = sizex + 2 * padsizex;
        size_t Npady = sizey + 2 * padsizey;

        int i      = blockIdx.x*blockDim.x + threadIdx.x;
        int j      = blockIdx.y*blockDim.y + threadIdx.y;
        int k      = blockIdx.z*blockDim.z + threadIdx.z;

        int ii     = (int)( i - padsizex );
        int jj     = (int)( j - padsizey );

        long long int index  = sizex * k * sizey + sizex * jj + ii;
        long long int indpad = Npadx * k * Npady + Npadx *  j +  i;

        if ( (i >= Npadx) || (j >= Npady) || (k >= sizez) ) return;

        inpadded[indpad].x = value;
        inpadded[indpad].y = 0.0;

        if ( (ii < 0) || (ii >= sizex) || (jj < 0) || (jj >= sizey) || (k >= sizez) ) return;

        inpadded[indpad].x = in[index];

    }

    __global__ void recuperate_padding(cufftComplex *inpadded, float *in, size_t sizex, size_t sizey, size_t sizez, size_t padsizex, size_t padsizey)
    {
        size_t Npadx = sizex + 2 * padsizex;
        size_t Npady = sizey + 2 * padsizey;

        int i      = blockIdx.x*blockDim.x + threadIdx.x;
        int j      = blockIdx.y*blockDim.y + threadIdx.y;
        int k      = blockIdx.z*blockDim.z + threadIdx.z;
        
        int ii     = (int)( i - padsizex );
        int jj     = (int)( j - padsizey );

        long long int index  = sizex * k * sizey + sizex * jj + ii;
        long long int indpad = Npadx * k * Npady + Npadx *  j +  i;

        if ( (ii < 0) || (ii >= sizex) || (jj < 0) || (jj >= sizey) || (k >= sizez) ) return;

        in[index] = inpadded[indpad].x;
        
    }

}