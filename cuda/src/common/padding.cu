#include "../../inc/include.h"

extern "C"{

    __global__ void padding(float *in, cufftComplex *inpadded, float value, dim3 size, dim3 padsize)
    {
        size_t Npadx = size.x + 2 * padsize.x;
        size_t Npady = size.y + 2 * padsize.y;

        int i      = blockIdx.x*blockDim.x + threadIdx.x;
        int j      = blockIdx.y*blockDim.y + threadIdx.y;
        int k      = blockIdx.z*blockDim.z + threadIdx.z;

        int ii     = (int)( i - padsize.x );
        int jj     = (int)( j - padsize.y );

        long long int index  = size.x * k * size.y + size.x * jj + ii;
        long long int indpad = Npadx  * k * Npady  + Npadx  *  j +  i;

        if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

        inpadded[indpad].x = value;
        inpadded[indpad].y = 0.0;

        if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

        inpadded[indpad].x = in[index];

    }

    __global__ void recuperate_padding(cufftComplex *inpadded, float *in, dim3 size, dim3 padsize)
    {
        size_t Npadx = size.x + 2 * padsize.x;
        size_t Npady = size.y + 2 * padsize.y;

        int i      = blockIdx.x*blockDim.x + threadIdx.x;
        int j      = blockIdx.y*blockDim.y + threadIdx.y;
        int k      = blockIdx.z*blockDim.z + threadIdx.z;
        
        int ii     = (int)( i - padsize.x );
        int jj     = (int)( j - padsize.y );

        long long int index  = size.x * k * size.y + size.x * jj + ii;
        long long int indpad = Npadx  * k * Npady  + Npadx  *  j +  i;

        if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

        in[index] = inpadded[indpad].x;
        
    }

}