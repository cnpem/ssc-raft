#include "common/configs.hpp"
#include "common/opt.hpp"

/*============================================================================*/
/* namespace opt (in 'inc/commons/opt.hpp') functions definitions */

__global__ void opt::paddR2C(float *in, cufftComplex *outpadded, 
dim3 size, dim3 pad, float value)
{
    int Npadx = ( 1 + pad.x ) * size.x;
    int Npady = ( 1 + pad.y ) * size.y;

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    long long int index  = IND(i,j,k,size.x,size.y);

    int ipad     = (int)( i + padx / 2 );
    int jpad     = (int)( j + pady / 2 );

    long long int indpad =  Npadx * k * Npady + Npadx * jpad + ipad;

    if ( (ipad >= Npadx) || (jpad >= Npady) || (k >= size.z) ) return;

    outpadded[indpad].x = value;
    outpadded[indpad].y = 0.0;

    if ( (i < 0) || (i >= size.x) || (j < 0) || (j >= size.y)) return;

    outpadded[indpad].x = in[index];
}

__global__ void opt::paddC2C(cufftComplex *in, cufftComplex *outpadded,
dim3 size, dim3 pad, float value)
{
    int Npadx = ( 1 + pad.x ) * size.x;
    int Npady = ( 1 + pad.y ) * size.y;

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i + pad.x * size.x / 2 );
    int jj     = (int)( j + pad.y * size.y / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad =  Npadx * k *  Npady +  Npadx *  j +  i;

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    outpadded[indpad].x = value;
    outpadded[indpad].y = value;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    outpadded[indpad].x = in[index].x;
    outpadded[indpad].y = in[index].y;
}

__global__ void opt::paddC2R(cufftComplex *in, float *outpadded, 
dim3 size, dim3 pad, float value)
{
    int Npadx = ( 1 + pad.x ) * size.x;
    int Npady = ( 1 + pad.y ) * size.y;

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - pad.x * size.x / 2 );
    int jj     = (int)( j - pad.y * size.y / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad =  Npadx * k *  Npady +  Npadx *  j +  i;

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    outpadded[indpad] = value;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    outpadded[indpad] = in[index].x;
}

__global__ void opt::paddR2R(float *in, float *outpadded, 
dim3 size, dim3 pad, float value)
{
    int Npadx = ( 1 + pad.x ) * size.x;
    int Npady = ( 1 + pad.y ) * size.y;

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    long long int index  = IND(i,j,k,size.x,size.y);

    int ipad     = (int)( i + padx / 2 );
    int jpad     = (int)( j + pady / 2 );

    long long int indpad =  Npadx * k *  Npady +  Npadx * jpad +  ipad;

    if ( (ipad >= Npadx) || (jpad >= Npady) || (k >= size.z) ) return;

    if ( (i < 0) || (i >= size.x) || (j < 0) || (j >= size.y) ) return;

    outpadded[indpad] = in[index];
}

__global__ void opt::remove_paddC2R(cufftComplex *inpadded, float *out, 
dim3 size, dim3 pad)
{
    int Npadx = ( 1 + pad.x ) * size.x;
    int Npady = ( 1 + pad.y ) * size.y;

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    long long int index  = IND(i,j,k,size.x,size.y);

    int ipad     = (int)( i + padx / 2 );
    int jpad     = (int)( j + pady / 2 );

    long long int indpad =  Npadx * k *  Npady +  Npadx * jpad +  ipad;

    if ( (i < 0) || (i >= size.x) || (j < 0) || (j >= size.y) || (k >= size.z) ) return;

    out[index] = inpadded[indpad].x;
}

__global__ void opt::remove_paddC2C(cufftComplex *inpadded, cufftComplex *out, 
dim3 size, dim3 pad)
{
    int Npadx = ( 1 + pad.x ) * size.x;
    int Npady = ( 1 + pad.y ) * size.y;

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - pad.x * size.x / 2 );
    int jj     = (int)( j - pad.y * size.y / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad =  Npadx * k *  Npady +  Npadx *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index].x = inpadded[indpad].x;
    out[index].y = inpadded[indpad].y; 
}

__global__ void opt::remove_paddR2C(float *inpadded, cufftComplex *out, 
dim3 size, dim3 pad)
{
    int Npadx = ( 1 + pad.x ) * size.x;
    int Npady = ( 1 + pad.y ) * size.y;

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - pad.x * size.x / 2 );
    int jj     = (int)( j - pad.y * size.y / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad =  Npadx * k *  Npady +  Npadx *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index].x = inpadded[indpad];         
}

__global__ void opt::remove_paddR2R(float *inpadded, float *out, 
dim3 size, dim3 pad)
{
    int Npadx = ( 1 + pad.x ) * size.x;
    int Npady = ( 1 + pad.y ) * size.y;

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    long long int index  = IND(i,j,k,size.x,size.y);

    int ipad     = (int)( i + padx / 2 );
    int jpad     = (int)( j + pady / 2 );

    long long int indpad =  Npadx * k *  Npady +  Npadx * jpad +  ipad;

    if ( (i < 0) || (i >= size.x) || (j < 0) || (j >= size.y) || (k >= size.z) ) return;

    out[index] = inpadded[indpad];   
}

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

