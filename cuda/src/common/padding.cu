#include "../../inc/include.h"
#include "../../../inc/common/ffts.h"


template<typename TypeIn, typename TypeOut>
padType Convolution::setPad(){return paddingType<TypeIn,TypeOut>();}

template<>
padType paddingType<TypeIn,TypeOut>(){return padType::none;}

template<>
padType paddingType<float,float>(){return padType::R2R;}

template<>
padType paddingType<cufftComplex,cufftComplex>(){return padType::C2C;}

    template<>
padType paddingType<cufftComplex,float>(){return padType::C2R;}

    template<>
padType paddingType<float,cufftComplex>(){return padType::R2C;}

template<typename TypeIn, typename TypeOut>
void Convolution::padd(GPU gpus, TypeIn *in, TypeOut *padded, padType type)
{
    switch (type){
        case 0:
            paddC2C<<<gpus.Grd,gpus.BT>>>(in, padded, padval, size, pad);
            break;
        case 1:
            paddR2C<<<gpus.Grd,gpus.BT>>>(in, padded, padval, size, pad);
            break;
        case 2:
            paddC2R<<<gpus.Grd,gpus.BT>>>(in, padded, padval, size, pad);
            break;
        case 3:
            paddR2R<<<gpus.Grd,gpus.BT>>>(in, padded, padval, size, pad);
            break;
        default:
            paddC2C<<<gpus.Grd,gpus.BT>>>(in, padded, padval, size, pad);
            break;
    }
}

template<typename TypeIn, typename TypeOut>
void Convolution::Recpadd(GPU gpus, TypeIn *inpadded, TypeOut *out)
{
    padType _type = Convolution::setPad<TypeIn, TypeOut>();

    switch (_type){
        case 0:
            recuperate_paddC2C<<<gpus.Grd,gpus.BT>>>(inpadded, out, size, pad, dim);
            break;
        case 0:
            recuperate_paddR2C<<<gpus.Grd,gpus.BT>>>(inpadded, out, size, pad, dim);
            break;
        case 0:
            recuperate_paddC2R<<<gpus.Grd,gpus.BT>>>(inpadded, out, size, pad, dim);
            break;
        case 0:
            recuperate_paddR2R<<<gpus.Grd,gpus.BT>>>(inpadded, out, size, pad, dim);
            break;
        default:
            recuperate_paddC2C<<<gpus.Grd,gpus.BT>>>(inpadded, out, size, pad, dim);
            break;
    }
}

__global__ void paddR2C(float *in, cufftComplex *outpadded, 
float value, dim3 size, dim3 padsize)
{
    int padx = (int)( ( padsize.x - size.x ) / 2 );
    int pady = (int)( ( padsize.y - size.y ) / 2 );

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - padx );
    int jj     = (int)( j - pady );

    long long int index  =    size.x * k *    size.y +    size.x * jj + ii;
    long long int indpad = padsize.x * k * padsize.y + padsize.x *  j +  i;

    if ( (i >= padsize.x) || (j >= padsize.y) || (k >= size.z) ) return;

    outpadded[indpad].x = value;
    outpadded[indpad].y = 0.0;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    outpadded[indpad].x = in[index];
}

__global__ void paddC2C(cufftComplex *in, cufftComplex *outpadded, 
float value, dim3 size, dim3 padsize)
{
    int padx = (int)( ( padsize.x - size.x ) / 2 );
    int pady = (int)( ( padsize.y - size.y ) / 2 );

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - padx );
    int jj     = (int)( j - pady );

    long long int index  =    size.x * k *    size.y +    size.x * jj + ii;
    long long int indpad = padsize.x * k * padsize.y + padsize.x *  j +  i;

    if ( (i >= padsize.x) || (j >= padsize.y) || (k >= size.z) ) return;

    outpadded[indpad].x = value;
    outpadded[indpad].y = value;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    outpadded[indpad].x = in[index].x;
    outpadded[indpad].y = in[index].y;
}

__global__ void paddC2R(cufftComplex *in, float *outpadded, 
float value, dim3 size, dim3 padsize)
{
    int padx = (int)( ( padsize.x - size.x ) / 2 );
    int pady = (int)( ( padsize.y - size.y ) / 2 );

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - padx );
    int jj     = (int)( j - pady );

    long long int index  =    size.x * k *    size.y +    size.x * jj + ii;
    long long int indpad = padsize.x * k * padsize.y + padsize.x *  j +  i;

    if ( (i >= padsize.x) || (j >= padsize.y) || (k >= size.z) ) return;

    outpadded[indpad] = value;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    outpadded[indpad] = in[index].x;
}

__global__ void paddR2R(float *in, float *outpadded, 
float value, dim3 size, dim3 padsize)
{
    int padx = (int)( ( padsize.x - size.x ) / 2 );
    int pady = (int)( ( padsize.y - size.y ) / 2 );

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - padx );
    int jj     = (int)( j - pady );

    long long int index  =    size.x * k *    size.y +    size.x * jj + ii;
    long long int indpad = padsize.x * k * padsize.y + padsize.x *  j +  i;

    if ( (i >= padsize.x) || (j >= padsize.y) || (k >= size.z) ) return;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    outpadded[indpad] = in[index];
}

__global__ void recuperate_paddC2R(cufftComplex *inpadded, float *out, 
dim3 size, dim3 padsize, int dim)
{
    size_t norm = (dim - 1) * size.x - (dim - 2) *(size.x * size.y);

    int padx    = (int)( ( padsize.x - size.x ) / 2 );
    int pady    = (int)( ( padsize.y - size.y ) / 2 );

    int i       = blockIdx.x*blockDim.x + threadIdx.x;
    int j       = blockIdx.y*blockDim.y + threadIdx.y;
    int k       = blockIdx.z*blockDim.z + threadIdx.z;

    int ii      = (int)( i - padx );
    int jj      = (int)( j - pady );

    long long int index  =    size.x * k *    size.y +    size.x * jj + ii;
    long long int indpad = padsize.x * k * padsize.y + padsize.x *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index] = inpadded[indpad].x / norm; 
}

__global__ void recuperate_paddC2C(cufftComplex *inpadded, cufftComplex *out, 
dim3 size, dim3 padsize, int dim)
{
    size_t norm = (dim - 1) * size.x - (dim - 2) *(size.x * size.y);

    int padx    = (int)( ( padsize.x - size.x ) / 2 );
    int pady    = (int)( ( padsize.y - size.y ) / 2 );

    int i       = blockIdx.x*blockDim.x + threadIdx.x;
    int j       = blockIdx.y*blockDim.y + threadIdx.y;
    int k       = blockIdx.z*blockDim.z + threadIdx.z;

    int ii      = (int)( i - padx );
    int jj      = (int)( j - pady );

    long long int index  =    size.x * k *    size.y +    size.x * jj + ii;
    long long int indpad = padsize.x * k * padsize.y + padsize.x *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index].x = inpadded[indpad].x / norm;
    out[index].y = inpadded[indpad].y / norm;
}

__global__ void recuperate_paddR2C(float *inpadded, cufftComplex *out, 
dim3 size, dim3 padsize, int dim)
{
    size_t norm = (dim - 1) * size.x - (dim - 2) *(size.x * size.y);

    int padx    = (int)( ( padsize.x - size.x ) / 2 );
    int pady    = (int)( ( padsize.y - size.y ) / 2 );

    int i       = blockIdx.x*blockDim.x + threadIdx.x;
    int j       = blockIdx.y*blockDim.y + threadIdx.y;
    int k       = blockIdx.z*blockDim.z + threadIdx.z;

    int ii      = (int)( i - padx );
    int jj      = (int)( j - pady );

    long long int index  =    size.x * k *    size.y +    size.x * jj + ii;
    long long int indpad = padsize.x * k * padsize.y + padsize.x *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index].x = inpadded[indpad] / norm;        
}

__global__ void recuperate_paddR2R(float *inpadded, float *out, 
dim3 size, dim3 padsize, int dim)
{
    size_t norm = (dim - 1) * size.x - (dim - 2) *(size.x * size.y);

    int padx    = (int)( ( padsize.x - size.x ) / 2 );
    int pady    = (int)( ( padsize.y - size.y ) / 2 );

    int i       = blockIdx.x*blockDim.x + threadIdx.x;
    int j       = blockIdx.y*blockDim.y + threadIdx.y;
    int k       = blockIdx.z*blockDim.z + threadIdx.z;

    int ii      = (int)( i - padx );
    int jj      = (int)( j - pady );

    long long int index  =    size.x * k *    size.y +    size.x * jj + ii;
    long long int indpad = padsize.x * k * padsize.y + padsize.x *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index] = inpadded[indpad] / norm;        
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