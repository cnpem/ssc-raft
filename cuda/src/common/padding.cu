#include <cstdio>
#include "common/configs.hpp"
#include "common/opt.hpp"
#include "processing/processing.hpp"

/*============================================================================*/
/* namespace opt (in 'inc/commons/opt.hpp') functions definitions */

__global__ void opt::paddR2C(float *in, cufftComplex *outpadded, 
dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - ( (int)padx / 2 ) );
    int jj     = (int)( j - ( (int)pady / 2 ) );

    long long int index  = size.x * k * size.y + size.x * jj + ii;

    long long int indpad =  Npadx * k * Npady + Npadx * j + i;

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    outpadded[indpad].y = 0.0;

    if ( ( i <=          (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad].x = in[size.x * k * size.y + size.x *            0 +            0];
    if ( ( i >= size.x + (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad].x = in[size.x * k * size.y + size.x *            0 + (size.x - 1)];
    if ( ( i <=          (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) +            0];
    if ( ( i >= size.x + (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) + (size.x - 1)];

    if ( (jj < 0) || (jj >= size.y) ) return;

    if ( ( i <=          (int)padx/2 ) ) outpadded[indpad].x = in[size.x * k * size.y + size.x * jj +            0];
    if ( ( i >= size.x + (int)padx/2 ) ) outpadded[indpad].x = in[size.x * k * size.y + size.x * jj + (size.x - 1)];

    if ( (ii < 0) || (ii >= size.x) ) return;

    if ( ( j <=          (int)pady/2 ) ) outpadded[indpad].x = in[size.x * k * size.y + size.x *            0 + ii];
    if ( ( j >= size.y + (int)pady/2 ) ) outpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) + ii];

    outpadded[indpad].x = in[index];
    
}

__global__ void opt::paddC2C(cufftComplex *in, cufftComplex *outpadded,
dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - (int)padx / 2 );
    int jj     = (int)( j - (int)pady / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad =  Npadx * k *  Npady +  Npadx *  j +  i;

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    if ( ( i <=          (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad].x = in[size.x * k * size.y + size.x *            0 +            0].x;
    if ( ( i >= size.x + (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad].x = in[size.x * k * size.y + size.x *            0 + (size.x - 1)].x;
    if ( ( i <=          (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) +            0].x;
    if ( ( i >= size.x + (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) + (size.x - 1)].x;

    if ( ( i <=          (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad].y = in[size.x * k * size.y + size.x *            0 +            0].y;
    if ( ( i >= size.x + (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad].y = in[size.x * k * size.y + size.x *            0 + (size.x - 1)].y;
    if ( ( i <=          (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad].y = in[size.x * k * size.y + size.x * (size.y - 1) +            0].y;
    if ( ( i >= size.x + (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad].y = in[size.x * k * size.y + size.x * (size.y - 1) + (size.x - 1)].y;

    if ( (jj < 0) || (jj >= size.y) ) return;

    if ( ( i <=          (int)padx/2 ) ) outpadded[indpad].x = in[size.x * k * size.y + size.x * jj +            0].x;
    if ( ( i >= size.x + (int)padx/2 ) ) outpadded[indpad].x = in[size.x * k * size.y + size.x * jj + (size.x - 1)].x;
    if ( ( i <=          (int)padx/2 ) ) outpadded[indpad].y = in[size.x * k * size.y + size.x * jj +            0].y;
    if ( ( i >= size.x + (int)padx/2 ) ) outpadded[indpad].y = in[size.x * k * size.y + size.x * jj + (size.x - 1)].y;

    if ( (ii < 0) || (ii >= size.x) ) return;

    if ( ( j <=          (int)pady/2 ) ) outpadded[indpad].x = in[size.x * k * size.y + size.x *            0 + ii].x;
    if ( ( j >= size.y + (int)pady/2 ) ) outpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) + ii].x;
    if ( ( j <=          (int)pady/2 ) ) outpadded[indpad].y = in[size.x * k * size.y + size.x *            0 + ii].y;
    if ( ( j >= size.y + (int)pady/2 ) ) outpadded[indpad].y = in[size.x * k * size.y + size.x * (size.y - 1) + ii].y;

    outpadded[indpad].x = in[index].x;
    outpadded[indpad].y = in[index].y;
}

__global__ void opt::paddC2R(cufftComplex *in, float *outpadded,
        dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - (int)padx / 2 );
    int jj     = (int)( j - (int)pady / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad =  Npadx * k *  Npady +  Npadx *  j +  i;

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    if ( ( i <=          (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad] = in[size.x * k * size.y + size.x *            0 +            0].x;
    if ( ( i >= size.x + (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad] = in[size.x * k * size.y + size.x *            0 + (size.x - 1)].x;
    if ( ( i <=          (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad] = in[size.x * k * size.y + size.x * (size.y - 1) +            0].x;
    if ( ( i >= size.x + (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad] = in[size.x * k * size.y + size.x * (size.y - 1) + (size.x - 1)].x;

    if ( (jj < 0) || (jj >= size.y) ) return;

    if ( ( i <=          (int)padx/2 ) ) outpadded[indpad] = in[size.x * k * size.y + size.x * jj +            0].x;
    if ( ( i >= size.x + (int)padx/2 ) ) outpadded[indpad] = in[size.x * k * size.y + size.x * jj + (size.x - 1)].x;

    if ( (ii < 0) || (ii >= size.x) ) return;

    if ( ( j <=          (int)pady/2 ) ) outpadded[indpad] = in[size.x * k * size.y + size.x *            0 + ii].x;
    if ( ( j >= size.y + (int)pady/2 ) ) outpadded[indpad] = in[size.x * k * size.y + size.x * (size.y - 1) + ii].x;

    outpadded[indpad] = in[index].x;
}

__global__ void opt::paddR2R(float *in, float *outpadded,
        dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - (int)padx / 2 );
    int jj     = (int)( j - (int)pady / 2 );

    long long int index  = IND(ii,jj,k,size.x,size.y);

    long long int indpad =  Npadx * k *  Npady +  Npadx * j +  i;

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    if ( ( i <=          (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad] = in[size.x * k * size.y + size.x *            0 +            0];
    if ( ( i >= size.x + (int)padx/2 ) && ( j <=          (int)pady/2 )) outpadded[indpad] = in[size.x * k * size.y + size.x *            0 + (size.x - 1)];
    if ( ( i <=          (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad] = in[size.x * k * size.y + size.x * (size.y - 1) +            0];
    if ( ( i >= size.x + (int)padx/2 ) && ( j >= size.y + (int)pady/2 )) outpadded[indpad] = in[size.x * k * size.y + size.x * (size.y - 1) + (size.x - 1)];

    if ( (jj < 0) || (jj >= size.y) ) return;

    if ( ( i <=          (int)padx/2 ) ) outpadded[indpad] = in[size.x * k * size.y + size.x * jj +            0];
    if ( ( i >= size.x + (int)padx/2 ) ) outpadded[indpad] = in[size.x * k * size.y + size.x * jj + (size.x - 1)];

    if ( (ii < 0) || (ii >= size.x) ) return;

    if ( ( j <=          (int)pady/2 ) ) outpadded[indpad] = in[size.x * k * size.y + size.x *            0 + ii];
    if ( ( j >= size.y + (int)pady/2 ) ) outpadded[indpad] = in[size.x * k * size.y + size.x * (size.y - 1) + ii];

    outpadded[indpad] = in[index];
}

__global__ void opt::remove_paddC2R(cufftComplex *inpadded, float *out, 
dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - (int)pad.x * size.x / 2 );
    int jj     = (int)( j - (int)pad.y * size.y / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;

    long long int indpad =  Npadx * k *  Npady +  Npadx * j +  i;
    
    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index] = inpadded[indpad].x;
}

__global__ void opt::remove_paddC2C(cufftComplex *inpadded, cufftComplex *out, 
dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - (int)pad.x * size.x / 2 );
    int jj     = (int)( j - (int)pad.y * size.y / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad =  Npadx * k *  Npady +  Npadx *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index].x = inpadded[indpad].x;
    out[index].y = inpadded[indpad].y; 
}

__global__ void opt::remove_paddR2C(float *inpadded, cufftComplex *out, 
dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - (int)pad.x * size.x / 2 );
    int jj     = (int)( j - (int)pad.y * size.y / 2 );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad =  Npadx * k *  Npady +  Npadx *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index].x = inpadded[indpad];         
}

__global__ void opt::remove_paddR2R(float *inpadded, float *out, 
dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int padx  = pad.x * size.x;
    int pady  = pad.y * size.y;

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - (int)padx / 2 );
    int jj     = (int)( j - (int)pady / 2 );

    long long int index  = IND(ii,jj,k,size.x,size.y);

    long long int indpad =  Npadx * k *  Npady +  Npadx * j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index] = inpadded[indpad];   
}

__global__ void contrast_enhance::copy(float *projection, float *kernel, dim3 size)
{
    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    size_t index  = size.x * k * size.y + size.x * j + i;
    size_t ind = size.x * j + i;

    if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;

    projection[index] = kernel[ind];
}

__global__ void contrast_enhance::padding(float *in, cufftComplex *inpadded, dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int padding_x = pad.x * size.x / 2;
    int padding_y = pad.y * size.y / 2;

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;

    int ii     = (int)( i - padding_x );
    int jj     = (int)( j - padding_y );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad = Npadx  * k * Npady  + Npadx  *  j +  i;

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    inpadded[indpad].x = 1.0;
    inpadded[indpad].y = 0.0;

    if ( ( i <=          (int)padding_x ) && ( j <=          (int)padding_y )) inpadded[indpad].x = in[size.x * k * size.y + size.x *            0 +            0];
    if ( ( i >= size.x + (int)padding_x ) && ( j <=          (int)padding_y )) inpadded[indpad].x = in[size.x * k * size.y + size.x *            0 + (size.x - 1)];
    if ( ( i <=          (int)padding_x ) && ( j >= size.y + (int)padding_y )) inpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) +            0];
    if ( ( i >= size.x + (int)padding_x ) && ( j >= size.y + (int)padding_y )) inpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) + (size.x - 1)];

    if ( (jj < 0) || (jj >= size.y) ) return;

    if ( ( i <=          (int)padding_x ) ) inpadded[indpad].x = in[size.x * k * size.y + size.x * jj +            0];
    if ( ( i >= size.x + (int)padding_x ) ) inpadded[indpad].x = in[size.x * k * size.y + size.x * jj + (size.x - 1)];

    if ( (ii < 0) || (ii >= size.x) ) return;

    if ( ( j <=          (int)padding_y ) ) inpadded[indpad].x = in[size.x * k * size.y + size.x *            0 + ii];
    if ( ( j >= size.y + (int)padding_y ) ) inpadded[indpad].x = in[size.x * k * size.y + size.x * (size.y - 1) + ii];

    inpadded[indpad].x = in[index];
}

__global__ void contrast_enhance::recuperate_padding(cufftComplex *inpadded, float *in, dim3 size, dim3 pad)
{
    int Npadx = size.x * ( 1 + pad.x );
    int Npady = size.y * ( 1 + pad.y );

    int padding_x = pad.x * size.x / 2;
    int padding_y = pad.y * size.y / 2;

    int i      = blockIdx.x*blockDim.x + threadIdx.x;
    int j      = blockIdx.y*blockDim.y + threadIdx.y;
    int k      = blockIdx.z*blockDim.z + threadIdx.z;
    
    int ii     = (int)( i - padding_x );
    int jj     = (int)( j - padding_y );

    long long int index  = size.x * k * size.y + size.x * jj + ii;
    long long int indpad = Npadx  * k * Npady  + Npadx  *  j +  i;

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    in[index] = inpadded[indpad].x;
    
}

