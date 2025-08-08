#include "processing/filters.hpp"
#include "processing/processing.hpp"
#include "common/opt.hpp"

__global__ void contrast_enhance::multiplication(cufftComplex *a, float *b, 
cufftComplex *ans, dim3 size)
{
    int i  = threadIdx.x + blockIdx.x*blockDim.x;
    int j  = threadIdx.y + blockIdx.y*blockDim.y;
    int k  = threadIdx.z + blockIdx.z*blockDim.z;

    size_t ind   = size.x * j + i;
    size_t index = size.y * size.x * k + ind; 

    if( (i >= size.x) || (j >= size.y) || (k >= size.z)) return;  

    ans[index].x = a[index].x * b[ind] / ( size.x * size.y );	
    ans[index].y = a[index].y * b[ind] / ( size.x * size.y );
}

__global__ void contrast_enhance::paganinKernel(float *kernel, float beta_delta, float wavelength, 
float pixel_objx, float pixel_objy, float z2, dim3 size)
{
    /* Version of Paganin by frames published by Paganin et al (2002)
    DOI:10.1046/j.1365-2818.2002.01010.x */
    int i        = blockIdx.x*blockDim.x + threadIdx.x;
    int j        = blockIdx.y*blockDim.y + threadIdx.y;
    int k        = blockIdx.z*blockDim.z + threadIdx.z;
    size_t ind   = size.x * j + i;

    float gamma  = wavelength * z2 / ( 4.0f * float(M_PI) );

    if ( (i >= size.x) || (j >= size.y) || (k >= 1) ) return;

    /* Reciprocal grid */
    float wx = fminf( i, size.x - i ) / (float)size.x;  
    float wy = fminf( j, size.y - j ) / (float)size.y;

    wx       = wx / pixel_objx;
    wy       = wy / pixel_objy;

    kernel[ind]  = 1.0f / ( beta_delta + gamma * (wx*wx + wy*wy) );
}

__global__ void contrast_enhance::contrast_paganin_based_Kernel(float *kernel, float regularization, dim3 size)
{
    int i        = blockIdx.x*blockDim.x + threadIdx.x;
    int j        = blockIdx.y*blockDim.y + threadIdx.y;
    int k        = blockIdx.z*blockDim.z + threadIdx.z;
    size_t ind   = size.x * j + i;

    if ( (i >= size.x) || (j >= size.y) || (k >= 1) ) return;

    float hx = 2.0f / size.x;
    float hy = 2.0f / size.y;

    /* Reciprocal grid */
    float wx = fminf( i, size.x - i ) / (float)size.x;  
    float wy = fminf( j, size.y - j ) / (float)size.y;

    wx       = wx / hx;
    wy       = wy / hy;

    kernel[ind]  = 1.0f / ( 1.0f + regularization * (wx*wx + wy*wy) );
}



