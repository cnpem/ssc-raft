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

    ans[index].x = a[index].x * b[ind];	
    ans[index].y = a[index].y * b[ind];
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

    // float wx = ( - 1.0f + float(i) ) * 0.5f * float(size.x);
    // float wy = ( - 1.0f + float(j) ) * 0.5f * float(size.y);

    // wx       = wx / ( (float)size.x * pixel_objx );
    // wy       = wy / ( (float)size.y * pixel_objy );

    wx       = wx / pixel_objx;
    wy       = wy / pixel_objy;

    kernel[ind]  = 1.0f / ( beta_delta + gamma * (wx*wx + wy*wy) );
}

__global__ void contrast_enhance::contrast_paganin_based_Kernel(float *kernel, float regularization, 
float pixel_objx, float pixel_objy, dim3 size)
{
    int i        = blockIdx.x*blockDim.x + threadIdx.x;
    int j        = blockIdx.y*blockDim.y + threadIdx.y;
    int k        = blockIdx.z*blockDim.z + threadIdx.z;
    size_t ind   = size.x * j + i;

    if ( (i >= size.x) || (j >= size.y) || (k >= 1) ) return;

    /* Reciprocal grid */
    
    float hx = 2.0f / size.x;
    float hy = 2.0f / size.y;

    float wx = fminf( i, size.x - i ) / (float)size.x;  
    float wy = fminf( j, size.y - j ) / (float)size.y;

    wx       = wx / hx;
    wy       = wy / hy;

    kernel[ind]  = 1.0f / ( 1.0f + regularization * (wx*wx + wy*wy) );
}

__global__ void paganinKernel_tomopy(CFG configs, cufftComplex *data, dim3 size)
{
    /* Version of Paganin by frames implemented on Tomopy
    DOI:10.1107/S1600577514013939 */
}

__global__ void paganinKernel_v0(CFG configs, cufftComplex *data, dim3 size)
{
    /* Version of Paganin by frames on Miqueles and Guerrero (2020)
    https://doi.org/10.1016/j.rinam.2019.100088 and published by 
    Yu et al (2002) https://doi.org/10.1364/OE.26.011110 */
}

void contrast_enhance::apply_contrast_filter(CFG configs, GPU gpus, float *projections, float *kernel,
dim3 size, dim3 size_pad, dim3 pad)
{
    size_t npad = opt::get_total_points(size_pad);
    float scale = (float)( size_pad.x * size_pad.y);

    cufftComplex *dataPadded = opt::allocGPU<cufftComplex>(npad);

    dim3 threadsPerBlock(TPBX,TPBY,1);
    dim3 gridBlock( (int)ceil( size_pad.x / threadsPerBlock.x ) + 1, 
                    (int)ceil( size_pad.y / threadsPerBlock.y ) + 1, size_pad.z);
    
    contrast_enhance::padding<<<gridBlock,threadsPerBlock>>>(projections, dataPadded, size, pad);

    HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_FORWARD));

    contrast_enhance::multiplication<<<gridBlock,threadsPerBlock>>>(dataPadded, kernel, dataPadded, size_pad);

    HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_INVERSE));

    opt::scale<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad, scale);

    opt::fftshift2D<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad);

    contrast_enhance::recuperate_padding<<<gridBlock,threadsPerBlock>>>(dataPadded, projections, size, pad);

    HANDLE_ERROR(cudaFree(dataPadded));
}



