#include "processing/filters.hpp"
#include "processing/processing.hpp"
#include "common/opt.hpp"


__global__ void phase_pag::mult(cufftComplex *a, float *b, 
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

__global__ void phase_pag::paganinKernel(float *kernel, float beta_delta, float wavelength, 
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
    float nyqx   = 0.5;
    float nyqy   = 0.5;
    
    float wx     = float( int( i + size.x / 2.0 ) % int( size.x ) ) / float( size.x ) - nyqx;
    float wy     = float( int( j + size.y / 2.0 ) % int( size.y ) ) / float( size.y ) - nyqy;

    wx           = wx / pixel_objx;
    wy           = wy / pixel_objy;

    kernel[ind]  = 1.0f / ( beta_delta + gamma * (wx*wx + wy*wy) );
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

void phase_pag::apply_paganin_filter(CFG configs, GPU gpus, float *projections, float *kernel,
dim3 size, dim3 size_pad, dim3 pad)
{
    size_t npad = opt::get_total_points(size_pad);
    float scale = (float)( size.x * size.y);

    cufftComplex *dataPadded = opt::allocGPU<cufftComplex>(npad);

    dim3 threadsPerBlock(TPBX,TPBY,1);
    dim3 gridBlock( (int)ceil( size_pad.x / threadsPerBlock.x ) + 1, 
                    (int)ceil( size_pad.y / threadsPerBlock.y ) + 1, size_pad.z);
    
    phase_pag::padding<<<gridBlock,threadsPerBlock>>>(projections, dataPadded, size, pad);

    HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_FORWARD));

    phase_pag::mult<<<gridBlock,threadsPerBlock>>>(dataPadded, kernel, dataPadded, size_pad);

    HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_INVERSE));

    // opt::scale<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad, scale);

    // opt::fftshift2D<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad);

    phase_pag::recuperate_padding<<<gridBlock,threadsPerBlock>>>(dataPadded, projections, size, pad);

    opt::scale<<<gridBlock,threadsPerBlock>>>(projections, size, scale);

    HANDLE_ERROR(cudaFree(dataPadded));

    // getLog(projections, size);
}



