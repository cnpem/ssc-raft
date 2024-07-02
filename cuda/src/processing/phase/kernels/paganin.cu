#include "processing/filters.hpp"
#include "processing/processing.hpp"
#include "common/opt.hpp"

extern "C" {

    __global__ void paganinKernel(CFG configs, cuComplex *data, dim3 size)
    {
        /* Version of Paganin by frames on Miqueles and Guerrero (2020)
        https://doi.org/10.1016/j.rinam.2019.100088 and published by 
        Yu et al (2002) https://doi.org/10.1364/OE.26.011110 */
        
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = size.x * j + i;
        size_t index = size.y * k * size.x + ind;

        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;

        /* Reciprocal grid */ 
        float hx   = 2.0f / size.x;
        float hy   = 2.0f / size.y;
        float nyqx = 0.5 * hx;
        float nyqy = 0.5 * hy;

        float wx = (float)i + hx - nyqx;
        float wy = (float)j + hy - nyqy;
        float gamma = configs.geometry.wavelenght * configs.geometry.z2x * float(M_PI) * ( configs.beta_delta == 0.0 ? 0.0:( 1.0f / configs.beta_delta ) );

        float kernel = 1.0f / ( 1.0f + gamma * (wx*wx + wy*wy) );
        
        data[index].x = data[index].x * kernel;
        data[index].y = data[index].y * kernel;
    }

    __global__ void paganinKernel_tomopy(CFG configs, cufftComplex *data, dim3 size)
    {
        /* Version of Paganin by frames implemented on Tomopy
        DOI:10.1107/S1600577514013939 */
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;

        size_t ind   = size.x * j + i;
        size_t index = size.y * k * size.x + ind;

        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;

        /* Reciprocal grid */ 
        
        float hx   = configs.geometry.obj_pixel_x;
        float hy   = configs.geometry.obj_pixel_y;
        float nyqx = 0.5 * hx;
        float nyqy = 0.5 * hy;

        float wx = (float)i + hx - nyqx;
        float wy = (float)j + hy - nyqy;

        float gamma = configs.geometry.wavelenght * configs.geometry.z2x / ( 4.0f * float(M_PI) );

        float kernel = 1.0f / ( configs.beta_delta + gamma * (wx*wx + wy*wy) );

        data[index].x = data[index].x * kernel;
        data[index].y = data[index].y * kernel;
    }

    __global__ void paganinKernel_v0(CFG configs, cufftComplex *data, dim3 size)
    {
        /* Version of Paganin by frames published by Paganin et al (2002)
        DOI:10.1046/j.1365-2818.2002.01010.x */
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = size.x * j + i;
        size_t index = size.y * k * size.x + ind;

        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;

        /* Reciprocal grid */ 
        float hx   = configs.geometry.obj_pixel_x;
        float hy   = configs.geometry.obj_pixel_y;
        float nyqx = 0.5 * hx;
        float nyqy = 0.5 * hy;

        float wx = (float)i + hx - nyqx;
        float wy = (float)j + hy - nyqy;

        float gamma = configs.geometry.wavelenght * configs.geometry.z2x * ( configs.beta_delta == 0.0 ? 0.0:( 1.0f / configs.beta_delta ) ) / ( 4.0f * float(M_PI) );

        float kernel = 1.0f / ( 1.0f + gamma * (wx*wx + wy*wy) );
        
        data[index].x = data[index].x * kernel;
        data[index].y = data[index].y * kernel;
    }

    void apply_paganin_filter(CFG configs, GPU gpus, float *data, 
    dim3 size, dim3 size_pad, dim3 pad)
    {
        size_t npad = opt::get_total_points(size_pad);
        float scale = ( 1.0f / (float)( size_pad.x * size_pad.y) );

        cufftComplex *dataPadded = opt::allocGPU<cufftComplex>(npad);

        dim3 threadsPerBlock(TPBX,TPBY,1);
        dim3 gridBlock( (int)ceil( size_pad.x / threadsPerBlock.x ) + 1, 
                        (int)ceil( size_pad.y / threadsPerBlock.y ) + 1, 
                        size_pad.z);
        
        opt::paddR2C<<<gridBlock,threadsPerBlock>>>(data, dataPadded, size, pad, 1.0f);

        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_FORWARD));

        paganinKernel<<<gridBlock,threadsPerBlock>>>(configs, dataPadded, size_pad);

        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_INVERSE));

        opt::scale<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad, scale);

        // opt::fftshift2D<<<gpus.Grd,gpus.BT>>>(dataPadded, size_pad);

        opt::remove_paddC2R<<<gridBlock,threadsPerBlock>>>(dataPadded, data, size, pad);

        HANDLE_ERROR(cudaFree(dataPadded));
    }

    void apply_paganin_filter_tomopy(CFG configs, GPU gpus, float *data, 
    dim3 size, dim3 size_pad, dim3 pad)
    {
        size_t npad = opt::get_total_points(size_pad);
        printf("Size dataPadded = %ld \n",npad);
        float scale = (float)( 1.0f / ( size_pad.x * size_pad.y) );

        cufftComplex *dataPadded = opt::allocGPU<cufftComplex>(npad); 
        // cufftComplex *dataPadded; 
        // HANDLE_ERROR(cudaMalloc((void **)&dataPadded, npad * sizeof(cufftComplex)))  

        dim3 threadsPerBlock(TPBX,TPBY,1);
        dim3 gridBlock( (int)ceil( size_pad.x / threadsPerBlock.x ) + 1, 
                        (int)ceil( size_pad.y / threadsPerBlock.y ) + 1, 
                        size_pad.z);
        
		printf("GD: %d, %d, %d \n", gridBlock.x, gridBlock.y, gridBlock.z);
        
        opt::paddR2C<<<gridBlock,threadsPerBlock>>>(data, dataPadded, size, pad, 1.0f);

        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_FORWARD));

        paganinKernel_tomopy<<<gridBlock,threadsPerBlock>>>(configs, dataPadded, size_pad);

        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_INVERSE));

        opt::scale<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad, scale);

        // opt::fftshift2D<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad);

        opt::remove_paddC2R<<<gridBlock,threadsPerBlock>>>(dataPadded, data, size, pad);

        HANDLE_ERROR(cudaFree(dataPadded));
    }

    void apply_paganin_filter_v0(CFG configs, GPU gpus, float *data, 
    dim3 size, dim3 size_pad, dim3 pad)
    {
        size_t npad = opt::get_total_points(size_pad);
        float scale = (float)( 1.0f / ( size_pad.x * size_pad.y) );

        cufftComplex *dataPadded = opt::allocGPU<cufftComplex>(npad);

        dim3 threadsPerBlock(TPBX,TPBY,1);
        dim3 gridBlock( (int)ceil( size_pad.x / threadsPerBlock.x ) + 1, 
                        (int)ceil( size_pad.y / threadsPerBlock.y ) + 1, 
                        size_pad.z);

        opt::paddR2C<<<gridBlock,threadsPerBlock>>>(data, dataPadded, size, pad, 1.0f);

        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_FORWARD));

        paganinKernel_v0<<<gridBlock,threadsPerBlock>>>(configs, dataPadded, size_pad);

        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_INVERSE));

        // opt::scale<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad, scale);

        // opt::fftshift2D<<<gridBlock,threadsPerBlock>>>(dataPadded, size_pad);

        opt::remove_paddC2R<<<gridBlock,threadsPerBlock>>>(dataPadded, data, size, pad);

        HANDLE_ERROR(cudaFree(dataPadded));
    }

    void _paganin_gpu(CFG configs, GPU gpus, float *projections,
    dim3 size, dim3 size_pad, dim3 pad)
    {
        apply_paganin_filter(configs, gpus, projections, size, size_pad, pad);
        // getLog(projections, size);
    }

    void _paganin_gpu_tomopy(CFG configs, GPU gpus, float *projections,
    dim3 size, dim3 size_pad, dim3 pad)
    {
        apply_paganin_filter_tomopy(configs, gpus, projections, size, size_pad, pad);
        // getLog(projections, size);
    }

    void _paganin_gpu_v0(CFG configs, GPU gpus, float *projections,
    dim3 size, dim3 size_pad, dim3 pad)
    {
        apply_paganin_filter_v0(configs, gpus, projections, size, size_pad, pad);
        // getLog(projections, size);
    }
}
