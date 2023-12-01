#include "../../../../inc/sscraft.h"

extern "C" {

	void _rytov_gpu(GPU gpus, float *projections, float *kernel, dim3 size, dim3 size_pad)
    {
        rytovPrep<<<gpus.Grd,gpus.BT>>>(projections, size);
        convolution_mplan2DR2R(gpus, projections, kernel, 1.0f, size, size_pad);
    }

 	__global__ void rytovKernel(GEO geometry, float *kernel, dim3 size, int phase_reg)
    {
        size_t i     = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j     = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k     = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = size.x * j + i;

        if ( (i >= size.x ) || (j >= size.y) || (k >= 1) ) return;

        /* Reciprocal grid */
        float nyq       = 0.5;
        float wx        = float( int( i + 0.5 + size.x / 2.0 ) % int( size.x ) ) / float( size.x ) - nyq;
        float wy        = float( int( j + 0.5 + size.y / 2.0 ) % int( size.y ) ) / float( size.y ) - nyq;

        float wxx = wx*wx; float wyy = wy*wy;

		float kernelX = float(M_PI) * geometry.z2x * geometry.lambda * wxx  / ( geometry.magnitude_x );
		float kernelY = float(M_PI) * geometry.z2y * geometry.lambda * wyy  / ( geometry.magnitude_y );

        kernel[ind] = 1.0 / ( phase_reg * cosf( kernelX + kernelY ) + sinf( kernelX + kernelY ) );

    }

    __global__ void rytovPrep(float *data, dim3 size)
    {
        size_t i     = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j     = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k     = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = size.x * k * size.y + size.x * j + i;

        float tol = 1e-10;

        if ( (i >= size.x ) || (j >= size.y) || (k >= size.z) ) return;

        if ( data[ind] < tol )
				data[ind] = 1.0;

        data[ind] = 0.5 * logf( data[ind] );
        // data[ind] = 0.5 * logf( fmaxf(data[ind], 0.5f) );

    }

}