#include "../../../../inc/filters.h"
#include "../../../../inc/processing.h"

extern "C" {

	void _paganin_gpu(GPU gpus, float *projections, float *kernel, dim3 size, dim3 size_pad)
    {
        convolution_mplan2DR2R(gpus, projections, kernel, 1.0f, size, size_pad);
        getLog(projections, size, gpus);
    }

 	__global__ void paganinKernel(GEO geometry, float *kernel, dim3 size, int phase_reg)
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

        float gamma = ( phase_reg == 0.0 ? 0.0 : phase_reg );

		float kernelX = 4.0 * float(M_PI) * float(M_PI) * geometry.z2x * gamma * wxx  / ( geometry.magnitude_x );
		float kernelY = 4.0 * float(M_PI) * float(M_PI) * geometry.z2y * gamma * wyy  / ( geometry.magnitude_y );

        kernel[ind] = 1.0 / ( 1.0 + kernelX + kernelY );

    }

}
