#include "../../../../inc/include.h"
#include "../../../../inc/common/types.hpp"
#include "../../../../inc/common/kernel_operators.hpp"
#include "../../../../inc/common/complex.hpp"
#include "../../../../inc/common/operations.hpp"
#include "../../../../inc/common/logerror.hpp"

extern "C" {

	void _bronnikov_gpu(PAR param, float *projections, size_t nrays, size_t nangles, size_t nslices)
    {
		size_t n    = nrays * nslices * nangles;
		size_t npad = param.Npadx * param.Npady * nangles;
		float *d_sino, *kernel;
		cufftComplex *d_sinoPadded;

		HANDLE_ERROR(cudaMalloc((void **)&kernel      , sizeof(float) * npad ));
		HANDLE_ERROR(cudaMalloc((void **)&d_sino      , sizeof(float) * n )); 
		HANDLE_ERROR(cudaMalloc((void **)&d_sinoPadded, sizeof(cufftComplex) * npad ));

		HANDLE_ERROR(cudaMemcpy(d_sino, projections, n * sizeof(float), cudaMemcpyHostToDevice));

        bronnikovPrep<<<param.Grd,param.BT>>>(d_sino, param, nrays, nslices, nangles);

        bronnikovKernel<<<param.Grd,param.BT>>>(kernel, param, param.Npadx, param.Npady, nangles);

        apply_filter(param, d_sino, kernel, d_sino, d_sinoPadded, nrays, nslices, nangles);

		HANDLE_ERROR(cudaMemcpy(projections, d_sino, n * sizeof(float), cudaMemcpyDeviceToHost));

		cudaFree(kernel);
		cudaFree(d_sinoPadded);
		cudaFree(d_sino);
    }

 	__global__ void bronnikovKernel(float *kernel, PAR param, size_t sizex, size_t sizey, size_t sizez)
    {
        size_t i     = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j     = blockIdx.y*blockDim.y + threadIdx.y;
        size_t ind   = sizex * j + i;

        if ( (i >= sizex ) || (j >= sizey) ) return;

        // Reciprocal grid
        float nyq       = 0.5;
        float wx        = float( int( i + 0.5 + sizex / 2.0 ) % int( sizex ) ) / float( sizex ) - nyq;
        float wy        = float( int( j + 0.5 + sizey / 2.0 ) % int( sizey ) ) / float( sizey ) - nyq;

		float magnx = 1.0, magny = 1.0; // Parallel case

        if ( param.z1x != 0.0) // Fanbeam Case!
            magnx = ( param.z1x + param.z2x ) / param.z1x;
        
        if ( param.z1y != 0.0) // Conebeam Case!
            magny = ( param.z1y + param.z2y ) / param.z1y;

        float wxx = wx*wx; float wyy = wy*wy;

		float kernelX = 2.0 * float(M_PI) * param.z2x * param.lambda * wxx  / ( magnx );
		float kernelY = 2.0 * float(M_PI) * param.z2y * param.lambda * wyy  / ( magny );

        kernel[ind] = 1.0 / ( param.alpha +  kernelX + kernelY );

    }

    __global__ void bronnikovPrep(float *data, PAR param, size_t sizex, size_t sizey, size_t sizez)
    {
        size_t i     = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j     = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k     = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = sizex * k * sizey + sizex * j + i;

        if ( (i >= sizex ) || (j >= sizey) || (k >= sizez)) return;

        data[ind] = data[ind] - 1.0;

    }

}
