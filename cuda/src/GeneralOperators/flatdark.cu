#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{

    __global__ void RemoveMeanKernel(float *in, float *mean, int sizex, int sizey, int sizez)
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j = blockIdx.y*blockDim.y + threadIdx.y;
        int k = blockIdx.z*blockDim.z + threadIdx.z;
        int index = sizex * ( k * sizey + j) + i;

        if ( (i >= sizex) || (j >= sizey) || ( k >= sizez) ) return;
			in[index] = in[index] - mean[j];
    }

    __global__ void ComputeMeanKernel(float *in, float *mean, int sizex, int sizey, int sizez)
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j = blockIdx.y*blockDim.y + threadIdx.y;
        int k = blockIdx.z*blockDim.z + threadIdx.z;
        int index = sizex * ( k * sizey + j) + i;

		size_t N = sizex * sizez;

        if ( (i >= sizex) || (j >= sizey) || ( k >= sizez) ) return;
		extern __shared__ float temp[];
		
		if (sizex < blockDim.x || sizey < blockDim.y || sizez < blockDim.z) {
            for (int ii = 0; ii < blockDim.x*blockDim.y*blockDim.z; ii++)
                temp[ii] = 0.0;
        }
        temp[blockDim.x*(threadIdx.z*blockDim.y + threadIdx.y) + threadIdx.x] = in[index] / N;
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) { // for each slice in z, only one thread/block runs
            float sum = 0.0;
            for (int ii = 0; ii < blockDim.x*blockDim.y; ii++)
                sum += temp[ii + threadIdx.z*blockDim.x*blockDim.y];
        	atomicAdd(mean + j,sum);
		}
    }

	void remove_mean_flat_dark(float *flat, float *dark, int sizex, int sizez, int numflats)
	{
		Image2D<float> dflat(sizex, numflats, sizez); // Flat in GPU
		Image2D<float> ddark(sizex, 1, sizez); // Dark in GPU

		dflat.CopyFrom(flat, 0, (size_t)sizex*numflats*sizez);
		ddark.CopyFrom(dark, 0, (size_t)sizex*sizez         );

		float *mean_flat, *mean_dark;

		dim3 blocks = dim3(sizex,numflats,sizez);
		blocks.x = ( sizex + 127 ) / 128; 

		HANDLE_ERROR(cudaMalloc((void **)&mean_flat, sizeof(float) * numflats )); 
		HANDLE_ERROR(cudaMalloc((void **)&mean_dark, sizeof(float)             )); 

		ComputeMeanKernel<<<blocks,128>>>(dflat.gpuptr, mean_flat, sizex, numflats, sizez);
		ComputeMeanKernel<<<blocks,128>>>(ddark.gpuptr, mean_dark, sizex,        1, sizez);

		RemoveMeanKernel<<<blocks,128>>>(dflat.gpuptr, mean_flat, sizex, numflats, sizez);
		RemoveMeanKernel<<<blocks,128>>>(ddark.gpuptr, mean_dark, sizex,        1, sizez);

		dflat.CopyTo(flat, 0, (size_t)sizex*numflats*sizez);
		ddark.CopyTo(dark, 0, (size_t)sizex*sizez         );

	}

   	static __global__ void KFlatDark(float* in, float* dark, float* flat, dim3 size, int numflats, int totalslices, int Initframe)
	{  
      // Supports 2 flats only
		size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t line;
		float ft, flat_before, flat_after, dk, T, Q, S, interp, tol = 1e-14;

		if(idx < size.x && blockIdx.y < size.y && blockIdx.z < size.z){
			
			dk          = dark[size.x * blockIdx.z            + idx];
			flat_before = flat[size.x * numflats * blockIdx.z + idx];
			line        = size.x * size.y * blockIdx.z + size.x * blockIdx.y + idx;

         	if(numflats > 1){
				interp     = float( blockIdx.y + Initframe ) / float( totalslices ); 
				flat_after = flat[size.x * numflats * blockIdx.z + size.x + idx];
				ft         = flat_before * ( 1.0f - interp ) + interp * flat_after;
			}else{
				ft = flat_before;
			}

			T = in[line] - dk;
			Q = ft       - dk;

			if ( T < tol )
				T = 1.0;
			
			if ( Q < tol )
				Q = 1.0;

			S = T / Q;

			if ( S < tol )
				S = 1.0;

			// in[line] = fmaxf(T, 0.5f) / fmaxf(Q,0.5f) ; // Old version (Giovanni)
			in[line] = S;

		}
	}

	static __global__ void Klog(float* in, dim3 size)
	{  
		size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
		size_t line;

		if(idx < size.x && blockIdx.y < size.y && blockIdx.z < size.z){
			
			line     = size.x * size.y * blockIdx.z + size.x * blockIdx.y + idx;

			in[line] = - logf( in[line] );

		}
	}

	void getFlatDarkCorrection(float* frames, float* flat, float* dark, int sizex, int sizey, int sizez, int numflats, int islog, dim3 BT, dim3 Grd)
	{
		/* Do here the mean of flat and dark acquisitions */

		/* Do the dark subtraction and division by flat (with or without log) */
		KFlatDark<<<Grd,BT>>>(frames, dark, flat, dim3(sizex,sizey,sizez), numflats, sizey, 0);
		
		if ( islog == 1 )
			Klog<<<Grd,BT>>>(frames, dim3(sizex,sizey,sizez));

	}

	void flatdark_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int is_log, int totalslices)
	{	
		// Supports 2 flats max
		cudaSetDevice(gpu);

		// Remove mean of flat and dark:
		// remove_mean_flat_dark(flat, dark, nrays, totalslices, numflats);

		int b;
		int blocksize = min(nslices,32); // Herança do Giovanni -> Mudar

		size_t nblock = (size_t)ceil( (float) nslices / blocksize );
		int ptr = 0, subblock;

		dim3 Grd = dim3(nrays,nangles,blocksize);
		Grd.x = ( nrays + 127 ) / 128;

		dim3 BT = dim3(128,1,1);

		// GPUs Pointers: declaration and allocation
		rImage data(nrays,nangles,blocksize); // Frames in GPU 

		Image2D<float> cflat(nrays, numflats, blocksize); // Flat in GPU
		Image2D<float> cdark(nrays,        1, blocksize); // Dark in GPU

		for(b = 0; b < nblock; b++){
			
			subblock   = min(nslices - ptr, blocksize);

			data.CopyFrom (frames + (size_t)ptr*nrays*nangles , 0, (size_t)subblock * nrays * nangles );
			cflat.CopyFrom(flat   + (size_t)ptr*nrays*numflats, 0, (size_t)subblock * nrays * numflats);
			cdark.CopyFrom(dark   + (size_t)ptr*nrays         , 0, (size_t)subblock * nrays           );

			getFlatDarkCorrection(data.gpuptr, cflat.gpuptr, cdark.gpuptr, nrays, nangles, subblock, numflats, is_log, BT, Grd);

			data.CopyTo(frames + (size_t)ptr*nrays*nangles, 0, (size_t)subblock * nrays * nangles);
      
			/* Update pointer */
			ptr = ptr + subblock;
		}

		cudaDeviceSynchronize();
	}

	void flatdark_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int is_log)
	{
		int t;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		int ptr = 0, subblock;
		
		std::vector<std::future<void>> threads;
		
		for(t = 0; t < ngpus; t++){ 
			
			subblock   = min(nslices - ptr, blockgpu);

			threads.push_back(std::async( std::launch::async, 
						flatdark_gpu, 
						gpus[t], 
						frames + (size_t)ptr*nrays*nangles, 
						flat   + (size_t)ptr*nrays*numflats, 
						dark   + (size_t)ptr*nrays, 
						nrays, subblock, nangles, 
						numflats, is_log, nslices
						));

			/* Update pointer */
			ptr = ptr + subblock;
		}

		for(auto& t : threads)
			t.get();
	}
}

