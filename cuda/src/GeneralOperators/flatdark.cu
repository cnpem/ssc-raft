#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{

    __global__ void remove_meanKernel(float *in, float *mean, int sizex, int sizey, int sizez)
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j = blockIdx.y*blockDim.y + threadIdx.y;
        int k = blockIdx.z*blockDim.z + threadIdx.z;
        int index = sizex * ( k * sizey + j) + i;

        if ( (i >= sizex) || (j >= sizey) || ( k >= sizez) ) return;
			in[index] = in[index] - mean[j];
    }

    __global__ void meanKernel(float *in, float *mean, int sizex, int sizey, int sizez)
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

	void remove_mean_flat_dark(float *flat, float *dark, int nrays, int nslices, int numflats)
	{
		Image2D<float> dflat(nrays, numflats, nslices); // Flat in GPU
		Image2D<float> ddark(nrays, 1, nslices); // Dark in GPU

		dflat.CopyFrom(flat, 0, (size_t)nrays*numflats*nslices);
		ddark.CopyFrom(dark, 0, (size_t)nrays*nslices         );

		float *mean_flat, *mean_dark;

		dim3 blocks = dim3(nrays,numflats,nslices);
		blocks.x = ( nrays + 127 ) / 128; 

		HANDLE_ERROR(cudaMalloc((void **)&mean_flat, sizeof(float) * numflats )); 
		HANDLE_ERROR(cudaMalloc((void **)&mean_dark, sizeof(float)             )); 

		meanKernel<<<blocks,128>>>(dflat.gpuptr, mean_flat, nrays, numflats, nslices);
		meanKernel<<<blocks,128>>>(ddark.gpuptr, mean_dark, nrays,        1, nslices);

		remove_meanKernel<<<blocks,128>>>(dflat.gpuptr, mean_flat, nrays, numflats, nslices);
		remove_meanKernel<<<blocks,128>>>(ddark.gpuptr, mean_dark, nrays,        1, nslices);

		dflat.CopyTo(flat, 0, (size_t)nrays*numflats*nslices);
		ddark.CopyTo(dark, 0, (size_t)nrays*nslices         );

	}

   	static __global__ void KFlatDark(float* in, float* dark, float* flat, dim3 size, int numflats, int Totalframes, int Initframe)
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
				interp     = float( blockIdx.y + Initframe ) / float( Totalframes ); 
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

   static __global__ void KFlatDarklog(float* in, float* dark, float* flat, dim3 size, int numflats, int Totalframes, int Initframe)
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
				interp     = float( blockIdx.y + Initframe ) / float( Totalframes ); 
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

			// in[line] = - logf( fmaxf(T, 0.5f) / fmaxf(Q,0.5f) ); // Old version (Giovanni)
			in[line] = - logf( S );

		}
	}

	void flatdark_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int Totalframes, int Initframe, int is_log, int totalslices)
	{	
		// Supports 2 flats max
		cudaSetDevice(gpu);

		// Remove mean of flat and dark:
		// remove_mean_flat_dark(flat, dark, nrays, totalslices, numflats);

		int b;
		int blocksize = min(nslices,32); // Herança do Giovanni -> Mudar
		size_t offset = (size_t)gpu * nslices * nrays;

		dim3 blocks = dim3(nrays,nangles,blocksize);
		blocks.x = ( nrays + 127 ) / 128;

		// GPUs Pointers: declaration and allocation
		rImage data(nrays,nangles,blocksize); // Frames in GPU 

		Image2D<float> cflat(nrays, numflats, blocksize); // Flat in GPU
		Image2D<float> cdark(nrays,        1, blocksize); // Dark in GPU

		for(b = 0; b < nslices; b += blocksize){
			
			blocksize = min(blocksize,nslices-b);

			// data.CopyFrom (frames + (size_t)b*nrays*nangles                   , 0, (size_t)blocksize * nrays * nangles                     );
			// cflat.CopyFrom(flat   + (size_t)b*nrays*numflats + offset*numflats, 0, (size_t)blocksize * nrays * numflats + offset * numflats);
			// cdark.CopyFrom(dark   + (size_t)b*nrays          + offset         , 0, (size_t)blocksize * nrays            + offset           );
			data.CopyFrom (frames + (size_t)b*nrays*nangles , 0, (size_t)blocksize * nrays * nangles );
			cflat.CopyFrom(flat   + (size_t)b*nrays*numflats, 0, (size_t)blocksize * nrays * numflats);
			cdark.CopyFrom(dark   + (size_t)b*nrays         , 0, (size_t)blocksize * nrays           );

			if ( is_log == 1 ){
				KFlatDarklog<<<blocks,128>>>(data.gpuptr, cdark.gpuptr, cflat.gpuptr, dim3(nrays,nangles,blocksize), numflats, Totalframes, Initframe);
			}else{
				KFlatDark<<<blocks,128>>>(data.gpuptr, cdark.gpuptr, cflat.gpuptr, dim3(nrays,nangles,blocksize), numflats, Totalframes, Initframe);
			}

			data.CopyTo(frames + (size_t)b*nrays*nangles, 0, (size_t)blocksize * nrays * nangles);
      
		}

		cudaDeviceSynchronize();
	}

	void flatdark_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int Totalframes, int Initframe, int is_log)
	{
		int t;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		
		std::vector<std::future<void>> threads;
		
		for(t = 0; t < ngpus; t++){ 
			
			blockgpu = min(nslices - blockgpu * t, blockgpu);

			threads.push_back(std::async( std::launch::async, 
						flatdark_gpu, 
						gpus[t], 
						frames + (size_t)t*blockgpu*nrays*nangles, 
						flat   + (size_t)t*blockgpu*nrays*numflats, 
						dark   + (size_t)t*blockgpu*nrays, 
						nrays, blockgpu, nangles, 
						numflats, Totalframes, Initframe,
						is_log, nslices
						));
			// threads.push_back(std::async( std::launch::async, 
			// 			flatdark_gpu, 
			// 			gpus[t], 
			// 			frames + (size_t)t*blockgpu*nrays*nangles, 
			// 			flat, 
			// 			dark, 
			// 			nrays, blockgpu, nangles, 
			// 			numflats, Totalframes, Initframe,
			// 			is_log, nslices
			// 			));
		}

		for(auto& t : threads)
			t.get();
	}
}

