#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{
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
		float ft, flat_before, flat_after, dk, T, Q, S, interp, tol = 1e-10;

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

	void flatdark_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int Totalframes, int Initframe, int is_log)
	{	
		// Supports 2 flats max
		cudaSetDevice(gpu);

		int b;
		int blocksize = min(nslices,32); // Herança do Giovanni -> Mudar

		dim3 blocks = dim3(nrays,nangles,blocksize);
		blocks.x = ( nrays + 127 ) / 128;

		// GPUs Pointers: declaration and allocation
		rImage data(nrays,nangles,blocksize); // Frames in GPU 

		Image2D<float> cflat(nrays, numflats, blocksize); // Flat in GPU
		Image2D<float> cdark(nrays, 1, blocksize); // Dark in GPU


		for(b = 0; b < nslices; b += blocksize){
			
			blocksize = min(blocksize,nslices-b);

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

		printf("Rings filter: Number of flats is %d \n", numflats);
		
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
						is_log
						));
		}

		for(auto& t : threads)
			t.get();
	}
}

