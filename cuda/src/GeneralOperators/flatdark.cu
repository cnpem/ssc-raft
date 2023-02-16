#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{
   static __global__ void KFlatDarklog(float* in, float* dark, float* flat, dim3 size, int numflats)
	{  
      // Supports 2 flats only
		size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
		float tol = 1e-15;

		if(idx < size.x && blockIdx.y < size.y)
		{
			float dk = dark[blockIdx.y * size.x + idx];
			float ft = flat[blockIdx.y * size.x + idx];
			
         	if(numflats > 1){
				float interp = float(blockIdx.z+1)/float(size.z+1);
				ft = ft*(1.0f-interp) + interp*(float)flat[size.x*size.y + blockIdx.y * size.x + idx];
			}

			size_t line = size.y*blockIdx.z + blockIdx.y;

			float T = in[line * size.x + idx] - dk;
			float Q = ft-dk;

			if ( T < tol)
				T = 1.0;
			
			if ( Q < tol)
				Q = 1.0;

			in[line * size.x + idx] = -logf( fmaxf(in[line * size.x + idx] - dk, 0.5f) / fmaxf(ft-dk,0.5f) );
			// in[line * size.x + idx] = -logf( T / Q );

		}
	}

	static __global__ void KFlatDark(float* in, float* dark, float* flat, dim3 size, int numflats)
	{  
      // Supports 2 flats only
		size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
		
		if(idx < size.x && blockIdx.y < size.y)
		{
			float dk = dark[blockIdx.y * size.x + idx];
			float ft = flat[blockIdx.y * size.x + idx];
			
         	if(numflats > 1){
				float interp = float(blockIdx.z+1)/float(size.z+1);
				ft = ft*(1.0f-interp) + interp*(float)flat[size.x*size.y + blockIdx.y * size.x + idx];
			}

			size_t line = size.y*blockIdx.z + blockIdx.y;
			
			in[line * size.x + idx] = fmaxf(in[line * size.x + idx] - dk, 0.5f) / fmaxf(ft-dk,0.5f);

		}
	}

	void flatdark_log_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats)
	{	
		// Supports 2 flats max
		cudaSetDevice(gpu);

		int b;
		int blocksize = min(nangles,32);

		dim3 blocks = dim3(nrays,nslices,blocksize);
		blocks.x = (nrays+127) / 128;

		rImage data(nrays,nslices,blocksize);

		Image2D<float> cflat(nrays, nslices, numflats);
		Image2D<float> cdark(nrays, nslices);

		for(b = 0; b < nangles; b += blocksize){
			blocksize = min(blocksize,nangles-b);

			data.CopyFrom(frames + (size_t)b*nrays*nslices, 0, (size_t)nrays*nslices*blocksize);
			cflat.CopyFrom(flat, 0, (size_t)nrays*numflats*nslices);
			cdark.CopyFrom(dark, 0, (size_t)nrays*nslices);

			KFlatDarklog<<<blocks,128>>>(data.gpuptr, cdark.gpuptr, cflat.gpuptr, dim3(nrays,nslices,blocksize), cflat.sizez);

			data.CopyTo(frames + (size_t)b*nrays*nslices, 0, (size_t)nrays*nslices*blocksize);
      
      	}

		cudaDeviceSynchronize();
	}

	void flatdark_log_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats)
	{
		int t;
		int blockgpu = (nangles + ngpus - 1) / ngpus;
		
		// printf("Aqui\n");
		std::vector<std::future<void>> threads;

		// printf("Aqui2\n");

		// printf("valores cu: %d, %d,%d,%d %d %d\n",ngpus,nrays,nangles,nslices,numflats,blockgpu);

		for(t = 0; t < ngpus; t++){ 
			
			blockgpu = min(nangles - blockgpu * t, blockgpu);
			// printf("Aqui3 %d %d %ld %d %d\n",t,blockgpu,(size_t)t*blockgpu*nrays*nangles,nrays,nangles);

			threads.push_back(std::async( std::launch::async, flatdark_log_gpu, gpus[t], frames + (size_t)t*blockgpu*nrays*nslices, 
						flat, dark, nrays, nslices, blockgpu, numflats
						));
		}

		for(auto& t : threads)
			t.get();
	}

	void flatdark_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats)
	{	
		// Supports 2 flats max
		cudaSetDevice(gpu);

		int b;
		int blocksize = min(nangles,32);

		dim3 blocks = dim3(nrays,nslices,blocksize);
		blocks.x = (nrays+127) / 128;

		rImage data(nrays,nslices,blocksize);

		Image2D<float> cflat(nrays, nslices, numflats);
		Image2D<float> cdark(nrays, nslices);

		for(b = 0; b < nangles; b += blocksize){
			blocksize = min(blocksize,nangles-b);

			data.CopyFrom(frames + (size_t)b*nrays*nslices, 0, (size_t)nrays*nslices*blocksize);
			cflat.CopyFrom(flat, 0, (size_t)nrays*numflats*nslices);
			cdark.CopyFrom(dark, 0, (size_t)nrays*nslices);

			KFlatDark<<<blocks,128>>>(data.gpuptr, cdark.gpuptr, cflat.gpuptr, dim3(nrays,nslices,blocksize), cflat.sizez);

			data.CopyTo(frames + (size_t)b*nrays*nslices, 0, (size_t)nrays*nslices*blocksize);
      
      	}

		cudaDeviceSynchronize();
	}

	void flatdark_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats)
	{
		int t;
		int blockgpu = (nangles + ngpus - 1) / ngpus;
		
		// printf("Aqui\n");
		std::vector<std::future<void>> threads;

		// printf("Aqui2\n");

		// printf("valores cu: %d, %d,%d,%d %d %d\n",ngpus,nrays,nangles,nslices,numflats,blockgpu);

		for(t = 0; t < ngpus; t++){ 
			
			blockgpu = min(nangles - blockgpu * t, blockgpu);
			// printf("Aqui3 %d %d %ld %d %d\n",t,blockgpu,(size_t)t*blockgpu*nrays*nangles,nrays,nangles);

			threads.push_back(std::async( std::launch::async, flatdark_gpu, gpus[t], frames + (size_t)t*blockgpu*nrays*nslices, 
						flat, dark, nrays, nslices, blockgpu, numflats
						));
		}

		for(auto& t : threads)
			t.get();
	}


	void _CPUReduceBLock16(float* out, uint16_t* frames, uint16_t* cflat, uint16_t* cdark, 
		size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats, int tidx, int nthreads)
	{
		for(size_t z = (size_t)tidx; z < sizez; z += (size_t)nthreads)
		{
			for(size_t by = 0; by < block; by++)
			{
				for(size_t x = 0; x < sizex; x++)
				{
					size_t step = sizey/block;
					float val = 0;

					for(size_t fy = 0; fy < step; fy++)
					{
						float flat = cflat[(by*step+fy)*sizex + x];
						float dark = cdark[(by*step+fy)*sizex + x];

						if(numflats>1)
						{
							float interp = float(z+1)/float(sizez+1);
							flat = flat*(1.0f-interp) + interp*cflat[sizex*sizey + (by*step+fy)*sizex + x];
						}

						val += -logf( fmaxf(frames[z*sizex*sizey + (by*step+fy)*sizex + x]-dark,0.5f) / fmaxf(flat-dark,0.5f) );

					}
					out[by*sizez*sizex + sizex*z + x] = val;
				}
			}
		}
	}

	void CPUReduceBLock16(float* out, uint16_t* frames, uint16_t* cflat, uint16_t* cdark, 
	size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats)
	{
		std::vector<std::future<void>> threads;
		for(int t=0; t<16; t++)
			threads.push_back( std::async(std::launch::async, _CPUReduceBLock16, out, frames, cflat, cdark, 
				sizex, sizey, sizez, block, numflats, t, 16) );

		for(int t=0; t<16; t++)
				threads[t].get();
	}
}

extern "C"{
	void flatdarkcpu(float* out, float* frames, float* cflat, float* cdark, 
		size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats, int tidx, int nthreads)
	{
		for(size_t z = (size_t)tidx; z < sizez; z += (size_t)nthreads)
		{
			for(size_t by = 0; by < block; by++)
			{
				for(size_t x = 0; x < sizex; x++)
				{
					size_t step = sizey/block;
					float val = 0;

					for(size_t fy = 0; fy < step; fy++)
					{
						float flat = cflat[(by*step+fy)*sizex + x];
						float dark = cdark[(by*step+fy)*sizex + x];

						if(numflats>1)
						{
							float interp = float(z+1)/float(sizez+1);
							flat = flat*(1.0f-interp) + interp*cflat[sizex*sizey + (by*step+fy)*sizex + x];
						}

						val += fmaxf(frames[z*sizex*sizey + (by*step+fy)*sizex + x]-dark,0.5f) / fmaxf(flat-dark,0.5f);

					}
					out[by*sizez*sizex + sizex*z + x] = val;
				}
			}
		}
	}

	void flatdarkcpu_log(float* out, float* frames, float* cflat, float* cdark, 
		size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats, int tidx, int nthreads)
	{
		for(size_t z = (size_t)tidx; z < sizez; z += (size_t)nthreads)
		{
			for(size_t by = 0; by < block; by++)
			{
				for(size_t x = 0; x < sizex; x++)
				{
					size_t step = sizey/block;
					float val = 0;

					for(size_t fy = 0; fy < step; fy++)
					{
						float flat = cflat[(by*step+fy)*sizex + x];
						float dark = cdark[(by*step+fy)*sizex + x];

						if(numflats>1)
						{
							float interp = float(z+1)/float(sizez+1);
							flat = flat*(1.0f-interp) + interp*cflat[sizex*sizey + (by*step+fy)*sizex + x];
						}

						val += -logf( fmaxf(frames[z*sizex*sizey + (by*step+fy)*sizex + x]-dark,0.5f) / fmaxf(flat-dark,0.5f) );

					}
					out[by*sizez*sizex + sizex*z + x] = val;
				}
			}
		}
	}

	void flatdarkcpu_block(float* out, float* frames, float* cflat, float* cdark, 
	size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats)
	{
		std::vector<std::future<void>> threads;
		for(int t=0; t<16; t++)
			threads.push_back( std::async(std::launch::async, flatdarkcpu, out, frames, cflat, cdark, 
				sizex, sizey, sizez, block, numflats, t, 16) );

		for(int t=0; t<16; t++)
				threads[t].get();
	}

	void flatdarkcpu_log_block(float* out, float* frames, float* cflat, float* cdark, 
		size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats)
		{
			std::vector<std::future<void>> threads;
			for(int t=0; t<16; t++)
				threads.push_back( std::async(std::launch::async, flatdarkcpu_log, out, frames, cflat, cdark, 
					sizex, sizey, sizez, block, numflats, t, 16) );
	
			for(int t=0; t<16; t++)
					threads[t].get();
		}
}