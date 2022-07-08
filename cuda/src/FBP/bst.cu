// Authors: Gilberto Martinez, Giovanni Baraldi, Eduardo X. Miqueles

#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"

// Call for dim3.x = half the number of nrays

extern "C"{
	__global__ void sino2p(float* padded, float* in, size_t nrays, size_t nangles, int pad0, int csino)
	{
		int center = nrays/2 - csino;
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		
		if(idx < nrays/2)
		{
			size_t fory = blockIdx.y;
			size_t revy = blockIdx.y + nangles;
			
			size_t slicez = blockIdx.z * nrays * nangles;
			
			float Arg2 = (2.0f*idx - nrays*pad0/2 + 1.0f)/(nrays*pad0/2 - 1.0f);
			float b1 = cyl_bessel_i0f(1.8f*sqrtf(1.0f - Arg2 * Arg2));
			float b2 = cyl_bessel_i0f(1.8f);
			float w_bessel = fabsf(b1)/fabsf(b2);
			
			if(center - 1 - idx >= 0)
				padded[pad0*slicez + pad0/2*fory*nrays + idx] = w_bessel * in[slicez + fory*nrays + center - 1 - idx];
			else
				padded[pad0*slicez + pad0/2*fory*nrays + idx] = w_bessel * in[slicez + fory*nrays];
				
			if(center + 0 + idx >= 0)
				padded[pad0*slicez + pad0/2*revy*nrays + idx] = w_bessel * in[slicez + fory*nrays + center + 0 + idx];
			else
				padded[pad0*slicez + pad0/2*revy*nrays + idx] = w_bessel * in[slicez + fory*nrays];
		}
	}

	__global__ void sino2p_NoBeer(float* padded, float* in, size_t nrays, size_t nangles, int pad0, int csino)
	{
		int center = nrays/2 - csino;
		int idx = blockIdx.x*blockDim.x + threadIdx.x;
		
		if(idx < nrays/2)
		{
			size_t fory = blockIdx.y;
			size_t revy = blockIdx.y + nangles;
			
			size_t slicez = blockIdx.z * nrays * nangles;
			
			if(center - 1 - idx >= 0)
				padded[pad0*slicez + pad0/2*fory*nrays + idx] = in[slicez + fory*nrays + center - 1 - idx];
			else
				padded[pad0*slicez + pad0/2*fory*nrays + idx] = in[slicez + fory*nrays];
				
			if(center + 0 + idx >= 0)
				padded[pad0*slicez + pad0/2*revy*nrays + idx] = in[slicez + fory*nrays + center + 0 + idx];
			else
				padded[pad0*slicez + pad0/2*revy*nrays + idx] = in[slicez + fory*nrays];
		}
	}

	__global__ void convBST(complex* block, size_t nrays, size_t nangles)
	{
		size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t ty = blockIdx.y;
		size_t tz = blockIdx.z;
		
		float sigma = 2.0f / (nrays * (fmaxf(fminf(tx,nrays-tx),1.0f)));
		size_t offset = tz*nangles*nrays + ty*nrays + tx;
		
		block[offset] *= sigma;
	}

	__global__ void polar2cartesian_fourier(complex* cartesian, complex* polar, size_t nrays, size_t nangles, size_t sizeimage)
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y;
		
		if(tx < sizeimage)
		{
			size_t cartplane = blockIdx.z * sizeimage * sizeimage;
			polar += blockIdx.z * nrays * nangles;
			
			int posx = tx - sizeimage/2;
			int posy = ty - sizeimage/2;
			
			float rho = nrays * hypotf(posx,posy) / sizeimage;
			float angle = (nangles)*(0.5f*atan2f(posy, posx)/float(M_PI)+0.5f);
			
			size_t irho = size_t(rho);
			int iarc = int(angle);
			complex interped = 0;
			
			if(irho < nrays/2-1)
			{
				float pfrac = rho-irho;
				float tfrac = iarc-angle;
				
				iarc = iarc%(nangles);
				
				int uarc = (iarc+1)%(nangles);
				
				complex interp0 = polar[iarc*nrays + irho]*(1.0f-pfrac) + polar[iarc*nrays + irho+1]*pfrac;
				complex interp1 = polar[uarc*nrays + irho]*(1.0f-pfrac) + polar[uarc*nrays + irho+1]*pfrac;
				
				interped = interp0*tfrac + interp1*(1.0f-tfrac);
			}
			
			cartesian[cartplane + sizeimage*((ty+sizeimage/2)%sizeimage) + (tx+sizeimage/2)%sizeimage] = interped*(4*(tx%2-0.5f)*(ty%2-0.5f));
		}
	}

	void GPUFST(char* blockRecon, float *wholesinoblock, int Nrays, int Nangles, int trueblocksize, int sizeimage, int csino, int pad0,
		float threshold, EType raftDataType)
	{
		int blocksize = 1;
		
		cImage cartesianblock(sizeimage, sizeimage*blocksize);
		cImage polarblock(Nrays * pad0,Nangles*blocksize);
		rImage realpolar(Nrays * pad0,Nangles*blocksize);
		
		cufftHandle plan1d;
		cufftHandle plan2d;
	
		int dimms1d[] = {(int)Nrays*pad0/2};
		int dimms2d[] = {(int)sizeimage,(int)sizeimage};
		int beds[] = {Nrays*pad0/2};
	
		HANDLE_FFTERROR( cufftPlanMany(&plan1d, 1, dimms1d, beds, 1, Nrays*pad0/2, beds, 1, Nrays*pad0/2, CUFFT_R2C, Nangles*blocksize*2) );
		HANDLE_FFTERROR( cufftPlanMany(&plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize) );
		
		size_t insize = Nrays*Nangles;
		size_t outsize = sizeimage*sizeimage;
		
		for(size_t zoff = 0; zoff < (size_t)trueblocksize; zoff+=blocksize)
		{
			float* sinoblock = wholesinoblock + insize*zoff;
			
			dim3 blocks((Nrays+255)/256,Nangles,blocksize);
			dim3 threads(128,1,1); 
			
			sino2p_NoBeer<<<blocks,threads>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, pad0, csino);
			
			Nangles *= 2;
			Nrays *= pad0/2;

			blocks.y *= 2;
			blocks.x *= pad0/2;

			HANDLE_FFTERROR(cufftExecR2C(plan1d, realpolar.gpuptr, polarblock.gpuptr));
			
			blocks = dim3((sizeimage+255)/256,sizeimage,blocksize);
			threads = dim3(256,1,1);

			polar2cartesian_fourier<<<blocks,threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, Nrays, Nangles, sizeimage);
		
			HANDLE_FFTERROR( cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE) );
			
			GetXBST<<<dim3((sizeimage+127)/128,sizeimage),128>>>(blockRecon + outsize*zoff*raftDataType.Size(), 
					cartesianblock.gpuptr, sizeimage, threshold, raftDataType.type, csino);
			HANDLE_ERROR( cudaPeekAtLastError() );
			
			Nangles /= 2;
			Nrays /= pad0/2;
		}
		cufftDestroy(plan1d);
		cufftDestroy(plan2d);
	}

	void GPUBST(char* blockRecon, float *wholesinoblock, int Nrays, int Nangles, int trueblocksize, int sizeimage, int csino, int pad0,
		float threshold, CFilter reg, EType raftDataType)
	{
		int blocksize = 1;
		cImage filtersino(Nrays, Nangles*blocksize);
		
		cImage cartesianblock(sizeimage, sizeimage*blocksize);
		cImage polarblock(Nrays * pad0,Nangles*blocksize);
		rImage realpolar(Nrays * pad0,Nangles*blocksize);
		
		cufftHandle filterplan;
		int dimmsfilter[] = {Nrays};
		HANDLE_FFTERROR( cufftPlanMany(&filterplan, 1, dimmsfilter, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, Nangles*blocksize) );
		
		cufftHandle plan1d;
		cufftHandle plan2d;
	
		int dimms1d[] = {(int)Nrays*pad0/2};
		int dimms2d[] = {(int)sizeimage,(int)sizeimage};
		int beds[] = {Nrays*pad0/2};
	
		HANDLE_FFTERROR( cufftPlanMany(&plan1d, 1, dimms1d, beds, 1, Nrays*pad0/2, beds, 1, Nrays*pad0/2, CUFFT_R2C, Nangles*blocksize*2) );
		HANDLE_FFTERROR( cufftPlanMany(&plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize) );
		
		size_t insize = Nrays*Nangles;
		size_t outsize = sizeimage*sizeimage;
		
		for(size_t zoff = 0; zoff < (size_t)trueblocksize; zoff+=blocksize)
		{
			float* sinoblock = wholesinoblock + insize*zoff;
			BSTFilter(filterplan, filtersino.gpuptr, sinoblock, Nrays, Nangles, csino, reg);
			
			dim3 blocks((Nrays+255)/256,Nangles,blocksize);
			dim3 threads(128,1,1); 
			
			sino2p<<<blocks,threads>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, pad0, 0);
			
			Nangles *= 2;
			Nrays *= pad0/2;

			blocks.y *= 2;
			blocks.x *= pad0/2;

			HANDLE_FFTERROR(cufftExecR2C(plan1d, realpolar.gpuptr, polarblock.gpuptr));
			convBST<<<blocks,threads>>>(polarblock.gpuptr, Nrays, Nangles);
			
			blocks = dim3((sizeimage+255)/256,sizeimage,blocksize);
			threads = dim3(256,1,1);

			polar2cartesian_fourier<<<blocks,threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, Nrays, Nangles, sizeimage);
		
			HANDLE_FFTERROR( cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE) );
		
			cudaDeviceSynchronize();

			GetXBST<<<dim3((sizeimage+127)/128,sizeimage),128>>>(blockRecon + outsize*zoff*raftDataType.Size(), 
					cartesianblock.gpuptr, sizeimage, threshold, raftDataType.type, csino);

			HANDLE_ERROR( cudaPeekAtLastError() );
			
			Nangles /= 2;
			Nrays /= pad0/2;
		}
		cufftDestroy(filterplan);
		cufftDestroy(plan1d);
		cufftDestroy(plan2d);
	}

	void bstgpu(int gpu, float* blockRecon, float* sinoblock, int nrays, int nangles, 
				int isizez, int sizeimage, int csino, float reg_val, int FilterType, float* angles)
	{
		CFilter reg(FilterType,reg_val);
		
		cudaSetDevice(gpu);
		size_t sizez = size_t(isizez);
		
		rImage sino(nrays, nangles, min(sizez,32ul), MemoryType::EAllocGPU);
		rImage recon(sizeimage,sizeimage, min(sizez,32ul), MemoryType::EAllocGPU);

		for(size_t b=0; b < sizez; b += 32){
			size_t blocksize = min(sizez-b,32ul);
			size_t reconoffset = b*sizeimage*sizeimage;

			sino.CopyFrom(sinoblock + b*nrays*nangles, 0, blocksize*nrays*nangles);

			GPUBST((char*)recon.gpuptr, sino.gpuptr, nrays, nangles, blocksize, sizeimage, csino, 16, 0, reg, EType::TypeEnum::FLOAT32);
			recon.CopyTo(reconoffset + blockRecon, 0, blocksize*sizeimage*sizeimage);
		}
		cudaDeviceSynchronize();
	}

	void bstblock(int* gpus, int ngpus, float* recon, float* tomogram, int nrays, int nangles, 
		int nslices, int reconsize, int centersino, float reg_val, int FilterType, float* angles)
	{
		int t;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		
		std::vector<std::future<void>> threads;

		for(t = 0; t < ngpus; t++){ 
			
			blockgpu = min(nslices - blockgpu * t, blockgpu);

			threads.push_back(std::async( std::launch::async, bstgpu, gpus[t], recon + (size_t)t * blockgpu * reconsize*reconsize, 
				tomogram + (size_t)t * blockgpu * nrays*nangles, nrays, nangles, blockgpu, reconsize, centersino,
				reg_val, FilterType, angles
			));
		}
	
		for(auto& t : threads)
			t.get();
	}

	void fstgpu(int gpu, float* blockRecon, float* sinoblock, int nrays, int nangles, 
		int isizez, int sizeimage, int csino, float reg_val, int FilterType, float* angles)
	{
		CFilter reg(FilterType,reg_val);

		cudaSetDevice(gpu);
		size_t sizez = size_t(isizez);

		rImage sino(nrays, nangles, min(sizez,32ul), MemoryType::EAllocGPU);
		rImage recon(sizeimage,sizeimage, min(sizez,32ul), MemoryType::EAllocGPU);

		for(size_t b=0; b < sizez; b += 32){
			size_t blocksize = min(sizez-b,32ul);
			size_t reconoffset = b*sizeimage*sizeimage;

			sino.CopyFrom(sinoblock + b*nrays*nangles, 0, blocksize*nrays*nangles);

			GPUFST(reconoffset + (char*)blockRecon, sino.gpuptr, nrays, nangles, blocksize, sizeimage, csino, 4, 0, EType::TypeEnum::FLOAT32);
			recon.CopyTo(reconoffset + blockRecon, 0, blocksize*sizeimage*sizeimage);
		}
		cudaDeviceSynchronize();
	}

	void fstblock(int* gpus, int ngpus, float* recon, float* tomogram, int nrays, int nangles, 
		int nslices, int reconsize, int centersino, float reg_val, int FilterType, float* angles)
	{
		int t;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		
		std::vector<std::future<void>> threads;

		for(t = 0; t < ngpus; t++){ 
			
			blockgpu = min(nslices - blockgpu * t, blockgpu);

			threads.push_back(std::async( std::launch::async, fstgpu, gpus[t], recon + (size_t)t * blockgpu * reconsize*reconsize, 
				tomogram + (size_t)t * blockgpu * nrays*nangles, nrays, nangles, blockgpu, reconsize, centersino,
				reg_val, FilterType, angles
			));
		}
	
		for(auto& t : threads)
			t.get();
	}
}