#include "../../../inc/processing.h"

extern "C"{
	__global__ void KPhaseCorrelation(complex* ph1, complex* ph2, size_t sizex)
	{
			size_t idx = threadIdx.x + blockIdx.x*blockDim.x;
			size_t idy = blockIdx.y + gridDim.y * blockIdx.z;
			size_t index = idx + idy*sizex;

			if(idx < sizex)
			{
				float gx = fminf(sizex - idx, idx)/sizex;
				float gy = fminf(gridDim.y - blockIdx.y, blockIdx.y)/gridDim.y;
				float gz = fminf(gridDim.z - blockIdx.z, blockIdx.z)/gridDim.z;

				float gauss = expf(-20.0f * (gx*gx + gy*gy + gz*gz) );

				complex mul = ph1[index] * ph2[index].conj();
				ph1[index] = mul * (gauss / (mul.abs() + 1E-15f));
			}
	}
			
	__forceinline__ __device__ void HALFDirectDFT(complex* __restrict__ vec, float sign, int prime, int totalsize, complex* __restrict__ accums)
	{
		complex acc=0;
		int step = totalsize/prime;

		if(threadIdx.x < step)
		{
			for(int j=0; j<prime; j++)
			{
				complex acc = 0;
				complex w1 = exp1j(sign*2.0f*float(M_PI)/prime * j);
				complex wc = 1.0f;
				
				for(int i=0; i<prime; i++)
				{
					acc += vec[threadIdx.x + i*step] * wc;
					wc *= w1;
				}
				accums[threadIdx.x + step*j] = acc;
			}
			complex w1 = exp1j(sign*2.0f*float(M_PI)/totalsize * threadIdx.x);
			complex wc = 1.0f;

			for(int j=0; j<prime; j++)
			{
				vec[threadIdx.x + j*step] = accums[threadIdx.x + step*j] * wc;
				wc *= w1;
			}
		}
		__syncthreads();
	}
}

template<int s>
__forceinline__ __device__ void KFFT_stepS(complex* vec, float sign, int tidx)
{
	int k = tidx%s;
	int j = (tidx/s) * (2*s) + k;

	complex w = exp1j(sign*float(M_PI)/s * k);

	complex e = vec[j];
	complex o = vec[j+s];

	vec[j] = e + o;
	vec[j+s] = (e - o)*w;

	__syncthreads();
}

extern "C"{
	__device__ void KFFT_pot2(complex* vec, float sign, int fftsize, int tidx, int log2size)
	{
			switch(fftsize)
			{
				case 4096: goto l4096;
				case 2048: goto l2048;
				case 1024: goto l1024;
				case 512: goto l512;
				case 256: goto l256;
				case 128: goto l128;
				case 64: goto l64;
				case 32: goto l32;
				case 16: goto l16;
				case 8: goto l8;
				case 4: goto l4;
				case 2: goto l2;
				default: break;
			}

				KFFT_stepS<4096>(vec, sign, tidx);
			l4096:
				KFFT_stepS<2048>(vec, sign, tidx);
			l2048:
				KFFT_stepS<1024>(vec, sign, tidx);
			l1024:
				KFFT_stepS<512>(vec, sign, tidx);
			l512:
				KFFT_stepS<256>(vec, sign, tidx);
			l256:
				KFFT_stepS<128>(vec, sign, tidx);
			l128:
				KFFT_stepS<64>(vec, sign, tidx);
			l64:
				KFFT_stepS<32>(vec, sign, tidx);
			l32:
				KFFT_stepS<16>(vec, sign, tidx);
			l16:
				KFFT_stepS<8>(vec, sign, tidx);
			l8:
				KFFT_stepS<4>(vec, sign, tidx);
			l4:
				KFFT_stepS<2>(vec, sign, tidx);
			l2:
				KFFT_stepS<1>(vec, sign, tidx);
			
			int revv = __brev(tidx) >> (32-log2size); // bit reversal
			
			if(tidx < revv)
			{
					complex temp = vec[revv];
					vec[revv] = vec[tidx];
					vec[tidx] = temp;
			}
			if(tidx+fftsize/2 < revv+1)
			{
					complex temp = vec[revv+1];
					vec[revv+1] = vec[tidx+fftsize/2];
					vec[tidx+fftsize/2] = temp;
			}

			__syncthreads();
	}

	__forceinline__ __device__ void KFFT_any(complex* __restrict__ vec, float sign, int size, complex* __restrict__ temp)
	{
		uint32_t log2size = __clz(__brev(size));
		uint32_t prime = size >> log2size; // prime ==== odd number :D
		uint32_t sizepot2 = 1<<log2size;
			
		if(prime > 1)
			HALFDirectDFT(vec, sign, prime, size, temp);
		if(sizepot2 > 1)
			KFFT_pot2(vec + sizepot2*(threadIdx.x/(sizepot2/2)), sign, sizepot2, threadIdx.x%(sizepot2/2), log2size);
	}
}

template<uint32_t XDIV>
__forceinline__ __device__ complex DPhaseCorrelation(complex f1, complex f2, float tidx, float sx)
{
	size_t bidx = XDIV > 1 ? (blockIdx.x*XDIV + threadIdx.y) : blockIdx.x;
	size_t gsize = XDIV > 1 ? (gridDim.x*XDIV) : gridDim.x;

	float gx = fminf(sx - tidx, tidx)/sx;
	float gy = fminf(gridDim.y - blockIdx.y, blockIdx.y)/gridDim.y;
	float gz = fminf(gsize - bidx, bidx)/gsize;

	float gauss = expf(-20.0f * (gx*gx + gy*gy + gz*gz) );

	complex ph = f1 * f2.conj();
	return ph * (gauss / (ph.abs() + 1E-15f));
}

template<uint32_t XDIV>
__global__ void PhaseCorrZ(complex* __restrict__ data1, complex* __restrict__ data2, size_t sizex)
{
		extern __shared__ complex vec[];
		const int size = 2*blockDim.x;

		int tidx2 = threadIdx.x+size/2;
		const size_t offset1 = blockIdx.y*sizex + XDIV*blockIdx.x+threadIdx.x%XDIV + (threadIdx.x/XDIV + size/XDIV*threadIdx.y)*sizex*gridDim.y;
		const size_t offset2 = blockIdx.y*sizex + XDIV*blockIdx.x+tidx2%XDIV + (tidx2/XDIV + size/XDIV*threadIdx.y)*sizex*gridDim.y;

		size_t tidx_block1 = (threadIdx.x%XDIV)*size + threadIdx.x/XDIV + size/XDIV*threadIdx.y;
		size_t tidx_block2 = (tidx2%XDIV)*size + tidx2/XDIV + size/XDIV*threadIdx.y;
		
		vec[tidx_block1] = data1[offset1];
		vec[tidx_block2] = data1[offset2];
		
		vec[tidx_block1+size*XDIV] = data2[offset1];
		vec[tidx_block2+size*XDIV] = data2[offset2];

		complex* ffttemp = vec + (2*blockDim.y+threadIdx.y)*size;
		__syncthreads();
		KFFT_any(vec + threadIdx.y*size, -1, size, ffttemp);
		KFFT_any(vec + threadIdx.y*size + size*XDIV, -1, size, ffttemp);
		__syncthreads();

		size_t localoffset1 = threadIdx.x + threadIdx.y*size;
		size_t localoffset2 = localoffset1 + size/2;
		
		vec[localoffset1] = DPhaseCorrelation<XDIV>(vec[localoffset1], vec[localoffset1 + size*XDIV], threadIdx.x, size);
		vec[localoffset2] = DPhaseCorrelation<XDIV>(vec[localoffset2], vec[localoffset2 + size*XDIV], threadIdx.x+size/2, size);

		__syncthreads();
		KFFT_any(vec + threadIdx.y*size, 1, size, ffttemp);
		__syncthreads();

		data1[offset1] = vec[tidx_block1];
		data1[offset2] = vec[tidx_block2];
}

extern "C"{
	__device__ void FULLDirectDFT(complex* vec, float sign, int size)
	{
		complex acc = 0;
		for(int i=0; i<size; i++)
			acc += vec[i] * exp1j(sign*2.0f*float(M_PI)/size * i * threadIdx.x);

		__syncthreads();
		vec[threadIdx.x] = acc;
	}
}

template<>
__global__ void PhaseCorrZ<0>(complex* __restrict__ data1, complex* __restrict__ data2, size_t sizex)
{
		extern __shared__ complex vec[];
		const int size = blockDim.x;
		const size_t offset1 = blockIdx.y*sizex + blockIdx.x + threadIdx.x*sizex*gridDim.y;

		size_t tidx_block1 = threadIdx.x;
		
		vec[tidx_block1] = data1[offset1];
		vec[tidx_block1+size] = data2[offset1];

		__syncthreads();
		FULLDirectDFT(vec, -1, size);
		FULLDirectDFT(vec + size, -1, size);
		__syncthreads();

		complex ph = vec[threadIdx.x] * vec[threadIdx.x + size].conj();
		vec[threadIdx.x] = ph/(ph.abs()+1E-7f);
		
		__syncthreads();
		FULLDirectDFT(vec, 1, size);
		__syncthreads();

		data1[offset1] = vec[tidx_block1];
}

extern "C"{
	void FFTZ_PhaseCorrelation(cImage& img1, cImage& img2, bool bAplyDFTZ = true)
	{
		complex* vec1 = img1.gpuptr;
		complex* vec2 = img2.gpuptr;
		size_t sx = img1.sizex;
		size_t sy = img1.sizey;
		size_t sz = img1.sizez;

		if(sz > 1 && bAplyDFTZ)
		{
			size_t shbytes_line = sz*sizeof(complex);

			if(__builtin_popcount(uint32_t(sz)) > 1)
				shbytes_line *= 3;
			else
				shbytes_line *= 2;
		
			if(sz % 2 == 0)
			{
				if(sx%8 == 0 && sz%8 == 0 && sz <= 32)
					PhaseCorrZ<8><<<dim3(sx/8,sy,1),dim3(sz/2,8,1),8*shbytes_line>>>(vec1,vec2,sx);
				else if(sx%4 == 0 && sz%4 == 0 && sz <= 128)
					PhaseCorrZ<4><<<dim3(sx/4,sy,1),dim3(sz/2,4,1),4*shbytes_line>>>(vec1,vec2,sx);
				else if(sx%2 == 0 && sz <= 512)
					PhaseCorrZ<2><<<dim3(sx/2,sy,1),dim3(sz/2,2,1),2*shbytes_line>>>(vec1,vec2,sx);
				else
					PhaseCorrZ<1><<<dim3(sx,sy,1),sz/2,shbytes_line>>>(vec1,vec2,sx);
			}
			else
				PhaseCorrZ<0><<<dim3(sx,sy,1),sz,shbytes_line>>>(vec1,vec2,sx);
		}
		else
			KPhaseCorrelation<<<img1.ShapeBlock(),img1.ShapeThread()>>>(vec1, vec2, sx);
		
		HANDLE_ERROR(cudaGetLastError());
	}

	__global__ void KMaxReduction(complex* img, float* maxvec, size_t* maxpos, size_t sizex, size_t size)
	{
		size_t index = threadIdx.x + blockIdx.x*blockDim.x;
		if(index < size)
		{
			float maxx = -1;
			size_t mpos = 0;
			for(size_t i=index; i<size; i+=sizex)
			{
				float cabs2 = img[i].abs2();
				if(cabs2 > maxx)
				{
					maxx = cabs2;
					mpos = i;
				}
				__syncwarp();
			}
			maxpos[index] = mpos;
			maxvec[index] = maxx;
		}
	}
}

template<typename Type>
__global__ void TransferMemory(const float* sinos, Type* restrict img1, Type* restrict img2, size_t sxy, size_t ips, size_t sz)
{
	size_t gindex = threadIdx.x + blockDim.x*blockIdx.x;
	if(gindex < sxy)
	{
		for(size_t idz = 0; idz < sz; idz ++)
		{
			img1[gindex + idz*sxy] = sinos[gindex + idz*ips];
			img2[gindex + idz*sxy] = sinos[gindex + idz*ips + sxy];
		}
	}
}

extern "C"{
	int Tomo360_PhaseCorrelation(float* sinograms, size_t sizex, size_t sizey, size_t sizez)
	{
		const size_t inputplanesize = sizey*sizex;
		sizey /= 2;

		cImage img1(sizex,sizey,sizez);
		cImage img2(sizex,sizey,sizez);

		size_t sxy = sizex*sizey;
		TransferMemory<<<(sxy+127)/128,128>>>(sinograms, img1.gpuptr, img2.gpuptr, sxy, inputplanesize, sizez);

		cufftHandle plan1,plan2;
		HANDLE_FFTERROR( cufftPlan1d(&plan1, sizex, CUFFT_C2C, sizey) );

		int n[] = {(int)sizey};
		int inembed[] = {(int)sizey};

		cufftPlanMany(&plan2, 1, n, inembed, sizex, 1, inembed, sizex, 1, CUFFT_C2C, sizex);

		for(size_t i=0; i<sizez; i++)
		{
			size_t zoff = i*sizex*sizey;
			complex* img1ptr = img1.gpuptr + zoff;
			complex* img2ptr = img2.gpuptr + zoff;

			HANDLE_FFTERROR( cufftExecC2C(plan1, img1ptr, img1ptr, CUFFT_FORWARD) );
			HANDLE_FFTERROR( cufftExecC2C(plan1, img2ptr, img2ptr, CUFFT_INVERSE) ); // Flip sinogram2 in x direction
			HANDLE_FFTERROR( cufftExecC2C(plan2, img1ptr, img1ptr, CUFFT_FORWARD) );
			HANDLE_FFTERROR( cufftExecC2C(plan2, img2ptr, img2ptr, CUFFT_FORWARD) );
		}

		FFTZ_PhaseCorrelation(img1, img2);
		
		for(size_t i=0; i<sizez; i++)
		{
			size_t zoff = i*sizex*sizey;
			complex* img1ptr = img1.gpuptr + zoff;

			HANDLE_FFTERROR( cufftExecC2C(plan1, img1ptr, img1ptr, CUFFT_INVERSE) );
			HANDLE_FFTERROR( cufftExecC2C(plan2, img1ptr, img1ptr, CUFFT_INVERSE) );
		}

		img1.LoadFromGPU();

		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

		cufftDestroy(plan1);
		cufftDestroy(plan2);

		int posx = 0;
		float maxx = 0;
		for(int k=-int(sizez/4); k<int(sizez+3)/4; k++) for(int j=-int(sizez/4); j<int(sizez+3)/2; j++) for(size_t i = 0; i<sizex; i++)
		{
			size_t idx = (((k+sizez)%sizez)*sizey+(j+sizey)%sizey)*sizex + i;
			float abss = img1.cpuptr[idx].abs2();
			if(abss > maxx)
			{
				maxx = abss;
				posx = i;
			}
		}
		
		/*
		rImage maxvec(sizey,sizez);
		Image2D<size_t> posvec(sizey,sizez);

		KMaxReduction<<<(sizex+31)/32,32>>>(img1.gpuptr, maxvec.gpuptr, posvec.gpuptr, sizex, sizex*sizey*sizez);
		HANDLE_ERROR(cudaGetLastError());

		maxvec.LoadFromGPU();
		posvec.LoadFromGPU();

		cudaDeviceSynchronize();

		size_t imax = 0;
		float maxx = -1;

		for(size_t k=0; k<16; k++) for(size_t j=0; j<16; j++)
		{
			size_t i = sizey*((k+sizez-8)%sizez) + ((j+sizey-8)%sizey);
			if(maxvec.cpuptr[i] > maxx)
			{
				maxx = maxvec.cpuptr[i];
				imax = posvec.cpuptr[i];
			}
		} 

		int posx = int(imax%sizex); */

		//if(posx > (int)sizex/2)
		posx -= sizex;
		return posx/2;
	}

	__global__ void KJoinX(float* sinogram, const float* temp1, const float* temp2, size_t sizex, int offset)
	{
		size_t outdx = threadIdx.x + blockDim.x*blockIdx.x;

		if(outdx < sizex)
		{
			size_t indx = offset > 0 ? outdx : (sizex-1-outdx);
			offset = abs(offset);

			float coef = fminf(0.5f+((int)outdx-offset)*0.5f/(offset+1E-3f),1.0f);

			atomicAdd(sinogram + blockIdx.y*sizex*2 + sizex-1-outdx+offset,coef*temp1[blockIdx.y*sizex + indx]);
			atomicAdd(sinogram + blockIdx.y*sizex*2 + outdx + sizex-offset,coef*temp2[blockIdx.y*sizex + indx]);

			if(outdx <= offset)
			{
				sinogram[blockIdx.y*sizex*2 + outdx] = temp2[blockIdx.y*sizex + 0];
				sinogram[blockIdx.y*sizex*2 + 2*sizex - 1 - outdx] = temp1[blockIdx.y*sizex + 0];
			}
		}
	}

	void Tomo360_To_180(float* sinograms, size_t sizex, size_t sizey, size_t sizez, int offset)
	{
		rImage temp(sizex,sizey);
		const size_t ips = sizey*sizex; // input place size

		sizey /= 2;

		const size_t sxy = sizey*sizex; // output place size. Note that sxy != ips/2 if sizey%2 == 1

		for(size_t z=0; z<sizez; z++)
		{
			// printf("for z: %ld %ld %ld %ld %ld\n",z,sizez,ips,sxy,sizey);
			float* temp1 = temp.gpuptr;
			float* temp2 = temp1 + ips/2;

			TransferMemory<<<(sxy+127)/128,128>>>(sinograms + z*ips, temp1, temp2, sxy, 0, 1);
			cudaMemset(sinograms + z*2*sxy, 0, ips*sizeof(float));
			KJoinX<<<dim3((sizex+127)/128,sizey,1),128>>>(sinograms + z*2*sxy, temp1, temp2, sizex, offset); // careful with odd sizey
		}
	}
}



extern "C"{
	int ComputeTomo360Offsetgpu(int gpu, float* cpusinograms, int sizex, int sizey, int sizez)
	{
		cudaSetDevice(gpu);
		rImage sinograms(cpusinograms, sizex, sizey*sizez);
		
		int offset = Tomo360_PhaseCorrelation(sinograms.gpuptr, sizex, sizey, sizez);
		
		cudaDeviceSynchronize();		
		
		return offset;
	}
	
	void Tomo360To180gpu(int gpu, float* data, int nrays, int nangles, int nslices, int offset)
	{
		cudaSetDevice(gpu);
		// printf("tomo360 gpou block: %d %d %d\n",nslices,nrays,nangles);

		rImage sinograms(nrays, (size_t)nangles*nslices);
		
		sinograms.CopyFrom(data, 0, (size_t)nrays*nangles*nslices);
		Tomo360_To_180(sinograms.gpuptr, (size_t)nrays, (size_t)nangles, (size_t)nslices, offset);
		sinograms.CopyTo(data, 0, (size_t)nrays*nangles*nslices);

		cudaDeviceSynchronize();
	}

	void Tomo360To180block(int* gpus, int ngpus, float* data, int nrays, int nangles, int nslices, int offset)
	{
		 int t;
		 int blockgpu = (nslices + ngpus - 1) / ngpus;
		 
		 std::vector<std::future<void>> threads;

		 for(t = 0; t < ngpus; t++){ 
			  
			blockgpu = min(nslices - blockgpu * t, blockgpu);
			// printf("tomo360 block: %d %d %d %ld \n",blockgpu,t, nslices, (size_t)t * blockgpu * nrays * nangles);

			threads.push_back(std::async( std::launch::async, Tomo360To180gpu, gpus[t], data + (size_t)t * blockgpu * nrays * nangles, nrays, nangles, blockgpu, offset));
		 }
	
		 for(auto& t : threads)
			  t.get();
	}

	int ComputeTomo360Offset16(int gpu, uint16_t* frames, uint16_t* cflat, uint16_t* cdark, int sizex, int sizey, int sizez, int numflats)
   {
		cudaSetDevice(gpu);

		size_t memframe = 16*sizex*sizey;
		size_t maxusage = 1ul<<32;
		size_t block = min(max(maxusage/memframe,size_t(1)),size_t(sizey));

		rImage avgsino(sizex,block,sizez);
		// Image2D<uint16_t> FLAT(cflat,sizex,sizey,numflats);
		// Image2D<uint16_t> DARK(cdark,sizex,sizey,1);
	
		// CPUReduceBLock16(avgsino.cpuptr, frames, cflat, cdark, sizex, sizey, sizez, block, numflats);
		// avgsino.LoadToGPU();

		int offset = Tomo360_PhaseCorrelation(avgsino.gpuptr, sizex, sizez, block);

		cudaDeviceSynchronize();

		return offset;
	}

}