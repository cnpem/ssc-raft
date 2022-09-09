#include "../../../inc/include.h"
#include "../../../inc/common/types.hpp"
#include "../../../inc/common/kernel_operators.hpp"
#include "../../../inc/common/complex.hpp"
#include "../../../inc/common/operations.hpp"
#include "../../../inc/common/logerror.hpp"

extern "C"{
	void SinoFilter(float* sino, size_t nrays, size_t nangles, size_t blocksize, int csino, bool bRampFilter, CFilter reg, bool bShiftCenter, float* sintable)
	{	
		cImage fft(nrays/2+1,nangles);
		cImage fft2(nrays/2+1,nangles);

		cufftHandle plan_r2c, plan_c2r;
		cufftPlan1d(&plan_r2c, nrays, CUFFT_R2C, nangles);
		cufftPlan1d(&plan_c2r, nrays, CUFFT_C2R, nangles);

		dim3 blk = fft.ShapeBlock();
		dim3 thr = fft.ShapeThread();

		for(int k=0; k<blocksize; k++)
		{
			HANDLE_FFTERROR( cufftExecR2C(plan_r2c, sino+k*nrays*nangles, fft.gpuptr) );

			if(bRampFilter)
				BandFilterReg<<<blk,thr>>>(fft.gpuptr, nrays/2+1, csino, bShiftCenter, sintable, reg);
			else
				std::cout << __FILE__ << " " << __LINE__ << " " << "Auto reg missing!" << std::endl;

			HANDLE_FFTERROR( cufftExecC2R(plan_c2r, fft.gpuptr, sino+k*nrays*nangles) );
		}
		
		cufftDestroy(plan_r2c);
		cufftDestroy(plan_c2r);
	}

	void Highpass(rImage& x, float wid)
	{
		size_t sizex = x.sizex;
		size_t sizey = x.sizey;

		cImage fourier(sizex/2+1, sizey);
		cufftHandle planrc;
		cufftHandle plancr;

		// Optimize allocation
		cufftPlan1d(&planrc, sizex, CUFFT_R2C, sizey);
		cufftPlan1d(&plancr, sizex, CUFFT_C2R, sizey);

		for(size_t bz=0; bz<x.sizez; bz++)
		{
			cufftExecR2C(planrc, x.gpuptr + sizex*sizey*bz, fourier.gpuptr);
			KFilter<<<dim3((sizex/2+32)/32,sizey),32>>>(fourier.gpuptr, sizex, wid);
			cufftExecC2R(plancr, fourier.gpuptr, x.gpuptr + sizex*sizey*bz);
		}

		cufftDestroy(planrc);
		cufftDestroy(plancr);
	}

	__global__ void KFilter(complex* x, size_t sizex, float wid)
	{
		const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

		if(idx > sizex/2)
			return;

		float xs = float(idx)*wid/sizex;
		x[blockIdx.y*(sizex/2+1) + idx] *= (1.0f - expf(-20.0f*xs*xs))/sizex;
	}

	__global__ void BandFilterReg(complex* vec, size_t sizex, int icenter, bool bShiftCenter, float* sintable, CFilter mfilter)
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

		float rampfilter = float(tx) / (float)sizex;
		rampfilter = mfilter.Apply(rampfilter);

		float fcenter = 1.0f - (bShiftCenter ? (sintable[ty]) : 0);
		fcenter = -2*float(M_PI)/float(2*sizex-2) * fcenter * icenter;

		if(tx < sizex)
			vec[ty*sizex + tx] *= exp1j(fcenter * tx) * rampfilter;
	}

	__device__ complex DeltaFilter(complex* img, int sizeimage, float fx, float fy)
	{
		fx = fminf(fx, sizeimage/2-1E-4f);
		int ix = int(fx);
		int iy = int(fy);

		float a = fx-ix;
		float b = fy-iy;

		const int h2n = sizeimage/2+1;

		return  img[(iy%sizeimage)*h2n + ix]*(1-a)*(1-b) +
				img[((iy+1)%sizeimage)*h2n + ix]*(1-a)*b +
				img[(iy%sizeimage)*h2n + (ix+1)]*a*(1-b) +
				img[((iy+1)%sizeimage)*h2n + (ix+1)]*a*b;
	}
	
	inline __global__ void SetX(complex* out, float* in, int sizex)
	{
		size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t ty = blockIdx.y + gridDim.y * blockIdx.z;
		
		if(tx < sizex)
		{
			out[ty*sizex + tx].x = in[ty*sizex + tx];
			out[ty*sizex + tx].y = 0;
		}
	}
	
	inline __global__ void GetX(float* out, complex* in, int sizex)
	{
		size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t ty = blockIdx.y + gridDim.y * blockIdx.z;
		
		if(tx < sizex)
			out[ty*sizex + tx] = (in[ty*sizex + tx].x)/sizex;
	}
	
	inline __global__ void GetXBST(void* out, complex* in, size_t sizex, float threshold, EType::TypeEnum raftDataType, int rollxy)
	{
		size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t ty = blockIdx.y + blockDim.y * blockIdx.z;
		
		if(tx >= sizex)
			return;
		
		float fpixel = (in[ty*sizex + tx].x)/float(sizex);
		 BasicOps::set_pixel(out, fpixel, tx, ty, sizex, threshold, raftDataType);
	}

	inline __global__ void BandFilterC2C(complex* vec, size_t sizex, int center, CFilter mfilter = CFilter())
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

		float rampfilter = 2.0f*fminf(tx,sizex-tx)/(float)sizex;
		rampfilter = mfilter.Apply(rampfilter);

		if(tx < sizex)
			vec[ty*sizex + tx] *= exp1j(-2*float(M_PI)/float(sizex) * center * tx) * rampfilter;
	}

	void BSTFilter(cufftHandle plan, complex* filtersino, float* sinoblock, size_t nrays, size_t nangles, int csino, CFilter reg)
	{

		dim3 filterblock((nrays+255)/256,nangles,1);
		dim3 filterthread(256,1,1);

		SetX<<<filterblock,filterthread>>>(filtersino, sinoblock, nrays);
			
		HANDLE_FFTERROR(cufftExecC2C(plan, filtersino, filtersino, CUFFT_FORWARD));
			
		BandFilterC2C<<<filterblock,filterthread>>>(filtersino, nrays, csino, reg);
			
		HANDLE_FFTERROR(cufftExecC2C(plan, filtersino, filtersino, CUFFT_INVERSE));
		
		GetX<<<filterblock,filterthread>>>(sinoblock, filtersino, nrays);

		//cudaMemset(sinoblock, 0, nrays*nangles*4);
	}

}

