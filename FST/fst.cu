// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi

#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{
	__device__ void filteredAtomicAdd(complex* img, complex value, int sizeimage, float fx, float fy, float mult)
	{
		fx = fminf(fx, sizeimage/2 - 1E-4f);
		int ix = int(fx);
		int iy = int(fy);

		float a = fx - ix;
		float b = fy - iy;

		const int h2n = (sizeimage/2+1);
		
		atomicAdd(img + (iy%sizeimage)*h2n + ix, value*(mult*(1-a)*(1-b)));
		atomicAdd(img + ((iy+1)%sizeimage)*h2n + ix, value*(mult*(1-a)*b));
		atomicAdd(img + (iy%sizeimage)*h2n + ix+1, value*(mult*a*(1-b)));
		atomicAdd(img + ((iy+1)%sizeimage)*h2n + ix+1, value*(mult*a*b));
	}
	
	__global__ void KBackProjection_FST(complex* img, complex* sino, int sizeimage, int nangles)
	{
		int rho = blockIdx.x*blockDim.x + threadIdx.x;
		int ang = blockIdx.y*blockDim.y + threadIdx.y;

		if(rho >= sizeimage/2 || ang >= nangles)
			return;

		sino += (sizeimage/2+1)*nangles*blockIdx.z;
		img += (sizeimage/2+1)*sizeimage*blockIdx.z;

		const float mult = 0.5f*float(M_PI)/(float(nangles)*float(sizeimage*sizeimage));

		float cos_t,sin_t;
		__sincosf(ang*float(M_PI)/nangles, &sin_t, &cos_t);

		float fy = cos_t*rho + sizeimage;
		float fx = sin_t*rho;

		//float lowpass = sinc(2*rho*float(M_PI)/sizeimage);
		complex value = sino[ang*(sizeimage/2+1) + rho].conj();
		filteredAtomicAdd(img, value, sizeimage, fx, fy, mult);
	}

	void BackProjection_FST(float* backptr, float* sinogram, complex* Temp, PlanRC& plan, size_t sizeimage, size_t nangles, size_t sizez)
	{
		complex* Temp2 = Temp + sizez*(sizeimage/2+1)*nangles;

		dim3 thr(32,4,1); 
		dim3 blk((sizeimage/2+31)/32, (nangles+3)/4, sizez);
		
		HANDLE_FFTERROR( cufftExecR2C(plan.plan1d_r2c, sinogram, Temp) );
		
		cudaMemset(Temp2, 0, sizez*sizeimage*(sizeimage/2+1)*sizeof(complex)); // -> Next kernel uses atomicAdd
		
		KBackProjection_FST<<<blk,thr>>>(Temp2, Temp, sizeimage, nangles);

		HANDLE_FFTERROR( cufftExecC2R(plan.plan2d_c2r, Temp2, backptr) );
	}

	void GPU_BackProjection_FST(rMImage& backptr, rMImage& sinogram, cMImage& Temp, PlanRC** plans)
	{
		auto& gpus = sinogram.gpuindices;

		size_t sizeimage = sinogram.sizex;
		size_t nangles = sinogram.sizey;

		size_t Temp2plane = (sizeimage/2+1)*nangles;

		dim3 thr(32,4,1); 
		dim3 blk((sizeimage/2+31)/32, (nangles+3)/4, 1);

		for(size_t bz = 0; bz<sinogram[0].sizez; bz++)
		{
			Temp.SetGPUToZero();
			
			for(int g=0; g<gpus.size(); g++) if( bz < sinogram[g].sizez )
			{
				sinogram.Set(g);
				HANDLE_FFTERROR( cufftExecR2C(plans[g]->plan1d_r2c, sinogram.Ptr(g) + sizeimage*nangles*bz, Temp.Ptr(g)) );
			
				KBackProjection_FST<<<blk,thr>>>(Temp.Ptr(g) + Temp2plane, Temp.Ptr(g), sizeimage, nangles);

				HANDLE_FFTERROR( cufftExecC2R(plans[g]->plan2d_c2r, Temp.Ptr(g) + Temp2plane, backptr.Ptr(g) + sizeimage*sizeimage*bz) );
			}
		}
	}

	__global__ void KRadon_FST(complex* sino, complex* img, int sizeimage, int nangles)
	{
		int rho = blockIdx.x*blockDim.x + threadIdx.x;
		int ang = blockIdx.y*blockDim.y + threadIdx.y;

		int hsize = sizeimage/2;

		if(rho >= hsize || ang >= nangles)
			return;

		sino += (sizeimage/2+1)*nangles*blockIdx.z;
		img += (sizeimage/2+1)*sizeimage*blockIdx.z;

		float cos_t,sin_t;
		__sincosf(ang*float(M_PI)/nangles, &sin_t, &cos_t);

		const float mult = 1.0f/(sizeimage);
		float fpy = rho*cos_t + sizeimage;
		float fpx = rho*sin_t + 1E-6f;

		//float lowpass = sinc(2*rho*float(M_PI)/sizeimage);
		complex lerped = DeltaFilter(img, sizeimage, fpx, fpy)*mult;
		sino[ang*(sizeimage/2+1) + rho] = lerped.conj();
	}

	void Radon_FST(float* sinogram, float* image, complex* Temp, PlanRC& plan, size_t sizeimage, size_t nangles, size_t sizez)
	{
		complex* Temp2 = Temp + sizez*(sizeimage/2+1)*sizeimage;

		dim3 thr(128,1,1); 
		dim3 blk((sizeimage+127)/128, nangles,sizez);
		
		HANDLE_FFTERROR( cufftExecR2C(plan.plan2d_r2c, image, Temp) );

		cudaMemset(Temp2, 0, sizez*(sizeimage/2+1)*nangles*sizeof(complex));
		KRadon_FST<<<blk,thr>>>(Temp2, Temp, sizeimage, nangles);

		HANDLE_FFTERROR( cufftExecC2R(plan.plan1d_c2r, Temp2, sinogram) );
	}

	void GPU_Radon_FST(rMImage& sinogram, rMImage& image, cMImage& Temp, PlanRC** plans)
	{
		auto& gpus = sinogram.gpuindices;
		size_t sizeimage = sinogram.sizex;
		size_t nangles = sinogram.sizey;

		size_t Temp2plane = (sizeimage/2+1)*sizeimage;

		dim3 thr(128,1,1); 
		dim3 blk((sizeimage+127)/128, nangles,1);
		
		for(size_t bz = 0; bz<sinogram[0].sizez; bz++)
		{
			Temp.SetGPUToZero();
			
			for(int g=0; g<gpus.size(); g++) if( bz < sinogram[g].sizez )
			{
				image.Set(g);
				HANDLE_FFTERROR( cufftExecR2C(plans[g]->plan2d_r2c, image.Ptr(g) + sizeimage*sizeimage*bz, Temp.Ptr(g)) )

				KRadon_FST<<<blk, thr>>>(Temp.Ptr(g) + Temp2plane, Temp.Ptr(g), sizeimage, nangles);

				HANDLE_FFTERROR( cufftExecC2R(plans[g]->plan1d_c2r, Temp.Ptr(g) + Temp2plane, sinogram.Ptr(g) + sizeimage*nangles*bz) );
			}
		}
	}
}