// Authors: Giovanni Baraldi, Eduardo X. Miqueles

#include "../../../../inc/sscraft.h"

__global__ void SetX(complex* out, float* in, int sizex)
{
	size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t ty = blockIdx.y + gridDim.y * blockIdx.z;
	
	if(tx < sizex)
	{
		out[ty*sizex + tx].x = in[ty*sizex + tx];
		out[ty*sizex + tx].y = 0;
	}
}

__global__ void GetX(float* out, complex* in, int sizex)
{
	size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
	size_t ty = blockIdx.y + gridDim.y * blockIdx.z;
	
	if(tx < sizex)
		out[ty*sizex + tx] = in[ty*sizex + tx].x;
}

__global__ void sino2p(complex* padded, float* in, size_t nrays, size_t nangles, int pad0, int csino)
{
	int center = nrays/2 - csino;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(idx < nrays/2)
	{
		size_t fory = blockIdx.y;
		size_t revy = blockIdx.y + nangles;
		
		size_t slicez = blockIdx.z * nrays * nangles;
		
		//float Arg2 = (2.0f*idx - pad0*nrays/2 + 1.0f)/(pad0*nrays/2 - 1.0f);
		//double b1 = cyl_bessel_i0f(sqrtf(fmaxf(1.0f - Arg2 * Arg2,0.0f)));
		//double b2 = cyl_bessel_i0f(1.0f);
		float w_bessel = 1;//fabsf(b1/b2);
		if(idx==0)
			w_bessel *= 0.5f;

		w_bessel *= sq(pad0);
		
		if(center - 1 - idx >= 0)
			padded[pad0*slicez + pad0*fory*nrays/2 + idx] = w_bessel * in[slicez + fory*nrays + center - 1 - idx];
		else
			padded[pad0*slicez + pad0*fory*nrays/2 + idx] = w_bessel * in[slicez + fory*nrays];
			
		if(center + 0 + idx >= 0)
			padded[pad0*slicez + pad0*revy*nrays/2 + idx] = w_bessel * in[slicez + fory*nrays + center + 0 + idx];
		else
			padded[pad0*slicez + pad0*revy*nrays/2 + idx] = w_bessel * in[slicez + fory*nrays];
	}
}

__global__ void convBST(complex* block, size_t nrays, size_t nangles)
{
	size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
 	size_t ty = blockIdx.y;
	size_t tz = blockIdx.z;
 	
	float sigma = 2.0f / (nrays * (fmaxf(fminf(tx,nrays-tx),0.5f)));
	size_t offset = tz*nangles*nrays + ty*nrays + tx;
	
	if(tx < nrays)
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

extern "C" {
	void BST(float* blockRecon, float *wholesinoblock, int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0)
	{
		int blocksize = 1;

		cImage cartesianblock(sizeimage, sizeimage*blocksize);
		cImage polarblock(Nrays * pad0,Nangles*blocksize);
		cImage realpolar(Nrays * pad0,Nangles*blocksize);
		
		cufftHandle plan1d;
		cufftHandle plan2d;
	
		int dimms1d[] = {(int)Nrays*pad0/2};
		int dimms2d[] = {(int)sizeimage,(int)sizeimage};
		int beds[] = {Nrays*pad0/2};
	
		HANDLE_FFTERROR( cufftPlanMany(&plan1d, 1, dimms1d, beds, 1, Nrays*pad0/2, beds, 1, Nrays*pad0/2, CUFFT_C2C, Nangles*blocksize*2) );
		HANDLE_FFTERROR( cufftPlanMany(&plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize) );
		
		size_t insize = Nrays*Nangles;
		size_t outsize = sizeimage*sizeimage;
		
		for(size_t zoff = 0; zoff < (size_t)trueblocksize; zoff+=blocksize)
		{
			float* sinoblock = wholesinoblock + insize*zoff;
			
			dim3 blocks((Nrays+255)/256,Nangles,blocksize);
			dim3 threads(128,1,1); 
			
			sino2p<<<blocks,threads>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, pad0, 0);
			
			Nangles *= 2;
			Nrays *= pad0;
			Nrays /= 2;

			blocks.y *= 2;
			blocks.x *= pad0;
			blocks.x /= 2;

			HANDLE_FFTERROR(cufftExecC2C(plan1d, realpolar.gpuptr, polarblock.gpuptr, CUFFT_FORWARD));
			convBST<<<blocks,threads>>>(polarblock.gpuptr, Nrays, Nangles);
			
			blocks = dim3((sizeimage+255)/256,sizeimage,blocksize);
			threads = dim3(256,1,1);

			polar2cartesian_fourier<<<blocks,threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, Nrays, Nangles, sizeimage);
		
			HANDLE_FFTERROR( cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE) );
		
			cudaDeviceSynchronize();

			GetX<<<dim3((sizeimage+127)/128,sizeimage),128>>>(blockRecon + outsize*zoff, cartesianblock.gpuptr, sizeimage);

			HANDLE_ERROR( cudaPeekAtLastError() );
			
			Nangles /= 2;
			Nrays *= 2;
			Nrays /= pad0;
		}
		cufftDestroy(plan1d);
		cufftDestroy(plan2d);
	}
}
// extern "C"
// {
	
// void GPUBST(int devv, float* blockRecon, float* sinoblock, int nrays, int nangles, int isizez, int padding)
// {	
// 	cudaSetDevice(devv);
// 	size_t sizez = size_t(isizez);
// 	size_t sizeimage = size_t(nrays);
	
// 	rImage sino(nrays, nangles, min(sizez,32ul), MemoryType::EAllocGPU);
// 	rImage recon(sizeimage,sizeimage, min(sizez,32ul), MemoryType::EAllocGPU);

// 	for(size_t b=0; b < sizez; b += 32)
// 	{
// 		size_t blocksize = min(sizez-b,32ul);
// 		size_t reconoffset = b*sizeimage*sizeimage;

// 		sino.CopyFrom(sinoblock + b*nrays*nangles, 0, blocksize*nrays*nangles);

// 		BST(recon.gpuptr, sino.gpuptr, nrays, nangles, blocksize, sizeimage, padding);
// 		recon.CopyTo(reconoffset + blockRecon, 0, blocksize*sizeimage*sizeimage);
// 	}
// 	cudaDeviceSynchronize();
// }
// }
