// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi

#include "../../inc/include.h"

#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"

inline __device__ float SoftThresh(float v, float mu)
{
    return fmaxf(fminf(v,mu),-mu);
}

inline __device__ complex SoftThresh(complex v, float mu)
{
    return complex(SoftThresh(v.x,mu),SoftThresh(v.y,mu));
}

extern "C" {
__global__ void KTV2D(float* imgout, const float* imgin, const float* f0, float mu, dim3 shape)
{
	const size_t N = shape.x;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int idy = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t index = blockIdx.z*N*N + idy*N + idx;

    if(idx < N && idy < N)
    {
        float center = imgin[index];

		const float* imgoff = imgin + blockIdx.z*N*N;
		float coef1 = SoftThresh(imgoff[N*idy + (idx+1)%N]-center, mu);
		float coef2 = SoftThresh(imgoff[N*((idy+1)%N) + idx]-center, mu);
		float coef3 = SoftThresh(imgoff[N*idy + (idx+N-1)%N]-center, mu);
		float coef4 = SoftThresh(imgoff[N*((idy+N-1)%N) + idx]-center, mu);

		center += (coef1+coef2+coef3+coef4)*0.35f;

		if(f0 != nullptr)
			imgout[index] = center + (f0[index] - imgin[index]);
		else
			imgout[index] = center;
	}
}

__global__ void KTV3D(float* imgout, const float* imgin, const float* f0, float mu, dim3 shape)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if(idx < shape.x && idy < shape.y)
    {
		const size_t N = shape.x;
		const int sizexy = shape.x*shape.y;

		const float* imgoff = imgin + blockIdx.z*sizexy;

        float center = imgoff[idy*N + idx];

		float coef1 = SoftThresh(imgoff[N*idy + (idx+1)%N]-center, mu);
		float coef2 = SoftThresh(imgoff[N*((idy+1)%N) + idx]-center, mu);
		float coef3 = SoftThresh(imgoff[N*idy + (idx+N-1)%N]-center, mu);
		float coef4 = SoftThresh(imgoff[N*((idy+N-1)%N) + idx]-center, mu);

		center += (coef1+coef2+coef3+coef4)*0.35f;

		if(blockIdx.z < shape.z-1)
			center += SoftThresh(imgoff[N*idy + idx + sizexy]-center, mu)*0.35f;
		if(blockIdx.z > 0)
			center += SoftThresh(imgoff[N*idy + idx - sizexy]-center, mu)*0.35f;

		if(f0 != nullptr)
			imgout[blockIdx.z*sizexy + idy*N + idx] = center + (f0[blockIdx.z*sizexy + idy*N + idx] - imgin[blockIdx.z*sizexy + idy*N + idx]);
		else
			imgout[blockIdx.z*sizexy + idy*N + idx] = center;
	}
}

void TotalVariation(rImage& img, rImage& temp, float* f0, float mu, int numiter)
{
	dim3 blk = img.ShapeBlock();
	dim3 thr = img.ShapeThread();

	auto KTV = (blk.z > 1) ? KTV3D : KTV2D;

	if(f0 != nullptr)
		img.CopyTo(f0);

	for(int i=0; i<numiter/2; i++)
	{
		KTV<<<blk,thr>>>(temp.gpuptr, img.gpuptr, f0, mu, img.Shape());
		KTV<<<blk,thr>>>(img.gpuptr, temp.gpuptr, f0, mu, img.Shape());
	}
	if(numiter%2==1)
	{
		KTV<<<blk,thr>>>(temp.gpuptr, img.gpuptr, f0, mu, img.Shape());
		img.CopyFrom(temp);
	}
}

__global__ void KTVMulti(float* imgout, const float* imgin, const float* imgin_prev, 
			const float* imgin_next, const float* f0, float mu, dim3 shape, bool bWriteback)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if(idx < shape.x && idy < shape.y)
    {
		const size_t N = shape.x;
		const int sizexy = shape.x*shape.y;

		const float* imgoff = imgin + blockIdx.z*sizexy;

        float center = imgoff[idy*N + idx];

		float coef1 = SoftThresh(imgoff[N*idy + (idx+1)%N]-center, mu);
		float coef2 = SoftThresh(imgoff[N*((idy+1)%N) + idx]-center, mu);
		float coef3 = SoftThresh(imgoff[N*idy + (idx+N-1)%N]-center, mu);
		float coef4 = SoftThresh(imgoff[N*((idy+N-1)%N) + idx]-center, mu);

		center += (coef1+coef2+coef3+coef4)*0.25f;

		if(blockIdx.z < shape.z-1)
			center += SoftThresh(imgoff[N*idy + idx + sizexy]-center, mu)*0.25f;
		else if(imgin_next != nullptr)
			center += SoftThresh(imgin_next[N*idy + idx]-center, mu)*0.25f;

		if(blockIdx.z > 0)
			center += SoftThresh(imgoff[N*idy + idx - sizexy]-center, mu)*0.25f;
		else if(imgin_prev != nullptr)
			center += SoftThresh(imgin_prev[N*idy + idx]-center, mu)*0.25f;

		if(f0 != nullptr)
			imgout[blockIdx.z*sizexy + idy*N + idx] = center + (f0[blockIdx.z*sizexy + idy*N + idx] - imgin[blockIdx.z*sizexy + idy*N + idx]);
		else if(bWriteback)
			imgout[blockIdx.z*sizexy + idy*N + idx] = center + (imgout[blockIdx.z*sizexy + idy*N + idx] - imgin[blockIdx.z*sizexy + idy*N + idx]);
		else
			imgout[blockIdx.z*sizexy + idy*N + idx] = center;
	}
}


void GPU_TotalVariation(rMImage& img, rMImage& temp, rMImage& f0, float mu, int numiter)
{
	dim3 blk = img[0].ShapeBlock();
	dim3 thr = img[0].ShapeThread();

	img.CopyTo(f0);

	for(int i=0; i<numiter/2; i++)
	{
		img.SyncDevices();

		for(int g=0; g<img.gpuindices.size(); g++)
		{
			img.Set(g);
			float* prev = (g>0) ? img.Ptr(g-1) : nullptr;
			float* next = (g<img.gpuindices.size()-1) ? img.Ptr(g+1) : nullptr;

			KTVMulti<<<blk,thr>>>(temp.Ptr(g), img.Ptr(g), prev, next, f0.Ptr(g), mu, img[g].Shape(), false);
		}
		
		img.SyncDevices();

		for(int g=0; g<img.gpuindices.size(); g++)
		{
			img.Set(g);
			float* prev = (g>0) ? temp.Ptr(g-1) : nullptr;
			float* next = (g<temp.gpuindices.size()-1) ? temp.Ptr(g+1) : nullptr;

			KTVMulti<<<blk,thr>>>(img.Ptr(g), temp.Ptr(g), prev, next, f0.Ptr(g), mu, img[g].Shape(), false);
		}
	}
	if(numiter%2==1)
	{
		img.SyncDevices();

		for(int g=0; g<img.gpuindices.size(); g++)
		{
			img.Set(g);
			float* prev = (g>0) ? img.Ptr(g-1) : nullptr;
			float* next = (g<img.gpuindices.size()-1) ? img.Ptr(g+1) : nullptr;

			KTVMulti<<<blk,thr>>>(temp.Ptr(g), img.Ptr(g), prev, next, f0.Ptr(g), mu, img[g].Shape(), false);
		}

		img.CopyFrom(temp);
	}
	img.SyncDevices();
}

__global__ void KTV3DShared(float* imgout, const float* imgin, const float* imgin_prev, 
	const float* imgin_next, float mu, dim3 shape, size_t prev_sizez)
{
	const int grpinner = 12; // assume bdim.x = bdim.y = grpinner1
	const int grpouter = grpinner + 2;

	__shared__ float fn[10][grpouter][16];
	__shared__ float f0[10][grpouter][16]; // pad to 16 to ensure cont. bank access

	for(size_t idz = threadIdx.z; idz < 10; idz += blockDim.z)
	for(size_t idy = threadIdx.y; idy < grpouter; idy += blockDim.y)
	for(size_t idx = threadIdx.x; idx < grpouter; idx += blockDim.x)
	{
		size_t posz = grpinner/2*blockIdx.z + idz;
		size_t posy = grpinner*blockIdx.y + idy;
		size_t posx = grpinner*blockIdx.x + idx;

		posx = (posx+shape.x-1+shape.x/2)%shape.x;
		posy = (posy+shape.y-1+shape.y/2)%shape.y;
		
		if(posz >= 1 && posz < shape.z + 1)
		{
			posz -= 1;
			fn[idz][idy][idx] = f0[idz][idy][idx] = imgin[(posz*shape.y + posy)*shape.x + posx];
		}
		else if(posz >= shape.z + 1)
		{
			if(imgin_next != nullptr)
			{
				posz -= shape.z + 1;
				fn[idz][idy][idx] = f0[idz][idy][idx] = imgin_next[(posz*shape.y + posy)*shape.x + posx];
			}
			else
			{
				posz = shape.z - 1;
				fn[idz][idy][idx] = f0[idz][idy][idx] = imgin[(posz*shape.y + posy)*shape.x + posx];
			}

		}
		else
		{
			if(imgin_prev != nullptr)
			{
				posz = posz + prev_sizez - 1;
				fn[idz][idy][idx] = f0[idz][idy][idx] = imgin_prev[(posz*shape.y + posy)*shape.x + posx];
			}
			else
			{
				posz = 0;
				fn[idz][idy][idx] = f0[idz][idy][idx] = imgin[(posz*shape.y + posy)*shape.x + posx];
			}
		}
	}

	size_t idx = threadIdx.x + 1;
	size_t idy = threadIdx.y + 1;

	for(int iter=0; iter<5; iter++)
	{
		__syncthreads();

		for(size_t idz = threadIdx.z+1; idz < 9; idz += blockDim.z)
		{
			float center = fn[idz][idy][idx];

			float coef1 = SoftThresh(fn[idz][idy][idx-1]-center, mu);
			float coef2 = SoftThresh(fn[idz][idy][idx+1]-center, mu);
			float coef3 = SoftThresh(fn[idz][idy-1][idx]-center, mu);
			float coef4 = SoftThresh(fn[idz][idy+1][idx]-center, mu);
			float coef5 = SoftThresh(fn[idz-1][idy][idx]-center, mu);
			float coef6 = SoftThresh(fn[idz+1][idy][idx]-center, mu);

			fn[idz][idy][idx] = 2*center - f0[idz][idy][idx] + (coef1+coef2+coef3+coef4+coef5+coef6)*0.35f;
		}
	}
	__syncthreads();
	
	size_t posx = grpinner*blockIdx.x + threadIdx.x;
	size_t posy = grpinner*blockIdx.y + threadIdx.y;

	if(posy < shape.y && posx < shape.x)
	{
		posx = (posx+shape.x/2)%shape.x;
		posy = (posy+shape.y/2)%shape.y;

		for(size_t idz = threadIdx.z; idz < 8; idz += blockDim.z)
		{
			size_t posz = grpinner/2*blockIdx.z + idz;
			
			if(posz < shape.z)
				imgout[(posz*shape.y + posy)*shape.x + posx] = fn[idz+1][idy][idx];
		}
	}
}

void GPU_TotalVariation_2iter(rMImage& img, rMImage& temp, float mu, rImage* gpu0prev, rImage* gpuneg1next)
{
	img.SyncDevices();

	for(int g=0; g<img.gpuindices.size(); g++)
	{
		dim3 blk = img[g].ShapeBlock();
		dim3 thr = img[g].ShapeThread();

		img.Set(g);
		float* prev = (g>0) ? (temp.Ptr(g-1) + temp.sizex*temp.sizey*(temp[g].sizez-1)) : (gpu0prev?gpu0prev->gpuptr:nullptr);
		float* next = (g<img.gpuindices.size()-1) ? img.Ptr(g+1) : (gpuneg1next?gpuneg1next->gpuptr:nullptr);

		KTVMulti<<<blk,thr>>>(temp.Ptr(g), img.Ptr(g), prev, next, nullptr, mu, img[g].Shape(), false);
		//KTV3DShared<<<dim3((img.sizex+11)/12,(img.sizey+11)/12,img[g].sizez/8),dim3(12,12,2)>>>(temp.Ptr(g), img.Ptr(g), prev, next, mu, img[g].Shape(), 1);
	}
	
	img.SyncDevices();

	for(int g=0; g<img.gpuindices.size(); g++)
	{
		dim3 blk = img[g].ShapeBlock();
		dim3 thr = img[g].ShapeThread();

		img.Set(g);
		float* prev = (g>0) ? (temp.Ptr(g-1) + temp.sizex*temp.sizey*(temp[g].sizez-1)) : (gpu0prev?gpu0prev->gpuptr:nullptr);
		float* next = (g<temp.gpuindices.size()-1) ? temp.Ptr(g+1) : (gpuneg1next?gpuneg1next->gpuptr:nullptr);

		KTVMulti<<<blk,thr>>>(img.Ptr(g), temp.Ptr(g), prev, next, nullptr, mu, img[g].Shape(), true);
		//KTV3DShared<<<dim3((img.sizex+11)/12,(img.sizey+11)/12,img[g].sizez/8),dim3(12,12,2)>>>(img.Ptr(g), temp.Ptr(g), prev, next, mu, img[g].Shape(), 1);
	}
	
	img.SyncDevices();
}


void TotalVariation_2iter(rImage& img, rImage& temp, float mu)
{
	dim3 blk = img.ShapeBlock();
	dim3 thr = img.ShapeThread();

	KTVMulti<<<blk,thr>>>(temp.gpuptr, img.gpuptr, nullptr, nullptr, nullptr, mu, img.Shape(), false);
	KTVMulti<<<blk,thr>>>(img.gpuptr, temp.gpuptr, nullptr, nullptr, nullptr, mu, img.Shape(), true);
}

}