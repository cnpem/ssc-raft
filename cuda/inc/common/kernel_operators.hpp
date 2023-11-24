#ifndef _TOMOOPR_H
#define _TOMOOPR_H

#include "types.hpp"
#include "logerror.hpp"

// Auto create 4 real <-> complex transforms sharing the same workarea
// Meant to be used with the FST versions of Radon and BackProjection
class PlanRC
{
public:
    PlanRC(size_t nrays, size_t nangles, size_t batch=1);
    ~PlanRC();

    const size_t sizex;
    const size_t nangles;
    const size_t batch;
    size_t maxworksize;

    void* WorkArea;

    cufftHandle plan1d_r2c;
	cufftHandle plan1d_c2r;
	cufftHandle plan2d_r2c;
	cufftHandle plan2d_c2r;
};

// FFT SHIFT -> Crop/Place -> FFT ISHIFT
// Warning: Fills empty space with non-zeros
// -> Blame EM
template<typename Type>
__global__ void KCopyShiftX(Type* out, Type* in, size_t outsizex, size_t insizex, size_t nangles, int csino, float filler)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y;
	
	size_t index = idy * insizex + (idx+insizex-csino)%insizex;
	size_t shift = idy * outsizex + (idx + outsizex - insizex/2) % outsizex;

	if(idx < insizex)
		out[shift] = in[index];
	else if(idx < outsizex)
		out[shift] = filler;
}

template<typename Type>
__global__ void KSetToOne(Type* vec, size_t size)
{
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < size)
		vec[idx] = Type(1);
}



extern "C"{
__global__ void KBackProjection_RT(char *recon,
    const float *sino,
    int wdI,
    int nrays,
    int nangles,
    EType::TypeEnum datatype, float threshold, const float* sintable, const float* costable);

__global__ void KRadon_RT(float* restrict frames, const float* image, int nrays, int nangles);

// __device__ void filteredAtomicAdd(complex* img, complex value, int sizeimage, float fx, float fy, float mult);

__global__ void KRadon_FST(complex* sino, complex* img, int sizeimage, int nangles);
void Radon_FST(float* sinogram, float* image, complex* Temp, PlanRC& plan, size_t sizeimage, size_t nangles, size_t sizez = 1);
void GPU_Radon_FST(rMImage& sinogram, rMImage& image, cMImage& Temp, PlanRC** plans);

__global__ void KBackProjection_FST(complex* img, complex* sino, int sizeimage, int nangles);
void BackProjection_FST(float* backptr, float* sinogram, complex* Temp, PlanRC& plan, size_t sizeimage, size_t nangles, size_t sizez = 1);
void GPU_BackProjection_FST(rMImage& backptr, rMImage& sinogram, cMImage& Temp, PlanRC** plans);

// __global__ void KShiftSwapAndFlip(float* image, size_t sizex, size_t sizey);

// __global__ void KTotalVariation(float* imgout, const float* imgin, float mu, size_t N);

// void TotalVariation(rImage& img, rImage& temp, float* f0, float mu, int numiter);
// inline void TotalVariation(rImage& img, rImage& temp, float mu, int numiter){ TotalVariation(img, temp, nullptr, mu, numiter); };

// void TotalVariation(rMImage& img, rMImage& temp, rMImage& f0, float mu, int numiter);

// void TotalVariation_2iter(rImage& img, rImage& temp, float mu);
// void TotalVariation_2iter(rMImage& img, rMImage& temp, float mu, rImage* prevconn = nullptr, rImage* nextconn = nullptr);

__global__ void KSupport(float* recon, dim3 shape, float reg, float radius);
__global__ void UpdateABC(float* restrict A, float* restrict B, float* restrict C, const float* restrict FDD, const float* restrict sino, dim3 shape, float stepb);
__global__ void KComputeRes(float* FDD, const float* restrict A, const float* restrict B, const float* restrict C, const float* sino, dim3 shape, float alpha);

__global__ void KDivide_transmission(GArray<float> ffk, const GArray<float> BP, GArray<float> BI);
__global__ void KExponential(GArray<float> ffk, GArray<float> flat);



void EnablePeerToPeer(const std::vector<int>& gpus);
void DisablePeerToPeer(const std::vector<int>& gpus);
}

#endif