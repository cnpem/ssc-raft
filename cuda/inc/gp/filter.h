// Authors: Giovanni Baraldi, Gilberto Martinez

#ifndef RAFT_FILTER_H
#define RAFT_FILTER_H

#include <string.h>

#include "common/types.hpp"
#include "common/operations.hpp"
#include "common/complex.hpp"

struct CFilter
{
	CFilter() = default;
	explicit CFilter(int _type, float _reg): type((EType)_type), reg(_reg) {} 

	enum EType
	{
		none=0,
		gaussian=1,
		lorentz=2,
		cosine=3,
		rectangle=4,
		FSC
	};

	float reg = 0;
	EType type = EType::none;

	__host__ __device__ inline 
	float Apply(float input)
	{
		if(type == EType::gaussian)
			input *= exp(-0.693f*reg*input*input );

		else if(type == EType::lorentz)
			input /= 1.0f + reg*input*input;

		else if(type == EType::cosine)
			input *= cosf(	float(M_PI)*0.5f*input	);

		else if(type == EType::rectangle){
			float param = fmaxf(input * reg * float(M_PI) * 0.5f, 1E-4f);
			input *= sinf(param) / param;
		}

		return input;
	}
};

extern "C"{
	void SinoFilter(float* sino, size_t nrays, size_t nangles, size_t blocksize, int csino, bool bRampFilter, struct CFilter reg, bool bShiftCenter, float* sintable);
	
	void Highpass(rImage& x, float wid);

	__global__ void BandFilterReg(complex* vec, size_t sizex, int icenter, bool bShiftCenter, float* sintable, struct CFilter mfilter);

	__global__ void KFilter(complex* x, size_t sizex, float wid);

	__device__ complex DeltaFilter(complex* img, int sizeimage, float fx, float fy);

	inline __global__ void SetX(complex* out, float* in, int sizex);

	inline __global__ void GetX(float* out, complex* in, int sizex);

	inline __global__ void GetXBST(void* out, complex* in, size_t sizex, float threshold, EType::TypeEnum raftDataType, int rollxy);
	
	inline __global__ void BandFilterC2C(complex* vec, size_t sizex, int center, struct CFilter mfilter);
	
	void BSTFilter(cufftHandle plan, complex* filtersino, float* sinoblock, size_t nrays, size_t nangles, int csino, struct CFilter reg);

}

#endif
