// Authors: Giovanni Baraldi, Gilberto Martinez

#ifndef RAFT_FILTER_H
#define RAFT_FILTER_H

#include <driver_types.h>
#include "common/configs.hpp"
#include "common/types.hpp"
#include "common/operations.hpp"
#include "common/complex.hpp"
#include "common/logerror.hpp"

enum PhaseFilterType
{
    paganin        = 0,
    paganin_tomopy = 1,
    paganin_v0     = 2,
    paganin_v1     = 3
};

struct Filter{

	Filter() = default;
	explicit Filter(int _type, float _paganin, float _reg, float _axis_offset, float _pixel): type((EType)_type), paganin(_paganin), reg(_reg), axis_offset(_axis_offset), pixel(_pixel) {}; 

	enum EType
	{
		none         = 0,
		gaussian     = 1,
		lorentz      = 2,
		cosine       = 3,
		rectangle    = 4,
		hann         = 5,
		hamming      = 6,
		ramp         = 7,
        differential = 8
	};

	EType type        = EType::none;

	float reg         = 0.0f;
	float paganin     = 0.0f;
	float axis_offset = 0.0f;
    float pixel       = 1.0f;

	__host__ __device__ inline float apply(float input);
};


extern "C"{
	void SinoFilter(float* sino, size_t nrays, size_t nangles, size_t blocksize, int csino, bool bRampFilter, struct Filter reg, bool bShiftCenter, float* sintable, float pixel);
	
	void Highpass(rImage& x, float wid);

	__global__ void BandFilterReg(complex* vec, size_t sizex, int icenter, bool bShiftCenter, float* sintable, struct Filter mfilter, float pixel);

	__global__ void KFilter(complex* x, size_t sizex, float wid);

	__device__ complex DeltaFilter(complex* img, int sizeimage, float fx, float fy);

	__global__ void SetX(complex* out, float* in, int sizex);

	__global__ void GetX(float* out, complex* in, int sizex, float scale);

	__global__ void GetXBST(void* out, complex* in, size_t sizex, float threshold, EType::TypeEnum raftDataType, int rollxy);
	
	__global__ void BandFilterC2C(complex* vec, size_t sizex, int center, float pixel, struct Filter mfilter);
	
	void BSTFilter(cufftHandle plan, complex* filtersino, float* sinoblock, size_t nrays, size_t nangles, int csino, struct Filter reg, float pixel, cudaStream_t stream = 0);

	void filterFBPpad(GPU gpus, Filter filter, 
    float *tomogram, dim3 size, dim3 size_pad, dim3 pad);

    void filterFBP(GPU gpus, Filter filter, float *tomogram, dim3 size);

    void filterFBP_Complex(GPU gpus, Filter filter, 
    float *tomogram, dim3 size, dim3 size_pad, dim3 pad, float pixel);

    void getFilterLowPassMultiGPU(int* gpus, int ngpus, 
    float* tomogram, float *paramf, int *parami);
    

}

#endif
