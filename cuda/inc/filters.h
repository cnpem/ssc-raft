// Authors: Giovanni Baraldi, Gilberto Martinez

#ifndef RAFT_FILTER_H
#define RAFT_FILTER_H

#include "./common/configs.h"
#include "./common/types.hpp"
#include "./common/operations.hpp"
#include "./common/complex.hpp"
#include "./common/logerror.hpp"

struct Filter{

	Filter() = default;
	explicit Filter(int _type, float _paganin, float _reg, int _axis_offset): type((EType)_type), paganin(_paganin), reg(_reg), axis_offset(_axis_offset) {}; 

	enum EType
	{
		none      = 0,
		gaussian  = 1,
		lorentz   = 2,
		cosine    = 3,
		rectangle = 4,
		hann      = 5,
		hamming   = 6,
		ramp      = 7
	};

	EType type      = EType::none;

	float reg       = 1.0f;
	float paganin   = 0.0f;
	int axis_offset = 0.0f;
	
	__host__ __device__ inline float apply(float input);
	// {
	// 	float param = 0.0f;

	// 	if(type == EType::gaussian){
	// 		input *= exp( -0.693f * reg * input * input );
	// 		input /= ( 1.0f + paganin * input * input );
	// 	}
	// 	else if(type == EType::lorentz){
	// 		input /= 1.0f + reg * input * input;
	// 		input /= ( 1.0f + paganin * input * input );
	// 	}
	// 	else if(type == EType::cosine){
	// 		input *= cosf(	float(M_PI) * 0.5f * input );
	// 		input /= ( 1.0f + paganin * input * input );
	// 	}
	// 	else if(type == EType::rectangle){
	// 		param  = fmaxf(input * reg * float(M_PI) * 0.5f, 1E-4f);
	// 		input *= sinf( param ) / param;
	// 		input /= ( 1.0f + paganin * input * input );
	// 	}
	// 	else if(type == EType::hann){
	// 		input *= input * 0.5f + 0.5f * cosf( 2.0f * float(M_PI) * input );
	// 		input /= ( 1.0f + paganin * input * input );
	// 	}
	// 	else if(type == EType::hamming){
	// 		input *= input * ( 0.54f + 0.46f * cosf( 2.0f * float(M_PI) * input ) );
	// 		input /= ( 1.0f + paganin * input * input );
	// 	}
	// 	else if(type == EType::ramp){
	// 		input /= ( 1.0f + paganin * input * input );
	// 	}

	// 	return input;
	// }
};









extern "C"{
	void SinoFilter(float* sino, size_t nrays, size_t nangles, size_t blocksize, int csino, bool bRampFilter, struct Filter reg, bool bShiftCenter, float* sintable);
	
	void Highpass(rImage& x, float wid);

	__global__ void BandFilterReg(complex* vec, size_t sizex, int icenter, bool bShiftCenter, float* sintable, struct Filter mfilter);

	__global__ void KFilter(complex* x, size_t sizex, float wid);

	__device__ complex DeltaFilter(complex* img, int sizeimage, float fx, float fy);

	// inline __global__ void SetX(complex* out, float* in, int sizex);

	// inline __global__ void GetX(float* out, complex* in, int sizex);

	// inline __global__ void GetXBST(void* out, complex* in, size_t sizex, float threshold, EType::TypeEnum raftDataType, int rollxy);
	
	__global__ void BandFilterC2C(complex* vec, size_t sizex, int center, struct Filter mfilter);
	
	// void BSTFilter(cufftHandle plan, complex* filtersino, float* sinoblock, size_t nrays, size_t nangles, int csino, struct Filter reg);

	void filterFBP(GPU gpus, Filter filter, float *tomogram, cufftComplex *filter_kernel, dim3 size, dim3 size_pad);
	void convolution_mplan2DR2R(GPU gpus, float *data, float *kernel, float pad_value, dim3 size, dim3 size_pad);
	void convolution_mplan1DR2R(GPU gpus, float *data, cufftComplex *kernel, float pad_value, dim3 size, dim3 size_pad);

	__global__ void _fbp_filter(Filter filter, cufftComplex *kernel, dim3 size);


	__global__ void ProdComplexFloat(cufftComplex *a, float *b, cufftComplex *ans, dim3 size);
	__global__ void ProdComplexComplex(cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 size);
    __global__ void fftNormalize1d(cufftComplex *c, dim3 size);
    __global__ void fftNormalize2d(cufftComplex *c, dim3 size);
    __global__ void fftshiftKernel(float *c, dim3 size);
    __global__ void Normalize(float *a, float b, dim3 size);

}

extern "C"{

	/* Phase Filters Functions */
	void getPhaseFilterMultiGPU(float *projections, float *paramf, int *parami, 
	int phase_type, float phase_reg, int *gpus, int ngpus);

	void getPhaseFilterGPU(GPU gpus, GEO geometry, DIM tomo,  
	float *projections, int phase_type, float phase_reg, int ngpu);

	void setPhaseFilterKernel(GPU gpus, GEO geometry, float *kernel, 
	dim3 size_pad, int phase_type, float phase_reg);

	void getPhaseFilter(GPU gpus, GEO geometry, float *projections, 
	int phase_type, int phase_reg, dim3 size, dim3 size_pad);

	void applyPhaseFilter(GPU gpus, float *projections, float *kernel, 
	int phase_type, dim3 tomo, dim3 tomo_pad);

	void _paganin_gpu(GPU gpus, float *projections, float *kernel, dim3 size, dim3 size_pad);
 	__global__ void paganinKernel(GEO geometry, float *kernel, dim3 size, int phase_reg);

	void _born_gpu(GPU gpus, float *projections, float *kernel, dim3 size, dim3 size_pad);
 	__global__ void bornKernel(GEO geometry, float *kernel, dim3 size, int phase_reg);
    __global__ void bornPrep(float *data, dim3 size);

	void _rytov_gpu(GPU gpus, float *projections, float *kernel, dim3 size, dim3 size_pad);
 	__global__ void rytovKernel(GEO geometry, float *kernel, dim3 size, int phase_reg);
    __global__ void rytovPrep(float *data, dim3 size);

	void _bronnikov_gpu(GPU gpus, float *projections, float *kernel, dim3 size, dim3 size_pad);
 	__global__ void bronnikovKernel(GEO geometry, float *kernel, dim3 size, int phase_reg);
    __global__ void bronnikovPrep(float *data, dim3 size);

}

#endif
