/**
@file operations.hpp
@author Giovanni L. Baraldi, Gilberto Martinez
@brief File contains kernels for the most basic operations
@version 0.1
@date 2021-06-14

@copyright Copyright (c) 2021

*/

#ifndef RAFT_OPERATIONS_H
#define RAFT_OPERATIONS_H

#ifdef __CUDACC__
	#define restrict __restrict__
#else
	#define restrict
#endif

#include "complex.hpp"
#include "logerror.hpp"
#include "configs.h"

extern "C"{

    __global__ void padding(float *in, cufftComplex *inpadded, float value, dim3 size, dim3 padsize);

    __global__ void recuperate_padding(cufftComplex *inpadded, float *in, dim3 size, dim3 padsize);

}

inline __device__ int iabs(int n){return n<0?-n:n;}

static __device__ __host__ float clamp(float f, float a, float b){return fmaxf(a, fminf(f, b));}

static __device__ half clamp(half f, half a, half b){return fmaxf(a, fminf(f, b));}

static __device__ __host__ half clamp(half f, float a=-32768.0f, float b=32768.0f){ return fmaxf(__float2half(a), fminf(f, __float2half(b)));}

#if __CUDA_ARCH__ >= 530
static __device__ complex16 clamp(complex16 f, float a, float b){return complex16(clamp(f.x,a,b),clamp(f.y,a,b));} 
#endif

static __device__ __host__ complex clamp(const complex& f, float a, float b)
{
	return complex(clamp(f.x,a,b),clamp(f.y,a,b));
}

static __device__ __host__ complex clamp(const complex& f, const complex& a, const complex& b)
{
	return complex(clamp(f.x,a.x,b.x),clamp(f.y,a.y,b.y));
}

inline __device__ complex exp1j(float f){complex c;__sincosf(f, &c.y, &c.x); return c;}

template<typename Type>
inline __device__ __host__ Type sq(const Type& x){ return x*x; }

struct EType{

	enum class TypeEnum 
	{
		INVALID = 0,
		UINT8   = 1,
		UINT16  = 2,
		INT32   = 3,
		HALF    = 4,
		FLOAT32 = 5,
		DOUBLE  = 6,
		NUM_ENUMS
	};

	EType() = default;
	EType(TypeEnum etype): type(etype) {};

	static size_t Size(TypeEnum datatype);

	size_t Size() const { return Size(type); };

	static std::string String(TypeEnum type);

	std::string String() { return String(type); };

	static EType Type(const std::string& nametype);

	TypeEnum type = TypeEnum::INVALID;
};

namespace BasicOps
{
	inline __device__ size_t GetIndex(){ return threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y*blockIdx.z)); }

	template<typename Type1, typename Type2>
	__global__ void KB_Add(Type1* a, const Type2* b, size_t size, size_t size2);

	template<typename Type1, typename Type2>
	__global__ void KB_Sub(Type1* a, const Type2* b, size_t size, size_t size2);

	template<typename Type1, typename Type2>
	__global__ void KB_Mul(Type1* a, const Type2* b, size_t size, size_t size2);

	template<typename Type1, typename Type2>
	__global__ void KB_Div(Type1* a, const Type2* b, size_t size, size_t size2);

	template<typename Type1, typename Type2=Type1>
	__global__ void KB_Add(Type1* a, Type2 n, size_t size);

	template<typename Type1, typename Type2=Type1>
	__global__ void KB_Sub(Type1* a, Type2 n, size_t size);

	template<typename Type1, typename Type2=Type1>
	__global__ void KB_Mul(Type1* a, Type2 n, size_t size);

	template<typename Type1, typename Type2=Type1>
	__global__ void KB_Div(Type1* a, Type2 n, size_t size);

	template<typename Type>
	__global__ void KB_Log(Type* a, size_t size);

	template<typename Type>
	__global__ void KB_Exp(Type* a, size_t size, bool bNeg);

	template<typename Type>
	__global__ void KB_Clamp(Type* a, const Type b, const Type c, size_t size);

	template<typename Type>
	__global__ void KB_log1j(float* out, const Type* in, size_t size);// {}
	
	template<typename Type>
	__global__ void KB_exp1j(Type* out, const float* in, size_t size);// {}

	template<>
	__global__ void KB_log1j<complex>(float* out, const complex* in, size_t size);
	
	template<>
	__global__ void KB_exp1j<complex>(complex* out, const float* in, size_t size);

	template<typename Type>
	__global__ void KB_Power(Type* a, float P, size_t size);// {}

	template<>
	__global__ void KB_Power<float>(float* out, float P, size_t size);

	template<>
	__global__ void KB_Power<complex>(complex* out, float P, size_t size);

	template<typename Type>
	__global__ void KB_ABS2(float* out, Type* a, size_t size);// {}

	template<>
	__global__ void KB_ABS2<complex>(float* out, complex* in, size_t size);

	template<>
	__global__ void KB_ABS2<float>(float* out, float* in, size_t size);

	#ifdef _USING_FP16
	template<typename Type2, typename Type1>
	__global__ void KConvert(Type2* out, Type1* in, size_t size, float threshold);

	template<>
	__global__ void KConvert<complex,complex16>(complex* out, complex16* in, size_t size, float threshold);

	template<>
	__global__ void KConvert<complex16,complex>(complex16* out, complex* in, size_t size, float threshold);

	#endif

	// #if !defined(DOXYGEN_SHOULD_SKIP_THIS) // adding because sphinx exhale has a bug
	template<typename Type>
	inline __device__ Type convert(float var, float threshold ){return var;};
	// #endif // DOXYGEN_SHOULD_SKIP_THIS

	/*we divide by 1024 just for normalization*/
	
	#ifdef _USING_FP16
	template<>
	inline __device__ half convert(float var, float threshold){return __float2half(var);};
	#endif

	template<>
	inline __device__ uint16_t convert(float var, float threshold)
	{
		var = __saturatef(var/(2*threshold) + 0.5f);
		return  ((1<<16)-1) * var;
	};

	template<>
	inline __device__ uint8_t convert(float var, float threshold)
	{
		var = __saturatef(var/(2*threshold) + 0.5f);
		return  ((1<<8)-1) * var;
	};

	// float just has 24 bits (e.g 1 << 24)
	template<>
	inline __device__ int convert(float var, float threshold)
	{
		var = __saturatef(var/(2*threshold) + 0.5f);
		return  ((1<<24)-1) * var;
	};

	template<typename Type>
	inline __device__ void set_pixel(void *recon, float var, int i, int j, int wdI, float threshold)
	{
		((Type *)recon)[j*wdI + i]  = convert<Type>(var, threshold);
	};
	
	inline __device__ void set_pixel(void *recon, float var, int i, int j, int wdI, float threshold, EType::TypeEnum raftDatatype)
	{
		switch(raftDatatype)
		{
			case EType::TypeEnum::UINT8:
				set_pixel<uint8_t>(recon, var, i, j, wdI, threshold);
			break;
			case EType::TypeEnum::UINT16:
				set_pixel<uint16_t>(recon, var, i, j, wdI, threshold);
			break;
			case EType::TypeEnum::INT32:
				set_pixel<int>(recon, var, i, j, wdI, threshold);
			break;
			case EType::TypeEnum::FLOAT32:
				set_pixel<float>(recon, var, i, j, wdI, threshold);
			break;
	#ifdef _USING_FP16
			case EType::TypeEnum::HALF:
				set_pixel<half>(recon, var, i, j, wdI, threshold);
			break;
	#endif
		}
	};

	template<typename Type>
	__global__ void KFFTshift1(Type* img, size_t sizex, size_t sizey);

	template<typename Type>
	__global__ void KFFTshift2(Type* img, size_t sizex, size_t sizey);

	template<typename Type>
	__global__ void KFFTshift3(Type* img, size_t sizex, size_t sizey, size_t sizez);

}

namespace Reduction
{
	inline __device__ void KSharedReduce32(float* intermediate)
	{
		if(threadIdx.x < 16)
		{
			for(int group = 16; group > 0; group /= 2)
			{
				if(threadIdx.x < group)
				{
					intermediate[threadIdx.x] += intermediate[threadIdx.x+group];
					__syncwarp((1<<group)-1);
				}
			}
		}
	}

	inline __device__ void KSharedReduce(float* intermediate, int size)
	{
		if(threadIdx.x < 32)
		{
			float s = 0;
			for(int idx = threadIdx.x; idx < size; idx += 32)
				s += intermediate[threadIdx.x];

			intermediate[threadIdx.x] = s;
			__syncwarp();
			KSharedReduce32(intermediate);
		}
	}

	__global__ void KGlobalReduce(float* out, const float* in, size_t size);

}

namespace Sync
{
	template<typename Type>
	__global__ void KWeightedLerp(Type* val, const Type* acc, const float* div, size_t size, float lerp);

	template<typename Type>
	__global__ void KMaskedSum(Type* cval, const Type* acc, size_t size, const uint32_t* mask2);

	template<typename Type>
	__global__ void KMaskedBroadcast(Type* cval, const Type* acc, size_t size, const uint32_t* mask2);

	template<typename Type>
	__global__ void KSetMask(uint32_t* mask, const Type* value, size_t size, float thresh); //{};

	template<>
	__global__ void KSetMask<float>(uint32_t* mask, const float* fval, size_t size, float thresh);

}

namespace opt
{
    inline __host__ __device__ int assert_dimension(int size1, int size2){ return ( size1 == size2 ? 1 : 0 ); };

	inline __host__ __device__ size_t getIndex3d(dim3 size, int i, int j, int k){ return (size_t)(size.y * k * size.x + size.x * j + i); };
    
    inline __host__ __device__ size_t getIndex2d(dim3 size, int i, int j, int k){ return (size_t)(size.x * j + i); };

    inline __host__ __device__ int assert_dimension_xyz(dim3 size1, dim3 size2)
    { 
        if ( assert_dimension(size1.x, size2.x) == 1 && 
             assert_dimension(size1.y, size2.y) == 1 && 
             assert_dimension(size1.z, size2.z) == 1
            )
            return 1; 
        else if (size2.z == 1)
            return 2;
        else 
            return 0;
    };

    inline __host__ __device__ int assert_dimension_xy(dim3 size1, dim3 size2)
    { 
        if ( assert_dimension(size1.x, size2.x) == 1 && 
             assert_dimension(size1.y, size2.y) == 1  
            )
            return 1; 
        else
            return 0;
    };

    inline __host__ __device__ int assert_dimension_xz(dim3 size1, dim3 size2)
    { 
        if ( assert_dimension(size1.x, size2.x) == 1 && 
             assert_dimension(size1.z, size2.z) == 1
            )
            return 1; 
        else
            return 0;
    };

    inline __host__ __device__ int assert_dimension_yz(dim3 size1, dim3 size2)
    { 
        if ( assert_dimension(size1.y, size2.y) == 1 && 
             assert_dimension(size1.z, size2.z) == 1
            )
            return 1; 
        else
            return 0;
    };

    template<typename Type>
    Type* allocGPU(size_t nsize);

    template<typename Type1, typename Type2>
    void pointTopointProd(GPU gpus, Type1 *a, Type2 *b, Type1 *ans, dim3 sizea, dim3 sizeb);

    template<>
    void product<float,float>(GPU gpus, float *a, float *b, float *ans, dim3 sizea, dim3 sizeb);

    template<>
    void product<float,cufftComplex>(GPU gpus, float *a, cufftComplex *b, float *ans, dim3 sizea, dim3 sizeb);

    template<>
    void product<cufftComplex,float>(GPU gpus, cufftComplex *a, float *b, cufftComplex *ans, dim3 sizea, dim3 sizeb);

    template<>
    void product<cufftComplex,cufftComplex>(GPU gpus, cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 sizea, dim3 sizeb);

    __global__ void product_Real_by_Real(float *a, float *b, float *ans, dim3 sizea, dim3 sizeb);
    __global__ void product_Real_by_Complex(float *a, cufftComplex *b, float *ans, dim3 sizea, dim3 sizeb);
    __global__ void product_Complex_by_Real(cufftComplex *a, float *b, cufftComplex *ans, dim3 sizea, dim3 sizeb);
    __global__ void product_Complex_by_Complex(cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 sizea, dim3 sizeb);

}


/* Function declaration for file ./cuda/src/common/opt.cu */

__global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles)


#endif