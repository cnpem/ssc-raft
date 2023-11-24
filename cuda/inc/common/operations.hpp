/**
@file operations.hpp
@author Giovanni L. Baraldi, Gilberto Martinez
@brief File contains kernels for the most basic operations
@version 0.1
@date 2021-06-14

@copyright Copyright (c) 2021

 */

#ifndef _OPERATIONS_H
#define _OPERATIONS_H

#ifdef __CUDACC__
 #define restrict __restrict__
#else
 #define restrict
#endif

#include "complex.hpp"
#include <string>
#include "logerror.hpp"


inline __device__ int iabs(int n)
{
	return n<0?-n:n;
}

static __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

static __device__ half clamp(half f, half a, half b)
{
    return fmaxf(a, fminf(f, b));
}
static __device__ __host__ half clamp(half f, float a=-32768.0f, float b=32768.0f)
{
    return fmaxf(__float2half(a), fminf(f, __float2half(b)));
}

#if __CUDA_ARCH__ >= 530
static __device__ complex16 clamp(complex16 f, float a, float b)
{
    return complex16(clamp(f.x,a,b),clamp(f.y,a,b));
} 
#endif

static __device__ __host__ complex clamp(const complex& f, float a, float b)
{
    return complex(clamp(f.x,a,b),clamp(f.y,a,b));
}
static __device__ __host__ complex clamp(const complex& f, const complex& a, const complex& b)
{
    return complex(clamp(f.x,a.x,b.x),clamp(f.y,a.y,b.y));
}

inline __device__ complex exp1j(float f)
{
	complex c;
	__sincosf(f, &c.y, &c.x);
	return c;
}

template<typename Type>
inline __device__ __host__ Type sq(const Type& x){ return x*x; }

struct EType
{
        enum class TypeEnum 
        {
                INVALID=0,
                UINT8=1,
                UINT16=2,
                INT32=3,
                HALF=4,
                FLOAT32=5,
                DOUBLE=6,
                NUM_ENUMS
        };

        EType() = default;
        EType(TypeEnum etype): type(etype) {};

        static size_t Size(TypeEnum datatype)
        {
                static const size_t datasizes[] = {0,1,2,4,2,4,8};
                return datasizes[static_cast<int>(datatype)];
        }
        size_t Size() const { return Size(type); }

        static std::string String(TypeEnum type)
        {  
                static const std::string datanames[] = {"INVALID", "UINT8", "UINT16", "INT32", "HALF", "FLOAT32", "DOUBLE"};
                return datanames[static_cast<int>(type)];
        };
        std::string String() { return String(type); };

        static EType Type(const std::string& nametype)
        {
                EType etype;

                for(int i=0; i<static_cast<int>(TypeEnum::NUM_ENUMS); i++)
                        if( nametype == String( static_cast<TypeEnum>(i) ) )
                                etype.type = static_cast<TypeEnum>(i);

                return etype;
        }

        TypeEnum type = TypeEnum::INVALID;
};

namespace BasicOps
{
        inline __device__ size_t GetIndex(){ return threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y*blockIdx.z)); }

        template<typename Type1, typename Type2>
        __global__ void KB_Add(Type1* a, const Type2* b, size_t size, size_t size2)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] += b[index%size2];
        }

        template<typename Type1, typename Type2>
        __global__ void KB_Sub(Type1* a, const Type2* b, size_t size, size_t size2)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] -= b[index%size2];
        }

        template<typename Type1, typename Type2>
        __global__ void KB_Mul(Type1* a, const Type2* b, size_t size, size_t size2)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] *= b[index%size2];
        }
        

        template<typename Type1, typename Type2>
        __global__ void KB_Div(Type1* a, const Type2* b, size_t size, size_t size2)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] /= b[index%size2];
        }

        template<typename Type1, typename Type2=Type1>
        __global__ void KB_Add(Type1* a, Type2 n, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] += n;
        }

        template<typename Type1, typename Type2=Type1>
        __global__ void KB_Sub(Type1* a, Type2 n, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] -= n;
        }

        template<typename Type1, typename Type2=Type1>
        __global__ void KB_Mul(Type1* a, Type2 n, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] *= n;
        }

        template<typename Type1, typename Type2=Type1>
        __global__ void KB_Div(Type1* a, Type2 n, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                if(index >= size)
                        return;

                a[index] /= n;
        }


        template<typename Type>
        __global__ void KB_Log(Type* a, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] = logf(fmaxf(a[index],1E-10f));
        }

        template<typename Type>
        __global__ void KB_Exp(Type* a, size_t size, bool bNeg)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                float var = a[index];
                a[index] = expf( bNeg ? (-var) : var );
        }

        template<typename Type>
        __global__ void KB_Clamp(Type* a, const Type b, const Type c, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                a[index] = clamp(a[index],b,c);
        }

        template<typename Type>
        __global__ void KB_log1j(float* out, const Type* in, size_t size) {}
        
        template<typename Type>
        __global__ void KB_exp1j(Type* out, const float* in, size_t size) {}

        template<>
        __global__ void KB_log1j<complex>(float* out, const complex* in, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                out[index] = in[index].angle();
        }
        
        template<>
        __global__ void KB_exp1j<complex>(complex* out, const float* in, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                out[index] = complex::exp1j(in[index]);
        }


        template<typename Type>
        __global__ void KB_Power(Type* a, float P, size_t size) {}

        template<>
        __global__ void KB_Power<float>(float* out, float P, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                out[index] = expf(logf(fmaxf(fabsf(out[index]),1E-25f))*P);
        }

        template<>
        __global__ void KB_Power<complex>(complex* out, float P, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                out[index] = complex::exp1j(out[index].angle()*P)*expf(logf(fmaxf(out[index].abs(),1E-25f))*P);
        }

        template<typename Type>
        __global__ void KB_ABS2(float* out, Type* a, size_t size) {}

        template<>
        __global__ void KB_ABS2<complex>(float* out, complex* in, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                out[index] = in[index].abs2();
        }

        template<>
        __global__ void KB_ABS2<float>(float* out, float* in, size_t size)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                out[index] = in[index]*in[index];
        }

        #ifdef _USING_FP16
        template<typename Type2, typename Type1>
        __global__ void KConvert(Type2* out, Type1* in, size_t size, float threshold)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;
                
                float res = (float)in[index];
                
                if(std::is_same<Type2,__half>::value == true)
                        res *= 1024.0f;
                else if(std::is_floating_point<Type2>::value == false)
                        res = fminf(res/threshold,1.0f) * ((1UL<<(8*(sizeof(Type2) & 0X7) ))-1);

                out[index] = Type2(res);
        }

        template<>
        __global__ void KConvert<complex,complex16>(complex* out, complex16* in, size_t size, float threshold)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                out[index] = complex(in[index]);
        }
        template<>
        __global__ void KConvert<complex16,complex>(complex16* out, complex* in, size_t size, float threshold)
        {
                const size_t index = GetIndex();
                if(index >= size)
                        return;

                out[index] = complex16(in[index]);
        }
        #endif

        // #if !defined(DOXYGEN_SHOULD_SKIP_THIS) // adding because sphinx exhale has a bug
        template<typename Type>
        inline __device__ Type convert(float var, float threshold )
        {
                return var; 
        }
        // #endif // DOXYGEN_SHOULD_SKIP_THIS

        /*we divide by 1024 just for normalization*/
        
        #ifdef _USING_FP16
        template<>
        inline __device__ half convert(float var, float threshold)
        {
                return __float2half(var);
        }
        #endif

        template<>
        inline __device__ uint16_t convert(float var, float threshold)
        {
                var = __saturatef(var/(2*threshold) + 0.5f);
                return  ((1<<16)-1) * var;
        }

        template<>
        inline __device__ uint8_t convert(float var, float threshold)
        {
                var = __saturatef(var/(2*threshold) + 0.5f);
                return  ((1<<8)-1) * var;
        }

        // float just has 24 bits (e.g 1 << 24)
        template<>
        inline __device__ int convert(float var, float threshold)
        {
                var = __saturatef(var/(2*threshold) + 0.5f);
                return  ((1<<24)-1) * var;
        }

        template<typename Type>
        inline __device__ void set_pixel(void *recon, float var, int i, int j, int wdI, float threshold)
        {
                ((Type *)recon)[j*wdI + i]  = convert<Type>(var, threshold);
        }
        
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
        }

        template<typename Type>
        __global__ void KFFTshift1(Type* img, size_t sizex, size_t sizey)
        {
                size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
                size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
                size_t idz = blockIdx.z;

                if(idx < sizex/2 && idy < sizey)
                {
                        size_t index1 = idz*sizex*sizey + idy*sizex + idx;
                        size_t index2 = idz*sizex*sizey + idy*sizex + idx+sizex/2;

                        Type temp = img[index1];
                        img[index1] = img[index2];
                        img[index2] = temp;
                }
        }
        template<typename Type>
        __global__ void KFFTshift2(Type* img, size_t sizex, size_t sizey)
        {
                size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
                size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
                size_t idz = blockIdx.z;

                if(idx < sizex && idy < sizey/2)
                {
                        size_t index1 = idz*sizex*sizey + idy*sizex + idx;
                        size_t index2 = idz*sizex*sizey + (idy+sizey/2)*sizex + (idx+sizex/2)%sizex;

                        Type temp = img[index1];
                        img[index1] = img[index2];
                        img[index2] = temp;
                }
        }
        template<typename Type>
        __global__ void KFFTshift3(Type* img, size_t sizex, size_t sizey, size_t sizez)
        {
                size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
                size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
                size_t idz = blockIdx.z;

                if(idx < sizex && idy < sizey/2)
                {
                        size_t index1 = idz*sizex*sizey + idy*sizex + idx;
                        size_t index2 = ((idz+sizez/2)%sizez)*sizex*sizey + (idy+sizey/2)*sizex + (idx+sizex/2)%sizex;

                        Type temp = img[index1];
                        img[index1] = img[index2];
                        img[index2] = temp;
                }
        }
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

        __global__ void KGlobalReduce(float* out, const float* in, size_t size)
        {
                __shared__ float intermediate[32];
                if(threadIdx.x<32)
                        intermediate[threadIdx.x] = 0;
                __syncthreads();
                
                float mine = 0;

                for(size_t index = threadIdx.x + blockIdx.x*blockDim.x; index < size; index += blockDim.x*gridDim.x)
                        mine += in[index];

                atomicAdd(intermediate + threadIdx.x%32, mine);

                __syncthreads();

                KSharedReduce32(intermediate);
                if(threadIdx.x==0)
                        out[blockIdx.x] = intermediate[0];
        }
}

namespace Sync
{
        template<typename Type>
        __global__ void KWeightedLerp(Type* val, const Type* acc, const float* div, size_t size, float lerp)
        {	
                size_t index = BasicOps::GetIndex();
                if(index >= size)
                        return;

                Type weighed = acc[index] / (div[index]+1E-10f);
                val[index] = weighed*lerp + val[index]*(1.0f-lerp);
        }

        template<typename Type>
        __global__ void KMaskedSum(Type* cval, const Type* acc, size_t size, const uint32_t* mask2)
        {
                size_t index = BasicOps::GetIndex();
                if(index >= size)
                        return;

                uint32_t mask = mask2[index/32];
                bool value = (mask>>(index&0x1F)) & 0x1;

                if(value)
                        cval[index] += acc[index];
        }

        template<typename Type>
        __global__ void KMaskedBroadcast(Type* cval, const Type* acc, size_t size, const uint32_t* mask2)
        {
                size_t index = BasicOps::GetIndex();
                if(index >= size)
                        return;

                uint32_t mask = mask2[index/32];
                bool value = (mask>>(index&0x1F)) & 0x1;

                if(value)
                        cval[index] = acc[index];
        }

        template<typename Type>
        __global__ void KSetMask(uint32_t* mask, const Type* value, size_t size, float thresh){};

        template<>
        __global__ void KSetMask<float>(uint32_t* mask, const float* fval, size_t size, float thresh)
        {
                __shared__ uint32_t shvalue[1];
                if(threadIdx.x < 32)
                        shvalue[threadIdx.x] = 0;

                __syncthreads();

                size_t index = BasicOps::GetIndex();
                if(index >= size)
                        return;

                uint32_t value = (fval[index] > thresh) ? 1 : 0;
                value = value << threadIdx.x;
                        
                atomicOr(shvalue, value);

                __syncthreads();
                if(threadIdx.x==0)
                        mask[index/32] = shvalue[0];
        }
}

#endif