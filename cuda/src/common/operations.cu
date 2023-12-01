#include "../../inc/include.h"
#include "../../inc/common/operations.hpp"

/*======================================================================*/
/* struct EType (in 'inc/commons/operations.hpp') functions definitions */

size_t EType::Size(EType::TypeEnum datatype)
{
    static const size_t datasizes[] = {0,1,2,4,2,4,8};
    return datasizes[static_cast<int>(datatype)];
}

std::string EType::String(EType::TypeEnum type)
{  
    static const std::string datanames[] = {"INVALID", 
                                            "UINT8", 
                                            "UINT16", 
                                            "INT32", 
                                            "HALF", 
                                            "FLOAT32", 
                                            "DOUBLE"
                                            };

    return datanames[static_cast<int>(type)];
}

EType EType::Type(const std::string& nametype)
{
    EType etype;

    for(int i=0; i<static_cast<int>(EType::TypeEnum::NUM_ENUMS); i++)
            if( nametype == EType::String( static_cast<EType::TypeEnum>(i) ) )
                    etype.type = static_cast<EType::TypeEnum>(i);

    return etype;
}

/*============================================================================*/
/* namespace BasicOps (in 'inc/commons/operations.hpp') functions definitions */

template<typename Type1, typename Type2>
__global__ void BasicOps::KB_Add(Type1* a, const Type2* b, size_t size, size_t size2)
{
    const size_t index = BasicOps::GetIndex();

    if(index >= size)
            return;

    a[index] += b[index % size2];
}

template<typename Type1, typename Type2>
__global__ void BasicOps::KB_Sub(Type1* a, const Type2* b, size_t size, size_t size2)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        a[index] -= b[index%size2];
}

template<typename Type1, typename Type2>
__global__ void BasicOps::KB_Mul(Type1* a, const Type2* b, size_t size, size_t size2)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        a[index] *= b[index%size2];
}


template<typename Type1, typename Type2>
__global__ void BasicOps::KB_Div(Type1* a, const Type2* b, size_t size, size_t size2)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        a[index] /= b[index%size2];
}

template<typename Type1, typename Type2=Type1>
__global__ void BasicOps::KB_Add(Type1* a, Type2 n, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        a[index] += n;
}

template<typename Type1, typename Type2=Type1>
__global__ void BasicOps::KB_Sub(Type1* a, Type2 n, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        a[index] -= n;
}

template<typename Type1, typename Type2=Type1>
__global__ void BasicOps::KB_Mul(Type1* a, Type2 n, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        a[index] *= n;
}

template<typename Type1, typename Type2=Type1>
__global__ void BasicOps::KB_Div(Type1* a, Type2 n, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        if(index >= size)
                return;

        a[index] /= n;
}


template<typename Type>
__global__ void BasicOps::KB_Log(Type* a, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        a[index] = logf(fmaxf(a[index],1E-10f));
}

template<typename Type>
__global__ void BasicOps::KB_Exp(Type* a, size_t size, bool bNeg)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        float var = a[index];
        a[index] = expf( bNeg ? (-var) : var );
}

template<typename Type>
__global__ void BasicOps::KB_Clamp(Type* a, const Type b, const Type c, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        a[index] = clamp(a[index],b,c);
}

template<typename Type>
__global__ void BasicOps::KB_log1j(float* out, const Type* in, size_t size) {}

template<typename Type>
__global__ void BasicOps::KB_exp1j(Type* out, const float* in, size_t size) {}

template<>
__global__ void BasicOps::KB_log1j<complex>(float* out, const complex* in, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        out[index] = in[index].angle();
}

template<>
__global__ void BasicOps::KB_exp1j<complex>(complex* out, const float* in, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        out[index] = complex::exp1j(in[index]);
}


template<typename Type>
__global__ void BasicOps::KB_Power(Type* a, float P, size_t size) {}

template<>
__global__ void BasicOps::KB_Power<float>(float* out, float P, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        out[index] = expf(logf(fmaxf(fabsf(out[index]),1E-25f))*P);
}

template<>
__global__ void BasicOps::KB_Power<complex>(complex* out, float P, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        out[index] = complex::exp1j(out[index].angle()*P)*expf(logf(fmaxf(out[index].abs(),1E-25f))*P);
}

template<typename Type>
__global__ void BasicOps::KB_ABS2(float* out, Type* a, size_t size) {}

template<>
__global__ void BasicOps::KB_ABS2<complex>(float* out, complex* in, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        out[index] = in[index].abs2();
}

template<>
__global__ void BasicOps::KB_ABS2<float>(float* out, float* in, size_t size)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        out[index] = in[index]*in[index];
}

#ifdef _USING_FP16
template<typename Type2, typename Type1>
__global__ void BasicOps::KConvert(Type2* out, Type1* in, size_t size, float threshold)
{
        const size_t index = BasicOps::GetIndex();
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
__global__ void BasicOps::KConvert<complex,complex16>(complex* out, complex16* in, size_t size, float threshold)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        out[index] = complex(in[index]);
}
template<>
__global__ void BasicOps::KConvert<complex16,complex>(complex16* out, complex* in, size_t size, float threshold)
{
        const size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        out[index] = complex16(in[index]);
}
#endif


template<typename Type>
__global__ void BasicOps::KFFTshift1(Type* img, size_t sizex, size_t sizey)
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
__global__ void BasicOps::KFFTshift2(Type* img, size_t sizex, size_t sizey)
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
__global__ void BasicOps::KFFTshift3(Type* img, size_t sizex, size_t sizey, size_t sizez)
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

/*=============================================================================*/
/* namespace Reduction (in 'inc/commons/operations.hpp') functions definitions */

__global__ void Reduction::KGlobalReduce(float* out, const float* in, size_t size)
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

        Reduction::KSharedReduce32(intermediate);
        if(threadIdx.x==0)
                out[blockIdx.x] = intermediate[0];
}

/*========================================================================*/
/* namespace Sync (in 'inc/commons/operations.hpp') functions definitions */

template<typename Type>
__global__ void Sync::KWeightedLerp(Type* val, const Type* acc, const float* div, size_t size, float lerp)
{	
        size_t index = BasicOps::GetIndex();
        if(index >= size)
                return;

        Type weighed = acc[index] / (div[index]+1E-10f);
        val[index] = weighed*lerp + val[index]*(1.0f-lerp);
}

template<typename Type>
__global__ void Sync::KMaskedSum(Type* cval, const Type* acc, size_t size, const uint32_t* mask2)
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
__global__ void Sync::KMaskedBroadcast(Type* cval, const Type* acc, size_t size, const uint32_t* mask2)
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
__global__ void Sync::KSetMask(uint32_t* mask, const Type* value, size_t size, float thresh){};

template<>
__global__ void Sync::KSetMask<float>(uint32_t* mask, const float* fval, size_t size, float thresh)
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
