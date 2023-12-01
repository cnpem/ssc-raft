// @Author: Giovanni L. Baraldi
// File contains implementations for single and multigpu memory management, along with some commom operations.

#ifndef _TYPES_H
#define _TYPES_H

#ifdef __CUDACC__
 #define restrict __restrict__
#else
 #define restrict
#endif

#include "../include.h"

#include "complex.hpp"
#include "operations.hpp"
#include "logerror.hpp"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

extern "C" {
__global__ void KSwapZ(float* vec, size_t N);
__global__ void KSwapZ_4gpu(float* vec0, float* vec1, float* vec2, float* vec3, int gpu, size_t N);
}

enum MemoryType
{
        ENoAlloc = 0,
        EAllocGPU = 1,
        EAllocCPU = 2,
        EAllocCPUGPU = 3,
        EAllocManaged = 4,
        EAllocSBManaged = 12,
};

template<typename Type> 
struct Image2D
{
        MemoryType memorytype;

        bool bIsManaged() const { return memorytype == MemoryType::EAllocManaged; }
        bool bHasAllocCPU() const { return (memorytype&MemoryType::EAllocCPU) == MemoryType::EAllocCPU; }
        bool bHasAllocGPU() const { return (memorytype&MemoryType::EAllocGPU) == MemoryType::EAllocGPU; }

        Image2D(size_t _sizex, size_t _sizey, size_t _sizez = 1, MemoryType memtype = MemoryType::EAllocManaged): 
                sizex(_sizex), sizey(_sizey), sizez(_sizez), size(_sizex *_sizey*_sizez), memorytype(memtype), gpuptr(nullptr), cpuptr(nullptr)
        {
                LogB("Creating Image of size: " << sizex << " " << sizey << " " << sizez << " (" << sizeof(Type) << ")");
                AllocManaged();
                AllocGPU();
                AllocCPU();
                LogE();
        }

        Image2D(Type* newdata, size_t _sizex, size_t _sizey, size_t _sizez = 1, MemoryType memtype = MemoryType::EAllocManaged): 
                Image2D<Type>(_sizex,_sizey,_sizez,memtype)
        {
                if(memtype == MemoryType::ENoAlloc)
                {
                        this->cpuptr = newdata;
                        this->gpuptr = newdata;
                }
                else
                {
                        LogB("Creating Image of size: " << sizex << " " << sizey << " " << sizez << " (" << sizeof(Type) << ") from pointer " << newdata);
                        if(bHasAllocGPU() || bIsManaged())
                                this->CopyFrom(newdata);
                        if(bHasAllocCPU())
                                memcpy(cpuptr, newdata, sizeof(Type)*GetSize());
                        LogE();
                }
        }
        explicit Image2D(Image2D<Type>& other): 
                Image2D(other.sizex,other.sizey,other.sizez,(other.memorytype != MemoryType::ENoAlloc) ? other.memorytype : MemoryType::EAllocManaged)
        {
                CopyFrom(other);
        }
        Image2D(const dim3& dim, MemoryType memtype = MemoryType::EAllocManaged): Image2D(dim.x,dim.y,dim.z,memtype) {};
        Image2D(Type* newdata, const dim3& dim, MemoryType memtype = MemoryType::EAllocManaged): Image2D(newdata, dim.x,dim.y,dim.z,memtype) {};

        virtual ~Image2D() 
        {
                cudaDeviceSynchronize();
                HANDLE_ERROR( cudaGetLastError() ); 
                LogB("Dealloc Image of pointer " << gpuptr);

                DeallocManaged();
                DeallocGPU(); 
                DeallocCPU();

                if(redarray && redarray != this) // redarray = this for 1D arrays
                        delete redarray;

                cudaDeviceSynchronize();
                LogE();
                HANDLE_ERROR( cudaGetLastError() );
        }

        size_t sizex = 0;
        size_t sizey = 0;
        size_t sizez = 1;
        size_t size = 0;

        Type* gpuptr = nullptr;
        Type* cpuptr = nullptr;

        void SetGPUToZero(){ HANDLE_ERROR( cudaMemset(gpuptr, 0, size*sizeof(Type)) ); SyncDebug; }
        void SetGPUToZero(cudaStream_t stream){ HANDLE_ERROR( cudaMemsetAsync(gpuptr, 0, size*sizeof(Type), stream) ); SyncDebug; }
        void SetCPUToZero(){  assert(cpuptr != nullptr); memset(cpuptr, 0, size*sizeof(Type)); }

        static const size_t blocksize = 64;
        dim3 Shape() const { return dim3(sizex,sizey,sizez); };
        dim3 ShapeThread() const { dim3 shp = Shape(); return dim3(blocksize, 1, 1); };
        dim3 ShapeBlock() const { dim3 shp = Shape(); return dim3((shp.x + blocksize - 1) / blocksize, shp.y, shp.z); };

        dim3 LinearThread() const { dim3 shp = Shape(); return dim3(size < blocksize ? ((size+31)%32) : blocksize, 1, 1); };
        dim3 LinearBlock() const { dim3 shp = Shape(); return dim3((size + blocksize - 1) / blocksize, 1, 1); };

        void AllocGPU()
        {
                if(!gpuptr && bHasAllocGPU()) HANDLE_ERROR( cudaMalloc((void**)&gpuptr, sizeof(Type)*size) );
        }
        void DeallocGPU()
        {
                if(gpuptr && bHasAllocGPU()) cudaFree(gpuptr);
                gpuptr = nullptr;
        }
        void AllocCPU()
        {
                if(!cpuptr && bHasAllocCPU()) cpuptr = new Type[size];
        }
        void DeallocCPU()
        { 
                if(cpuptr && bHasAllocCPU()) delete[] cpuptr; //cudaFreeHost(cpuptr);
                cpuptr = nullptr;
        }
        void AllocManaged()
        {
                if(bIsManaged())
                {
                        HANDLE_ERROR( cudaMallocManaged((void**)&gpuptr, sizeof(Type)*size) );
                        cpuptr = gpuptr;
                }
        }
        void DeallocManaged()
        {
                if(gpuptr && bIsManaged())
                {
                        cudaFree(gpuptr);
                        cpuptr = gpuptr = nullptr;
                }
        }
        
        template<typename Type2=Type>
        Image2D<Type>& operator+=(const Image2D<Type2>& other)
        {
                ErrorAssert( this->size == other.size || this->sizex == other.sizex, "Incompatible GPU shape for addition." );
                BasicOps::KB_Add<Type,Type2><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, (const Type2*)other.gpuptr, this->size, other.size);
                return *this;
        }
        template<typename Type2=Type>
        Image2D<Type>& operator-=(const Image2D<Type2>& other)
        {
                ErrorAssert( this->size == other.size || this->sizex == other.sizex, "Incompatible GPU shape for subtraction." );
                BasicOps::KB_Sub<Type,Type2><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, (const Type2*)other.gpuptr, this->size, other.size);
                return *this;
        }
        template<typename Type2=Type>
        Image2D<Type>& operator*=(const Image2D<Type2>& other)
        {
                ErrorAssert( this->size == other.size || this->sizex == other.sizex, "Incompatible GPU shape for multiplication." );
                BasicOps::KB_Mul<Type,Type2><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, (const Type2*)other.gpuptr, this->size, other.size);
                return *this;
        }
        template<typename Type2=Type>
        Image2D<Type>& operator/=(const Image2D<Type2>& other)
        {
                ErrorAssert( this->size == other.size || this->sizex == other.sizex, "Incompatible GPU shape for division." );
                BasicOps::KB_Div<Type,Type2><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, (const Type2*)other.gpuptr, this->size, other.size);
                return *this;
        }



        template<typename Type2=Type>
        Image2D<Type>& operator+=(Type2 other)
        {                
                BasicOps::KB_Add<Type,Type2><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, other, this->size);
                return *this;
        }
        template<typename Type2=Type>
        Image2D<Type>& operator-=(Type2 other)
        {                
                BasicOps::KB_Sub<Type,Type2><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, other, this->size);
                return *this;
        }
        template<typename Type2=Type>
        Image2D<Type>& operator*=(Type2 other)
        {                
                BasicOps::KB_Mul<Type,Type2><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, other, this->size);
                return *this;
        }
        template<typename Type2=Type>
        Image2D<Type>& operator/=(Type2 other)
        {                
                BasicOps::KB_Div<Type,Type2><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, other, this->size);
                return *this;
        }

#ifdef _USING_FP16
        template<typename Type2>
        void DataAs(Type2* fill, size_t tsize = 0, float threshold = -1) const
        {
                if(tsize == 0)
                        tsize = this->size;

                //if(!std::is_same<Type,Type2>::value)

                BasicOps::KConvert<Type2,Type><<<(tsize+31)/32,32>>>(fill, this->gpuptr, tsize, threshold);
        }

        void DataAs(void* fill, EType::TypeEnum type, size_t tsize = 0, float threshold = -1) const
        {
                switch(type){
                        case EType::TypeEnum::UINT8:
                                DataAs<uint8_t>((uint8_t*)fill, tsize, threshold);
                        break;
                        case EType::TypeEnum::UINT16:
                                DataAs<uint16_t>((uint16_t*)fill, tsize, threshold);
                        break;
                        case EType::TypeEnum::INT32:
                                DataAs<int32_t>((int32_t*)fill, tsize, threshold);
                        break;
                        case EType::TypeEnum::FLOAT32:
                                DataAs<float>((float*)fill, tsize, threshold);
                        break;
                        case EType::TypeEnum::DOUBLE:
                                DataAs<double>((double*)fill, tsize, threshold);
                        break;
                        case EType::TypeEnum::HALF:
                                DataAs<__half>((__half*)fill, tsize, threshold);
                        break;
                        default:
                                std::cerr << "Invalid data type!" << std::endl; exit(-1);
                        break;
                }
        }
#endif

        void CopyFrom(const Type* other, cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                if(cpysize == -1)
                        cpysize = this->size;
                        
                ErrorAssert(other != nullptr, "Syncing from empty pointer!");
                ErrorAssert(gpuptr != nullptr && cpysize <= this->size, "Not enough space for sync.!");

                if(stream==0)
                        cudaMemcpy(this->gpuptr, other, cpysize*sizeof(Type), cudaMemcpyDefault);
                else
                        cudaMemcpyAsync(this->gpuptr, other, cpysize*sizeof(Type), cudaMemcpyDefault, stream);
        }
        void CopyTo(Type* outptr, cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                ErrorAssert(gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");

                if(cpysize < 0)
                        cpysize = size*sizeof(Type);
                else
                        cpysize = cpysize*sizeof(Type);

                if( stream == 0 )
                        HANDLE_ERROR( cudaMemcpy((void*)outptr, (void*)gpuptr, cpysize, cudaMemcpyDefault) )
                else
                        HANDLE_ERROR( cudaMemcpyAsync((void*)outptr, (void*)gpuptr, cpysize, cudaMemcpyDefault, stream) )
                SyncDebug;
        }

        void CopyFrom(const Image2D<Type>& other, cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                CopyFrom(other.gpuptr, stream, cpysize);
        }
        void CopyTo(Image2D<Type>& other, cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                CopyTo(other.gpuptr, stream, cpysize);
        }

        void LoadToGPU(cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                ErrorAssert(cpuptr && gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");

                if(gpuptr == cpuptr)
                        return;

                if(cpysize < 0)
                        cpysize = size*sizeof(Type);
                else
                        cpysize = cpysize*sizeof(Type);

                if( stream == 0 )
                        HANDLE_ERROR( cudaMemcpy((void*)gpuptr, (void*)cpuptr, sizeof(Type)*size, cudaMemcpyHostToDevice) )
                else
                        HANDLE_ERROR( cudaMemcpyAsync((void*)gpuptr, (void*)cpuptr, sizeof(Type)*size, cudaMemcpyHostToDevice, stream) )
                SyncDebug;
        }
        void LoadFromGPU(cudaStream_t stream = 0, int64_t cpysize = -1)
        {
                ErrorAssert(cpuptr && gpuptr && size != 0, "Call to gpu->cpu memcopy without allocated array.");
                if(gpuptr == cpuptr)
                        return;

                if(cpysize < 0)
                        cpysize = size*sizeof(Type);
                else
                        cpysize = cpysize*sizeof(Type);

                if( stream == 0 )
                        HANDLE_ERROR( cudaMemcpy((void*)cpuptr, (void*)gpuptr, cpysize, cudaMemcpyDeviceToHost) )
                else
                        HANDLE_ERROR( cudaMemcpyAsync((void*)cpuptr, (void*)gpuptr, cpysize, cudaMemcpyDeviceToHost, stream) )
                SyncDebug;
        }

        size_t GetSize() const { return size; };

        void exp(bool bNeg = false){ BasicOps::KB_Exp<<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, size, bNeg); };
        void ln(){ BasicOps::KB_Log<<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, size); };

        void exp1j(Image2D<float>& other){ BasicOps::KB_exp1j<<<ShapeBlock(), ShapeThread()>>>(this->gpuptr, other.gpuptr, size); };
        void angle(Image2D<float>& other){ BasicOps::KB_log1j<<<ShapeBlock(), ShapeThread()>>>(other.gpuptr, this->gpuptr, size); };

        void SwapZ()
        {
                ErrorAssert(sizex == sizey && sizex == sizez, "Invalid shape for square transpose.");
                KSwapZ<<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, sizex);
        };

        void InitFFTC2C()
        {

        }

        void Clamp(Type a, Type b)
        {
                BasicOps::KB_Clamp<<<LinearBlock(),LinearThread()>>>(this->gpuptr, a, b, this->size);
        }
        void Power(float P)
        {
                BasicOps::KB_Power<Type><<<ShapeBlock(),ShapeThread()>>>(this->gpuptr, P, this->size);
        }

        void abs2(Image2D<float>& out)
        {
                BasicOps::KB_ABS2<Type><<<ShapeBlock(),ShapeThread()>>>(out.gpuptr, this->gpuptr, this->size);
        }

        Type SumCPU()
        {
                Type s = 0;
                for(int i=0; i<this->size; i++)
                        s += this->cpuptr[i];
                return s;
        }

        Type SumGPU(cudaStream_t stream = 0)
        {
                if(sizey*sizez > 1)
                {
                        if(redarray == nullptr)
                                redarray = new Image2D<Type>(sizey*sizez,1,1);

                        redarray->SetGPUToZero();
                        Reduction::KGlobalReduce<<<sizey*sizez,sizex>>>(redarray->gpuptr, gpuptr, size);
                }
                else if(redarray == nullptr) // Avoid memory leak on resize ? 
                        redarray = this; // Careful on delete!

                redarray->LoadFromGPU(stream);
                cudaStreamSynchronize(stream);
                return redarray->SumCPU();
        }

        void FFTShift1()
        {
                BasicOps::KFFTshift1<<<ShapeBlock(),ShapeThread()>>>(gpuptr, sizex, sizey);
        }
        void FFTShift2()
        {
                BasicOps::KFFTshift2<<<ShapeBlock(),ShapeThread()>>>(gpuptr, sizex, sizey);
        }
        void FFTShift3()
        {
                BasicOps::KFFTshift3<<<ShapeBlock(),ShapeThread()>>>(gpuptr, sizex, sizey, sizez);
        }

protected:
        Image2D<Type>* redarray = nullptr;
};

template<typename Type>
struct GArray
{
        GArray<Type>(Type* _ptr, const dim3& _shape): ptr(_ptr), shape(_shape) {};
        GArray<Type>(Type* _ptr, size_t x, size_t y, size_t z): ptr(_ptr), shape(dim3(x,y,z)) {};
        GArray<Type>(const Image2D<Type>& img): ptr(img.gpuptr), shape(img.sizex,img.sizey,img.sizez) {};

        Type* ptr;
        dim3 shape;

        __host__ __device__ Type& operator[](size_t i){ return ptr[i]; };
        __host__ __device__ const Type& operator[](size_t i) const { return ptr[i]; };
        __host__ __device__ Type& operator()(size_t k, size_t j, size_t i){ return ptr[(k*shape.y+j)*shape.x+i]; };
        __host__ __device__ const Type& operator()(size_t k, size_t j, size_t i) const { return ptr[(k*shape.y+j)*shape.x+i]; };
        __host__ __device__ Type& operator()(size_t j, size_t i){ return ptr[j*shape.x+i]; };
        __host__ __device__ const Type& operator()(size_t j, size_t i) const { return ptr[j*shape.x+i]; };
};

typedef Image2D<float> rImage;
typedef Image2D<complex> cImage;

#ifdef _USING_FP16
typedef Image2D<complex16> hcImage;
#endif

// Note: The outer brackets are necessary for shallow declaration of g. 
// Removing them may cause compile errors.

#define MEXEC(func,array1,...)\
{\
        for(int g=0; g<array1.ngpus; g++)\
        {\
                cudaSetDevice(array1.indices[g]);\
                dim3& blk = array1.blk[g];\
                dim3& thr = array1.thr[g];\
\
                func<<<blk,thr>>>(array1.Ptr(g),__VA_ARGS__);\
                HANDLE_ERROR( cudaGetLastError() );\
        }\
}

#define MGPULOOP(execcode) { for(int g=0; g<GetN(); g++) { Set(g); execcode; } }
#define MultiKernel(gpus,func,blk,thr,...) { for(int g=0; g<gpus.size(); g++) { cudaSetDevice(gpus[g]); func<<<blk,thr>>>(__VA_ARGS__); } }

struct MultiGPU
{
        MultiGPU(const std::vector<int>& gpus): ngpus(gpus.size()), gpuindices(gpus) {};

        const int ngpus = 1;
        const std::vector<int> gpuindices;

        void Set(int g){ cudaSetDevice(this->gpuindices[g]); }
        int GetN(){ return this->ngpus; }
};

template<typename Type>
struct MImage: public MultiGPU
{ 
        MemoryType memorytype;

        bool bIsManaged() const { return memorytype == MemoryType::EAllocManaged; }
        bool bHasAllocCPU() const { return (memorytype&MemoryType::EAllocCPU) == MemoryType::EAllocCPU; }
        bool bHasAllocGPU() const { return (memorytype&MemoryType::EAllocGPU) == MemoryType::EAllocGPU; }

        Image2D<Type>* arrays[16];
        size_t offsets[16];

        bool bBroadcast = false;
        
        Type* Ptr(int g) const { return arrays[g]->gpuptr; }
        Image2D<Type>& operator[](int n){ return *(arrays[n]); };

        MImage(size_t _sizex, size_t _sizey, size_t _sizez, bool _bBroadcast, const std::vector<int>& gpus, MemoryType memtype = MemoryType::EAllocManaged): 
                MultiGPU(gpus), sizex(_sizex), sizey(_sizey), sizez(_sizez), size(_sizex *_sizey*_sizez), bBroadcast(_bBroadcast), memorytype(memtype)
        {
                ErrorAssert(gpus.size() <= 16, "Too many gpus!");
                Log("Creating MImage of size: " << sizex << " " << sizey << " " << sizez << " (" << sizeof(Type) << ")");

                size_t zstep, zdistrib;

                if(bBroadcast)
                {
                        zstep = 0;
                        zdistrib = sizez;
                }
                else
                {
                        Warning(gpus.size() <= sizez, "Warning: More GPUs than slices!");
                        zstep = (sizez+gpus.size()-1)/gpus.size();
                        zdistrib = zstep;
                }

                for(int g=0; g<this->ngpus; g++)
                {
                        if(zstep*(g+1) > sizez)
                        {
                                if(sizez > zstep*g)
                                        zdistrib = sizez - zstep*g;
                                else
                                        zdistrib = 1;
                        }

                        Set(g);

                        arrays[g] = new Image2D<Type>(sizex,sizey,zdistrib);
                        offsets[g] = sizex*sizey*zstep*g;

                        if(sizez <= zstep*g)
                                arrays[g]->sizez = 0;
                }
        }

        MImage(Type* newdata, size_t _sizex, size_t _sizey, size_t _sizez, bool _bBroadcast, const std::vector<int>& gpus,
                MemoryType memtype = MemoryType::EAllocManaged): MImage<Type>(_sizex,_sizey,_sizez, _bBroadcast, gpus, memtype)
        {
                for(int g=0; g<this->ngpus; g++)
                {
                        Set(g);
                        arrays[g]->CopyFrom(newdata + offsets[g]);
                }
                Log("^ from pointer " << newdata);
        }
        MImage(dim3 dim, bool _bBroadcast, const std::vector<int>& gpus, MemoryType memtype = MemoryType::EAllocManaged):
                MImage<Type>(dim.x,dim.y,dim.z, _bBroadcast, gpus, memtype) {};
        MImage(Type* newdata, dim3 dim, bool _bBroadcast, const std::vector<int>& gpus, MemoryType memtype = MemoryType::EAllocManaged):
                MImage<Type>(newdata,dim.x,dim.y,dim.z, _bBroadcast, gpus, memtype) {};

        MImage(MImage<Type>& other): 
                MImage<Type>(other.sizex,other.sizey,other.sizez, other.bBroadcast, other.gpuindices, other.memorytype) 
        {
                CopyFrom(other);
        };

        virtual ~MImage()
        { 
                HANDLE_ERROR( cudaGetLastError() );
                Log("Deleting MImage of shape: " << sizex << " " << sizey << " " << sizez << " (" << sizeof(Type) << ")");
                MGPULOOP( delete arrays[g]; arrays[g] = nullptr; );
                HANDLE_ERROR( cudaGetLastError() );
        }
        dim3 Shape() const { return dim3(sizex,sizey,sizez); };

        size_t sizex = 0;
        size_t sizey = 0;
        size_t sizez = 1;
        size_t size = 0;

        void SetGPUToZero(){ MGPULOOP( arrays[g]->SetGPUToZero(); ); };
        void Clamp(Type a, Type b) { MGPULOOP( arrays[g]->Clamp(a,b); ); }

        template<typename Type2=Type>
        MImage<Type>& operator+=(const MImage<Type2>& other)
        {
                MGPULOOP( *(arrays[g]) += *(other.arrays[g]); );
                return *this;
        }
        template<typename Type2=Type>
        MImage<Type>& operator-=(const MImage<Type2>& other)
        {
                MGPULOOP( *(arrays[g]) -= *(other.arrays[g]); );
                return *this;
        }
        template<typename Type2=Type>
        MImage<Type>& operator*=(const MImage<Type2>& other)
        {
                MGPULOOP( *(arrays[g]) *= *(other.arrays[g]); );
                return *this;
        }
        template<typename Type2=Type>
        MImage<Type>& operator/=(const MImage<Type2>& other)
        {
                MGPULOOP( *(arrays[g]) /= *(other.arrays[g]); );
                return *this;
        }

        template<typename Type2=Type>
        MImage<Type>& operator+=(Type2 other)
        {
                MGPULOOP( *(arrays[g]) += other; );
                return *this;
        }
        template<typename Type2=Type>
        MImage<Type>& operator-=(Type2 other)
        {                
                MGPULOOP( *(arrays[g]) -= other; );
                return *this;
        }
        template<typename Type2=Type>
        MImage<Type>& operator*=(Type2 other)
        {                
                MGPULOOP( *(arrays[g]) *= other; );
                return *this;
        }
        template<typename Type2=Type>
        MImage<Type>& operator/=(Type2 other)
        {              
                MGPULOOP( *(arrays[g]) /= other; );
                return *this;
        }

        void exp(bool bNeg = false){ for(int g=0; g<GetN(); g++) { Set(g); arrays[g]->exp(bNeg); } };
        void ln(){ for(int g=0; g<GetN(); g++) { Set(g); arrays[g]->ln(); } };

        void exp1j(MImage<float>& other){ MGPULOOP(     arrays[g]->exp1j(*(other.arrays[g]))    )  }
        void angle(MImage<float>& other){ MGPULOOP(     arrays[g]->angle(*(other.arrays[g]))    )  }


        void CopyFrom(const Type* other)
        {
                MGPULOOP( arrays[g]->CopyFrom(other + offsets[g]); );
        }
        void CopyTo(Type* other)
        {
                //ErrorAssert(!bBroadcast, "Source copy has multiple instances!");
                LogB("Copying MImage to pointer. Broadcast=" << bBroadcast);
                if(!bBroadcast)
                {
                        MGPULOOP( arrays[g]->CopyTo(other + offsets[g]); );
                }
                else
                {
                        Set(0); 
                        arrays[0]->CopyTo(other);
                }
                LogE();
        }

        void CopyFrom(const MImage<Type>& other)
        {
                ErrorAssert(other.GetSize() >= GetSize() && other.sizez/other.ngpus >= sizez/ngpus, "Source will not fit!");
                MGPULOOP( arrays[g]->CopyFrom(other.arrays[g][0]); );
        }
        void CopyTo(MImage<Type>& other)
        {
                ErrorAssert(other.GetSize() >= GetSize() && other.sizez/other.ngpus >= sizez/ngpus, "Destination will not fit!");
                MGPULOOP( arrays[g]->CopyTo(other.arrays[g][0]); );
        }

        void LoadToGPU()
        {
                MGPULOOP( arrays[g]->LoadToGPU(); );
        }
        void LoadFromGPU()
        {
                MGPULOOP( arrays[g]->LoadFromGPU(); );
        }

        dim3 ShapeBlock() const { return arrays[0]->ShapeBlock(); }
        dim3 ShapeThread() const { return arrays[0]->ShapeThread(); }

        size_t GetSize() const { return size; };

        Type SumGPU()
        {
                Type s = 0;
                MGPULOOP( s += arrays[g]->SumGPU(); );
                return s;
        }
        Type SumCPU()
        {
                MGPULOOP( arrays[g]->LoadFromGPU(); );
                MGPULOOP( cudaDeviceSynchronize(); );

                Type s = 0;
                MGPULOOP( s += arrays[g]->SumCPU(); );
                return s;
        }

        void FFTShift1()
        {
                MGPULOOP( arrays[g]->FFTShift1(); );
        }
        void FFTShift2()
        {
                MGPULOOP( arrays[g]->FFTShift2(); );
        }
        void FFTShift3()
        {
                MGPULOOP( arrays[g]->FFTShift3(); );
        }

        void SwapZ()
        {
                ErrorAssert(ngpus == 1 || (sizez%4 == 0 && ngpus==4), "Cant SwapZ!");

                if(ngpus == 4)
                {
                        for(int g=0; g<ngpus; g++){ Set(g); cudaDeviceSynchronize(); }

                        for(int g=0; g<GetN(); g++)
                        {
                                Set(g);
                                dim3 blk = arrays[g]->ShapeBlock();
                                dim3 thr = arrays[g]->ShapeThread();
                                KSwapZ_4gpu<<<blk,thr>>>(Ptr(0), Ptr(1), Ptr(2), Ptr(3), g, sizex);
                        }
                        MGPULOOP( cudaDeviceSynchronize(); );
                }
                else
                        arrays[0]->SwapZ();
        }

        void ReduceSync()
        {
                if(ngpus<2)
                        return;

                for(int s=1; s<ngpus; s*=2)
                {
                        for(int g=0; g<ngpus; g+=2*s) if(g+s < ngpus)
                        {
                                Set(g+s);
                                cudaDeviceSynchronize();
                                Set(g);
                                cudaDeviceSynchronize();

                                arrays[g][0] += arrays[g+s][0];
                        }
                }
                Set(0);
                cudaDeviceSynchronize();
        }
        void BroadcastSync()
        {
                if(ngpus<2)
                        return;

                int potgpus = ngpus-1; // convert ngpus to power-of-two
                potgpus |= potgpus>>1;
                potgpus |= potgpus>>2;
                potgpus |= potgpus>>4;
                potgpus += 1;

                for(int s = potgpus/2; s >= 1; s /= 2)
                {
                        for(int g=0; g<ngpus; g+=2*s) if(g+s < ngpus)
                        {
                                Set(g);
                                cudaDeviceSynchronize();

                                arrays[g]->CopyTo(arrays[g+s]->gpuptr);
                        }
                }

                MGPULOOP( cudaDeviceSynchronize(); );
        }

        void WeightedLerpSync(MImage<Type>& acc, MImage<float>& div, float lerp)
        {
                acc.ReduceSync();
                div.ReduceSync();

                Set(0);
                Sync::KWeightedLerp<<<ShapeBlock(),ShapeThread()>>>(Ptr(0), acc.Ptr(0), div.Ptr(0), size, lerp);

                BroadcastSync();
        }

        int __StupidLog2(int s)
        {
                int t;
                if(s==1)
                        t = 0;
                else if(s==2)
                        t = 1;
                else if(s==4)
                        t = 2;
                else if(s==8)
                        t = 3;
                else if(s==16)
                        t = 4;

                return t;
        }

        void ReduceSyncMasked(Image2D<uint32_t>** syncmask)
        {
                if(ngpus<2)
                        return;

                dim3 blk = ShapeBlock();
                dim3 thr = ShapeThread();

                for(int s=1; s<ngpus; s*=2)
                {
                        int t = __StupidLog2(s);

                        for(int g=0; g<ngpus; g+=2*s) if(g+s < ngpus)
                        {
                                Set(g+s);
                                cudaDeviceSynchronize();
                                Set(g);
                                cudaDeviceSynchronize();

                                //arrays[g][0] += arrays[g+s][0];
                                Sync::KMaskedSum<Type><<<blk,thr>>>(Ptr(g),Ptr(g+s),size,syncmask[t+g*4]->gpuptr);
                        }
                }
                Set(0);
                cudaDeviceSynchronize();
        }

        void BroadcastSyncMasked(Image2D<uint32_t>** syncmask)
        {
                if(ngpus<2)
                        return;

                int potgpus = ngpus-1; // convert ngpus to power-of-two
                potgpus |= potgpus>>1;
                potgpus |= potgpus>>2;
                potgpus |= potgpus>>4;
                potgpus += 1;

                dim3 blk = ShapeBlock();
                dim3 thr = ShapeThread();

                for(int s = potgpus/2; s >= 1; s /= 2)
                {
                        int t = __StupidLog2(s);

                        for(int g=0; g<ngpus; g+=2*s) if(g+s < ngpus)
                        {
                                Set(g);
                                cudaDeviceSynchronize();

                                //arrays[g]->CopyTo(arrays[g+s]->gpuptr);
                                Sync::KMaskedBroadcast<Type><<<blk,thr>>>(Ptr(g+s), Ptr(g), this->size, syncmask[t+g*4]->gpuptr);
                        }
                }

                MGPULOOP( cudaDeviceSynchronize(); );
        }

        Image2D<uint32_t>** GenSyncMask()
        {
                if(ngpus<2)
                        return nullptr;

                Image2D<uint32_t>** maskmatrix = new Image2D<uint32_t>*[4*ngpus];
                memset(maskmatrix, 0, 4*ngpus*sizeof(Image2D<uint32_t>*));

                for(int s=1; s<ngpus; s*=2)
                {
                        int t = __StupidLog2(s);

                        for(int g=0; g<ngpus; g+=2*s) if(g+s < ngpus)
                        {
                                Set(g);
                                maskmatrix[t+g*4] = new Image2D<uint32_t>((size+31)/32,1,1);
                        }
                }

                return maskmatrix;
        }
        void DeleteSyncMask(Image2D<uint32_t>** maskmatrix)
        {
                if(maskmatrix == nullptr)
                        return;

                for(int s=1; s<ngpus; s*=2)
                {
                        int t = __StupidLog2(s);

                        for(int g=0; g<ngpus; g+=2*s) if(g+s < ngpus) if(maskmatrix[t+g*4])
                        {
                                Set(g);
                                delete maskmatrix[t+g*4]; 
                                maskmatrix[t+g*4] = nullptr;
                        }
                }

                delete[] maskmatrix;
        }

        void UpdateSyncMask(Image2D<uint32_t>** maskmatrix, float thresh)
        {
                if(ngpus<2)
                        return;

                dim3 blk = dim3((this->size+31)/32,1,1);
                dim3 thr = dim3(32,1,1);

                for(int s=1; s<ngpus; s*=2)
                {
                        int t = __StupidLog2(s);

                        for(int g=0; g<ngpus; g+=2*s) if(g+s < ngpus)
                        {
                                Set(g);
                                maskmatrix[t+g*4]->SetGPUToZero();
                                cudaDeviceSynchronize();

                                Set(g+s);
                                cudaDeviceSynchronize();

                                Sync::KSetMask<Type><<<blk,thr>>>(maskmatrix[t+g*4]->gpuptr, Ptr(g+s), this->size, thresh);
                        }
                }

                MGPULOOP( cudaDeviceSynchronize(); );
        }

        void WeightedLerpSyncMasked(MImage<Type>& acc, MImage<float>& div, float lerp, Image2D<uint32_t>** syncmask)
        {
                acc.ReduceSyncMasked(syncmask);
                div.ReduceSyncMasked(syncmask);

                Set(0);
                Sync::KWeightedLerp<<<ShapeBlock(),ShapeThread()>>>(Ptr(0), acc.Ptr(0), div.Ptr(0), size, lerp);

                BroadcastSyncMasked(syncmask);
        }

        // Computes newval = oldval * (1-lerp) + acc/div * lerp. Useful for probe and object updates.

        void SyncDevices(){ for(int g : gpuindices){ cudaSetDevice(g); cudaDeviceSynchronize(); } };
};

typedef MImage<float> rMImage;
typedef MImage<complex> cMImage;




#endif