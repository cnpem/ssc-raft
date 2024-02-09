/**
@file opt.hpp
@author Paola Ferraz
@brief File contains kernels for the most basic operations
@version 0.1
@date 2024-02-09

@copyright Copyright (c) 2024

*/

#ifndef RAFT_OPT_H
#define RAFT_OPT_H

#include <cuda.h>
#include <cufft.h>

#include "common/configs.hpp"
#include "common/logerror.hpp"

void getLog(float *data, dim3 size);

__global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles)
    
static __global__ void Klog(float* data, dim3 size);

extern "C"{
    __global__ void padding(float *in, cufftComplex *inpadded, 
    float value, dim3 size, dim3 padsize);

    __global__ void recuperate_padding(cufftComplex *inpadded, float *in, 
    dim3 size, dim3 padsize);
}

namespace opt
{
    inline __host__ __device__ int assert_dimension(int size1, int size2){ return ( size1 == size2 ? 1 : 0 ); };

	inline __host__ __device__ size_t getIndex3d(dim3 size)
    { 
        size_t I = blockIdx.x*blockDim.x + threadIdx.x; 
        size_t J = blockIdx.y*blockDim.y + threadIdx.y; 
        size_t K = blockIdx.z*blockDim.z + threadIdx.z; 
        
        return IND(I,J,K,size.x,size.y); 
    };
    
    inline __host__ __device__ size_t getIndex2d(dim3 size)    
    { 
        int I = blockIdx.x*blockDim.x + threadIdx.x; 
        int J = blockIdx.y*blockDim.y + threadIdx.y; 
        
        return IND(I,J,0,size.x,size.y); 
    };

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
    Type* allocGPU(size_t size);

    template<typename Type>
    void CPUToGPU(Type *cpuptr, Type *gpuptr, size_t size);

    template<typename Type>
    void GPUToCPU(Type *cpuptr, Type *gpuptr, size_t size);

    void MPlanFFT(cufftHandle mplan, int dim, dim3 size);

    __global__ void product_Real_Real(float *a, float *b, float *ans, dim3 sizea, dim3 sizeb);
    __global__ void product_Complex_Real(cufftComplex *a, float *b, cufftComplex *ans, dim3 sizea, dim3 sizeb);
    __global__ void product_Complex_Complex(cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 sizea, dim3 sizeb);

     __global__ void paddR2C(float *in, cufftComplex *outpadded, 
    dim3 size, dim3 pad, float value)

    __global__ void paddC2C(cufftComplex *in, cufftComplex *outpadded,
    dim3 size, dim3 pad, float value)

    __global__ void paddC2R(cufftComplex *in, float *outpadded, 
    dim3 size, dim3 pad, float value)

    __global__ void paddR2R(float *in, float *outpadded, 
    dim3 size, dim3 pad, float value)

    __global__ void remove_paddC2R(cufftComplex *inpadded, float *out, 
    dim3 size, dim3 padsize, int dim);

    __global__ void remove_paddC2C(cufftComplex *inpadded, cufftComplex *out, 
    dim3 size, dim3 padsize, int dim);

    __global__ void remove_paddR2C(float *inpadded, cufftComplex *out, 
    dim3 size, dim3 padsize, int dim);

    __global__ void remove_paddR2R(float *inpadded, float *out, 
    dim3 size, dim3 pad, int dim);

}



#endif