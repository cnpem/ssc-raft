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
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>

// #include <cublas.h>

#include "common/configs.hpp"
#include "common/logerror.hpp"

void getLog(float *data, dim3 size, cudaStream_t stream = 0);

__global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles);
    
static __global__ void Klog(float* data, dim3 size);

extern "C"{
    __global__ void padding(float *in, cufftComplex *inpadded, 
    float value, dim3 size, dim3 padsize);

    __global__ void recuperate_padding(cufftComplex *inpadded, float *in, 
    dim3 size, dim3 padsize);
}

namespace opt{
    inline __host__ __device__ int assert_dimension(int size1, int size2)
    { return ( size1 == size2 ? 1 : 0 ); };

    inline __host__ __device__ size_t getIndex(dim3 size)
    { 
        size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
        size_t j = blockIdx.y*blockDim.y + threadIdx.y; 
        size_t k = blockIdx.z*blockDim.z + threadIdx.z; 
        
        return IND(i,j,k,size.x,size.y); 
    };

	inline __host__ __device__ size_t getIndex3d(dim3 size)
    { 
        size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
        size_t j = blockIdx.y*blockDim.y + threadIdx.y; 
        size_t k = blockIdx.z*blockDim.z + threadIdx.z; 
        
        return IND(i,j,k,size.x,size.y); 
    };
    
    inline __host__ __device__ size_t getIndex2d(dim3 size)    
    { 
        int i = blockIdx.x*blockDim.x + threadIdx.x; 
        int j = blockIdx.y*blockDim.y + threadIdx.y; 
        
        return IND(i,j,0,size.x,size.y); 
    };

    inline __host__ __device__ int assert_dimension_xyz(dim3 size1, dim3 size2)
    { 
        int i;
        if ( assert_dimension(size1.x, size2.x) == 1 && 
             assert_dimension(size1.y, size2.y) == 1 && 
             assert_dimension(size1.z, size2.z) == 1
             ){ i = 1; }else{ if (size2.z == 1){ i = 2; }else{ i = 0; } }
        return i;
    };

    inline __host__ __device__ int assert_dimension_xy(dim3 size1, dim3 size2)
    { 
        int i;
        if ( assert_dimension(size1.x, size2.x) == 1 && 
             assert_dimension(size1.y, size2.y) == 1  
            ){ i = 1; }else{ i = 0; }
        return i;
    };

    inline __host__ __device__ int assert_dimension_xz(dim3 size1, dim3 size2)
    { 
        int i;
        if ( assert_dimension(size1.x, size2.x) == 1 && 
             assert_dimension(size1.z, size2.z) == 1
            ){ i = 1; }else{ i = 0; }
        return i;
    };

    inline __host__ __device__ int assert_dimension_yz(dim3 size1, dim3 size2)
    { 
        int i;
        if ( assert_dimension(size1.y, size2.y) == 1 && 
             assert_dimension(size1.z, size2.z) == 1
            ){ i = 1; }else{ i = 0; }
        return i;
    };

    inline __host__ __device__ size_t get_total_points(dim3 size)
    { size_t total_points = size.x * size.y * size.z; return total_points; };

    template<typename Type>
    __global__ void scale(Type *data, dim3 size, float scale)
    {
        size_t index        = opt::getIndex3d(size);
        size_t total_points = opt::get_total_points(size);

        if ( index >= total_points ) return;

        data[index] *= scale;
    };

    template<typename Type>
    Type* allocGPU(size_t size, cudaStream_t stream = 0)
    { Type *ptr; HANDLE_ERROR(cudaMallocAsync((void **)&ptr, size * sizeof(Type), stream)); return ptr; };

    template<typename Type>
    void CPUToGPU(Type *cpuptr, Type *gpuptr, size_t size, cudaStream_t stream = 0)
    {HANDLE_ERROR(cudaMemcpyAsync(gpuptr, cpuptr, size * sizeof(Type), cudaMemcpyHostToDevice, stream));};

    template<typename Type>
    void GPUToCPU(Type *cpuptr, Type *gpuptr, size_t size, cudaStream_t stream = 0)
    {HANDLE_ERROR(cudaMemcpyAsync(cpuptr, gpuptr, size * sizeof(Type), cudaMemcpyDeviceToHost, stream));};

    void MPlanFFT(cufftHandle *mplan, int RANK, dim3 DATASIZE, cufftType FFT_TYPE);

    dim3 setGridBlock(dim3 size);
    
    template<typename Type>
    __global__ void fftshift2D(Type *c, dim3 size)
    {
        Type temp; size_t shift;
        
        size_t N = ( (size.x * size.y) + size.x ) / 2 ;	
        size_t M = ( (size.x * size.y) - size.x ) / 2 ;	
        
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j = blockIdx.x*blockDim.x + threadIdx.x; 
        int k = blockIdx.x*blockDim.x + threadIdx.x; 

        size_t index = IND(i,j,k,size.x,size.y);

        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;
        
        if ( i < ( size.x / 2 ) ){	

            if ( j < ( size.y / 2 ) ){	
                shift = index + N;
                temp 	 = c[index];	
                c[index] = c[shift];	
                c[shift] = temp;
            }
        }else{
            if ( j < ( size.y / 2 ) ){
                shift = index + M;
                temp 	 = c[index];	
                c[index] = c[shift];	
                c[shift] = temp;
            }
        }
    };

    template<typename Type>
    __global__ void fftshift1D(Type *data, dim3 size)
    {
        Type temp;

        int i = blockIdx.x*blockDim.x + threadIdx.x;
        int j = blockIdx.x*blockDim.x + threadIdx.x; 
        int k = blockIdx.x*blockDim.x + threadIdx.x; 

        size_t index = IND(i,j,k,size.x,size.y);

        if (index < ( size.x * size.y * size.z / 2 )){
            if ( size.x / 2 <= i) {
                index += -size.x/2 + size.x * size.y * size.z/2;
                temp = data[index];
                data[index] = data[index + size.x/2];
                data[index + size.x/2] = temp;
            } else {
                temp = data[index];
                data[index] = data[index + size.x/2];
                data[index + size.x/2] = temp;
            }
        }
    };

    __global__ void product_Real_Real(float *a, float *b, float *ans, dim3 sizea, dim3 sizeb);
    __global__ void product_Complex_Real(cufftComplex *a, float *b, cufftComplex *ans, dim3 sizea, dim3 sizeb);
    __global__ void product_Complex_Complex(cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 sizea, dim3 sizeb);

     __global__ void paddR2C(float *in, cufftComplex *outpadded, 
    dim3 size, dim3 pad, float value);

    __global__ void paddC2C(cufftComplex *in, cufftComplex *outpadded,
    dim3 size, dim3 pad, float value);

    __global__ void paddC2R(cufftComplex *in, float *outpadded, 
    dim3 size, dim3 pad, float value);

    __global__ void paddR2R(float *in, float *outpadded, 
    dim3 size, dim3 pad, float value);

    __global__ void remove_paddC2R(cufftComplex *inpadded, float *out, 
    dim3 size, dim3 padsize);

    __global__ void remove_paddC2C(cufftComplex *inpadded, cufftComplex *out, 
    dim3 size, dim3 padsize);

    __global__ void remove_paddR2C(float *inpadded, cufftComplex *out, 
    dim3 size, dim3 padsize);

    __global__ void remove_paddR2R(float *inpadded, float *out, 
    dim3 size, dim3 pad);

    // template<typename Type>
    // __global__ void InnerProduct(Type *a, Type *b, float *norm, dim3 size)
    // {
    //     int i = blockIdx.x*blockDim.x + threadIdx.x;
    //     int j = blockIdx.y*blockDim.y + threadIdx.y;
    //     int k = blockIdx.z*blockDim.z + threadIdx.z;
    //     size_t index = IND(i,j,k,size.x,size.y);

    //     if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;
    //     extern __shared__ float temp[];
    //     // if sizex not multiple of 128 (e.g. gnomio), then fill temp with zeros,
    //     // beacuse threads will pass the image limits
    //     if (size.x < blockDim.x || size.y < blockDim.y || size.z < blockDim.z) {
    //         for (int ii = 0; ii < blockDim.x*blockDim.y*blockDim.z; ii++)
    //             temp[ii] = 0.0;
    //     }
    //     temp[blockDim.x*(threadIdx.z*blockDim.y + threadIdx.y) + threadIdx.x] = a[index] * b[index];
    //     __syncthreads();
    //     if (threadIdx.x == 0 && threadIdx.y == 0) { // for each slice in z, only one thread/block runs
    //         float sum = 0.0;
    //         for (int ii = 0; ii < blockDim.x*blockDim.y; ii++)
    //             sum += temp[ii + threadIdx.z*blockDim.x*blockDim.y];
    //         atomicAdd(norm + k, sum);
    //     }
    // }


    // template<typename Type>
    // float norm2_max(Type *data, dim3 size)
    // {
    //     float norm_calc;
    //     float *norm = opt::allocGPU<float>((size_t)size.z);

    //     dim3 threadsBlock = dim3(TPBX,TPBY,TPBZ);
    //     dim3 gridBlock    = opt::setGridBlock(size);

    //     opt::InnerProduct<<<gridBlock,threadsBlock>>>(data, data, norm, size);
    //     int max = cublasIsamax((int)size.z, norm, 1);
    //     HANDLE_ERROR(cudaMemcpy(&norm_calc, norm+max, sizeof(float), cudaMemcpyDeviceToHost));

    //     cudaFree(norm);
    //     return norm_calc;
    // }


}



#endif
