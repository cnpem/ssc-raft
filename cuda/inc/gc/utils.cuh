#include <string>
#include <stdio.h>  // fprintf (function).
#include <stdlib.h> // stderr (poiter to error file).
#include "manager_types.cuh"

// CHECK_CUDA macro from: https://github.com/NVIDIA/multi-gpu-programming-models/blob/master/single_threaded_copy/jacobi.cu
#define CHECK_CUDA(call)                                               \
do {                                                                                           \
    bool is_cudaGetLastError = false;                                                          \
    cudaError_t cudaStatus = call;                                                             \
    checkCuda(cudaStatus, is_cudaGetLastError);                                                \
} while(0);

//inline
void checkCuda(cudaError_t cudaStatus, bool is_cudaGetLastError) {
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "ERROR: CUDA call in line %d of file %s failed with %s (cudaStatus=%d).\n",
            __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);
        if (is_cudaGetLastError) {
            if (cudaGetLastError() == cudaStatus) {
                throw CUDA_GOT_CONTEXT_CORRUPTOR_ERROR;
            }
            else {
                throw CUDA_ASYNC_ERROR;
            }
        }
        else {
            if (cudaGetLastError() != cudaSuccess) {
                if (cudaGetLastError() != cudaSuccess) {
                    throw CUDA_GOT_CONTEXT_CORRUPTOR_ERROR;
                }
                else {
                    throw CUDA_CALL_ERROR;
                }
            }
        }
    }
}