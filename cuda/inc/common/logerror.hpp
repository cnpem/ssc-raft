#ifndef LOG_H
#define LOG_H

#include <cufft.h>
#include "cuComplex.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <string.h>
#include <helper_cuda.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <sstream>
#include <string>
#include <math.h>


#define cudaCheckError(){ \
	cudaError_t e = cudaGetLastError(); \
	if (e != cudaSuccess){ \
		fprintf(stderr," Cuda Failure %s: %d: '%s' \n", __FILE__,__LINE__, cudaGetErrorString(e)); \
		exit(0); \
	} \
}

#define gpuErrchk(ans) { gpuAssert((ans)); }
inline void gpuAssert(cudaError_t code)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"Cuda Failure %s %s %d\n", cudaGetErrorString(code), __FILE__,__LINE__);
        //if (abort) { getchar(); exit(code); }
        getchar(); exit(code); 
    }
}

#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
        fprintf(stderr, "CUFFT error in file '%s', line %d\n error %d: %s\nterminating!\n", file, line, err, _cudaGetErrorEnum(err)); \
        cudaDeviceReset(); assert(0); \
    }
}


inline void SaveLog(){};
#define LogB(x)
#define LogE()
#define SyncDebug

#define Log(message){ std::cout << message << std::endl; }

#define LogTbegin(file, message){ std::ofstream Timefile; Timefile.open(file, std::ofstream::out); Timefile << message ; Timefile.close();}

#define LogT(file, message){ std::ofstream Timefile; Timefile.open(file, std::ofstream::out | std::ofstream::app); Timefile << message ; Timefile.close();}

#define ErrorAssert(statement, message){ if(!(statement)){ std::cerr << __LINE__ << " in " << __FILE__ << ": " << message << "\n" << std::endl; Log(message); SaveLog(); exit(-1); } }

#define Warning(statement, message){ if(!(statement)){ std::cerr << __LINE__ << " in " << __FILE__ << ": " << message << "\n" << std::endl; Log(message); SaveLog(); } }

#define HANDLE_ERROR(errexp){ cudaError_t cudaerror = errexp; ErrorAssert( cudaerror == cudaSuccess, "Cuda error: " << std::string(cudaGetErrorString( cudaerror )) ) }

#define HANDLE_FFTERROR(errexp){ cufftResult fftres = errexp; ErrorAssert( fftres == CUFFT_SUCCESS, "Cufft error: " << std::string(_cudaGetErrorEnum( fftres )) ) }

//// cuFFT API errors
//static const char *_cudaGetErrorEnum(cufftResult error)
//{
//    switch (error)
//    {
//        case CUFFT_SUCCESS:
//            return "CUFFT_SUCCESS";
//
//        case CUFFT_INVALID_PLAN:
//            return "CUFFT_INVALID_PLAN";
//
//        case CUFFT_ALLOC_FAILED:
//            return "CUFFT_ALLOC_FAILED";
//
//        case CUFFT_INVALID_TYPE:
//            return "CUFFT_INVALID_TYPE";
//
//        case CUFFT_INVALID_VALUE:
//            return "CUFFT_INVALID_VALUE";
//
//        case CUFFT_INTERNAL_ERROR:
//            return "CUFFT_INTERNAL_ERROR";
//
//        case CUFFT_EXEC_FAILED:
//            return "CUFFT_EXEC_FAILED";
//
//        case CUFFT_SETUP_FAILED:
//            return "CUFFT_SETUP_FAILED";
//
//        case CUFFT_INVALID_SIZE:
//            return "CUFFT_INVALID_SIZE";
//
//        case CUFFT_UNALIGNED_DATA:
//            return "CUFFT_UNALIGNED_DATA";
//    }

//    return "<unknown>";
//}

#pragma once

// CUDA API error checking
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL

// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
    }
#endif  // CUFFT_CALL


#endif