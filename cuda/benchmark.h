#ifndef SSC_BENCHMARK_H
#define SSC_BENCHMARK_H

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <cuda.h>
#include <cufft.h>
#include <cuComplex.h>
#include <cuda_runtime.h>


#define SSC_MIN( a, b ) ( ( ( a ) < ( b ) ) ? ( a ) : ( b ) )
#define SSC_MAX( a, b ) ( ( ( a ) > ( b ) ) ? ( a ) : ( b ) )
#define SSC_SQUARE(x) ((x)*(x))
#define SSC_SIGN( x ) ( ( ( x ) > 0.0 ) ? 1.0 : ( ( ( x ) < 0.0 ) ? -1.0 : 0.0 ) )
#define SSC_PI 3.141592653589793238462643383279502884

#define TPBX 16
#define TPBY 16
#define TPBZ 4

#define BILLION 1E9
#define CLOCK  CLOCK_REALTIME
#define TIME(End,Start) (End.tv_sec - Start.tv_sec) + (End.tv_nsec-Start.tv_nsec)/BILLION

typedef struct{
  
  float *output;
  float *input;
  cufftComplex *fftio;
  cufftHandle fftplan;
  
}ssc_benchmark_plan;

#ifdef __cplusplus
extern "C" {
#endif

  // prototypes

  void ssc_benchmark_fftio_worker(ssc_benchmark_plan *workspace, int N, int z, float *elapsed);

  void ssc_benchmark_fftio_create_plan(ssc_benchmark_plan *workspace, int N, int z);

  void ssc_benchmark_fftio_free_plan(ssc_benchmark_plan *workspace);
  
  void ssc_benchmark_fftio_set_data(ssc_benchmark_plan *workspace,
				    float *input,
				    int N,
				    int z,
				    float *elapsed);
  
  void ssc_benchmark_fftio_get_data(float *output,
				    ssc_benchmark_plan *workspace,
				    int N,
				    int z,
				    float *elapsed);
  
  void ssc_benchmark_fftio(float *output,
			   float *input,
			   float *elapsed,
			   int N,
			   int z,
			   int device);

#ifdef __cplusplus
} // extern "C" {
#endif

#endif // #ifndef SSC_BENCHMARK_H

