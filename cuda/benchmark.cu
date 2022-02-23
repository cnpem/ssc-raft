#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern "C" {
  __global__ void kernel_benchmark_scale_fftio(cufftComplex *array,
					       double factor,
					       int N,
					       int z)
  {
    int tx = blockIdx.x*blockDim.x + threadIdx.x; //N
    int ty = blockIdx.y*blockDim.y + threadIdx.y; //N
    int tz = blockIdx.z*blockDim.z + threadIdx.z; //blockSize

    int voxel;
    
    if( (tx< N) && (ty<N) && (tz < z) )
      {
	voxel = tz * N * N + (ty * N ) + tx;
	
	array[ voxel ].x = array[ voxel ].x * factor ;
	array[ voxel ].y = array[ voxel ].y * factor ;
      }
  }
}

extern "C" {
  __global__ void kernel_benchmark_set_fftio(cufftComplex *output,
					     float *input,
					     int N,
					     int z)
  {
    int tx = blockIdx.x*blockDim.x + threadIdx.x; //N
    int ty = blockIdx.y*blockDim.y + threadIdx.y; //N
    int tz = blockIdx.z*blockDim.z + threadIdx.z; //blockSize

    int voxel;
    
    if( (tx< N) && (ty<N) && (tz < z) )
      {
	voxel = tz * N * N + (ty * N ) + tx;
	
	output[ voxel ].x = input[ voxel ];
	output[ voxel ].y = 0;
      }
  }
}

extern "C" {
  __global__ void kernel_benchmark_get_real_fftio(float *output,
						  cufftComplex *input,
						  int N,
						  int z)
  {
    int tx = blockIdx.x*blockDim.x + threadIdx.x; //N
    int ty = blockIdx.y*blockDim.y + threadIdx.y; //N
    int tz = blockIdx.z*blockDim.z + threadIdx.z; //blockSize

    int voxel;
    
    if( (tx< N) && (ty<N) && (tz < z) )
      {
	voxel = tz * N * N + (ty * N ) + tx;
	
	output[ voxel ] = input[ voxel ].x ;
      }
  }
}

extern "C" {

  void ssc_benchmark_fftio_worker(ssc_benchmark_plan *workspace, int N, int z, float *elapsed){
    
    cudaEvent_t start,stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);	    
    cudaEventRecord(start);

    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
       
    dim3 gridBlock((int)ceil(N/threadsPerBlock.x)+1,
		       (int)ceil(N/threadsPerBlock.y)+1,
		       (int)ceil(z/threadsPerBlock.z)+1);

    // set input to fftio pointer
    kernel_benchmark_set_fftio<<<gridBlock, threadsPerBlock>>>(workspace->fftio,
							       workspace->input,
							       N,
							       z);

    
    if( cufftExecC2C(workspace->fftplan, workspace->fftio, workspace->fftio, CUFFT_FORWARD ) != CUFFT_SUCCESS)
      {
	fprintf(stderr,"Benchmark Error: CUFFT error. ExecC2C forward failed\n ");
	return;
      }
    
    if( cufftExecC2C(workspace->fftplan, workspace->fftio, workspace->fftio, CUFFT_INVERSE ) != CUFFT_SUCCESS)
      {
	fprintf(stderr,"Benchmark Error: CUFFT error. ExecC2C inverse failed\n ");
	return;
      }
    
    // scale to 1/N^2 
    kernel_benchmark_scale_fftio<<<gridBlock, threadsPerBlock>>>(workspace->fftio,
								 1.0/(N*N),
								 N,
								 z);
    
    
    // get input to fftio pointer
    kernel_benchmark_get_real_fftio<<<gridBlock, threadsPerBlock>>>(workspace->output,
								    workspace->fftio,
								    N,
								    z);
    
    checkCudaErrors(cudaPeekAtLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed,start,stop);
        
  }
}

extern "C" {
  void ssc_benchmark_fftio_create_plan(ssc_benchmark_plan *workspace, int N, int z){
    
    size_t voxels = N * N * z;
    
    checkCudaErrors( cudaMalloc((void**)&workspace->output, voxels*sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&workspace->input,  voxels*sizeof(float)) );
    checkCudaErrors(cudaMalloc((void**)&workspace->fftio, sizeof(cufftComplex)* voxels ));
    
    //Complex_2_Complex Plan
    int rank = 2;				//--- 2D FFTS
    int n[] = { N, N };		                //--- Size of the Fourier transform
    int istride = 0;
    int ostride = 0;			        //--- Distance between two successive input/output elements
    int idist = 0;
    int odist = 0;			        //--- Distance between Batches
    int batch = z;
    
    if( cufftPlanMany(&workspace->fftplan, rank, n,
		      NULL, istride, idist, 
		      NULL, ostride, odist, CUFFT_C2C, batch) != CUFFT_SUCCESS )
      {
	fprintf(stderr,"Benchmark: CUFFT error. Plan creation failed\n"); 
	return;
      }
  }
}

extern "C" {
  void ssc_benchmark_fftio_free_plan(ssc_benchmark_plan *workspace){
    
    cudaFree(workspace->output);
    cudaFree(workspace->input);
    cudaFree(workspace->fftio);
  }
}

extern "C" {
  void ssc_benchmark_fftio_set_data(ssc_benchmark_plan *workspace,
				   float *input,
				   int N,
				   int z,
				   float *elapsed){
    
    size_t voxels = (N * N * z);
    cudaEvent_t start,stop;

    //----------------------
    //copying data to device
    cudaEventCreate(&start);
    cudaEventCreate(&stop);	    
    cudaEventRecord(start);
   
    cudaMemcpy(workspace->input, input, voxels * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed,start,stop);
  }  
}

extern "C" {
  void ssc_benchmark_fftio_get_data(float *output,
				   ssc_benchmark_plan *workspace,
				   int N,
				   int z,
				   float *elapsed){
    
    size_t voxels = (N * N * z);
    cudaEvent_t start,stop;

    //----------------------
    //copying data to device
    cudaEventCreate(&start);
    cudaEventCreate(&stop);	    
    cudaEventRecord(start);
   
    cudaMemcpy(output, workspace->output, voxels * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsed,start,stop);
  }  
}


extern "C" {
  void ssc_benchmark_fftio(float *output,
			   float *input,
			   float *elapsed,
			   int N,
			   int z,
			   int device){

    ssc_benchmark_plan workspace;
    
    cudaSetDevice(device);
        
    ssc_benchmark_fftio_create_plan( &workspace, N, z );
	
    ssc_benchmark_fftio_set_data( &workspace, input, N, z, &elapsed[0]);
	
    ssc_benchmark_fftio_worker(&workspace, N, z, &elapsed[1]);
    
    ssc_benchmark_fftio_get_data(output, &workspace, N, z, &elapsed[2]);
    
    ssc_benchmark_fftio_free_plan( &workspace );

    //fprintf(stderr,"\nCopy Host to Device: \t\t %g",elapsed[0]/1000);
    //fprintf(stderr,"\nKernel Execution:    \t\t %g",elapsed[1]/1000);  
    //fprintf(stderr,"\nCopy Device to host: \t\t %g\n",elapsed[2]/1000);  
    
    cudaDeviceReset();
  }
}





