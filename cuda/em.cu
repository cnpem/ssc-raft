#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.141592653589793238462643383279502884

#define TPBX 16
#define TPBY 16
#define TPBZ 4
#define TPBE 256

extern "C" {
  __global__ void kernel_ones(float *output, int sizeImage, int nrays, int nangles,  int blockSize)
  {
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
  
    if ( (tx<sizeImage) && (ty < sizeImage) && (tz<blockSize)  )
      {
	int voxel = tz * sizeImage * sizeImage + ty * sizeImage + tx;
	
       	output[voxel] = 1;
      }
  }
}

extern "C" {
  __global__ void kernel_flatTimesExp(float *tmp, float *flat,
				      int sizeImage, int nrays, int nangles,  int blockSize)
  {
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
  
    if ( (tx<nrays) && (ty < nangles) && (tz<blockSize)  )
      {
	int voxel = tz * nrays * nangles + ty * nrays + tx;
	
       	tmp[voxel] = flat[voxel] * expf( - tmp[voxel]);	
      }
  }
}
    
extern "C" {
  __global__ void kernel_update(float *output, float *back, float *backcounts,
				int sizeImage, int nrays, int nangles,  int blockSize)
  {
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
  
    if ( (tx<sizeImage) && (ty < sizeImage) && (tz<blockSize)  )
      {
	int voxel = tz * sizeImage * sizeImage + ty * sizeImage + tx;
	
       	output[voxel] = output[voxel] * back[voxel] / backcounts[voxel];	
      }
  }
}


extern "C" {
  __global__ void kernel_backprojection(float *image, float *blocksino,
					int sizeImage, int nrays, int nangles,  int blockSize)
  {
    int i, j, k, T, z;
    float t, cs, x, y, cosk, sink;
    float xymin = -1.0;
    float dxy = 2.0 / (sizeImage - 1);
    float dt = 2.0 / (nrays - 1);
    float dth = PI / nangles;
    float tmin = -1.0;
    
    i = (blockDim.x * blockIdx.x + threadIdx.x);
    j = (blockDim.y * blockIdx.y + threadIdx.y);
    z = (blockDim.z * blockIdx.z + threadIdx.z);
  
    if ( (i<sizeImage) && (j < sizeImage) && (z<blockSize)  ){
    
      cs = 0;
    
      x = xymin + i * dxy;
      y = xymin + j * dxy;
    
      for(k=0; k < (nangles); k++)
	{
	  __sincosf(k * dth, &sink, &cosk);
	
	  t = x * cosk + y * sink;
	
	  T = (int) ((t - tmin)/dt);	     

	  if ( (T > -1) && (T<nrays) )
	    {
	      //cs = cs + blocksino[ T * nangles + k];
	      cs += blocksino[ z * sizeImage * sizeImage + k * nrays + T];
	    }
	}
      image[z * sizeImage * sizeImage + j * sizeImage + i]  = (cs*dth); 
    }
  }
}

extern "C" {
  __global__ void kernel_radon(float *output, float *input,
			       int sizeImage, int nrays, int nangles,
			       int blockSize, float a)
  {
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
  
    if ( (tx<nrays) && (ty < nangles) && (tz<blockSize)  ){
 
      int k, X, Y;

      float s, x, y, linesum, ctheta, stheta, theta, t;  
      float dt = 2.0*a/(nrays-1);
      float dtheta = PI/(nangles-1);
 
      theta = ty * dtheta;
      ctheta =cosf(theta);
      stheta =sinf(theta);
      
      t = - a + tx * dt; 
      
      linesum = 0;
      for( k = 0; k < nrays; k++ ) {
	s = - a + k * dt;
	x = t * ctheta - s * stheta;
	y = t * stheta + s * ctheta;
	X = (int) ((x + 1)/dt);
	Y = (int) ((y + 1)/dt);	 
	if ((X >= 0) & (X<sizeImage) & (Y>=0) & (Y<sizeImage) )
	  linesum += input[ tz * sizeImage * sizeImage + Y * sizeImage + X ];
      }

      output[tz * nrays * nangles + ty * nrays + tx] = linesum * dt;	
    }
  }
}

extern "C" {

  void EM(float *output, float *count, float *flat,
	  int sizeImage, int nrays, int nangles, int blockSize, int device, int niter)
  {
    cudaSetDevice(device);
    int k;
    float *d_output, *d_count, *d_flat, *d_backcounts, *d_temp, *d_back;
    
    // Allocate GPU memory for the output image
    cudaMalloc(&d_output, sizeof(float) *sizeImage *sizeImage*blockSize);
    cudaMalloc(&d_temp, sizeof(float)  * nrays * nangles*blockSize);
    cudaMalloc(&d_back, sizeof(float) *sizeImage *sizeImage*blockSize);
    cudaMalloc(&d_backcounts, sizeof(float) * nrays * nangles*blockSize);
    
    // Allocate GPU memory for input image and copy
    cudaMalloc(&d_count, sizeof(float) * nrays * nangles*blockSize);
    cudaMemcpy(d_count, count, sizeof(float) * nrays * nangles*blockSize, cudaMemcpyHostToDevice);	

    cudaMalloc(&d_flat, sizeof(float) * nrays * nangles*blockSize);
    cudaMemcpy(d_flat, flat, sizeof(float) * nrays * nangles*blockSize, cudaMemcpyHostToDevice);	

    
    //GRID and BLOCKS SIZE
    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlock((int)ceil((nrays)/threadsPerBlock.x)+1,
		   (int)ceil((nangles)/threadsPerBlock.y)+1,
		   (int)ceil(blockSize/threadsPerBlock.z)+1);

    kernel_ones<<<gridBlock, threadsPerBlock>>>(d_output, sizeImage, nrays, nangles, blockSize);

    kernel_backprojection<<<gridBlock, threadsPerBlock>>>(d_backcounts, d_count, sizeImage, nrays, nangles, blockSize);
 
    for( k=0; k < niter; k++ ) {

      kernel_radon<<<gridBlock, threadsPerBlock>>>(d_temp, d_output, sizeImage, nrays, nangles, blockSize, 1.0);
      
      kernel_flatTimesExp<<<gridBlock, threadsPerBlock>>>(d_temp, d_flat, sizeImage, nrays, nangles, blockSize);
      
      kernel_backprojection<<<gridBlock, threadsPerBlock>>>(d_back, d_temp, sizeImage, nrays, nangles, blockSize);
      
      kernel_update<<<gridBlock, threadsPerBlock>>>(d_output, d_back, d_backcounts, sizeImage, nrays, nangles, blockSize);
      
      cudaDeviceSynchronize();
    }
    
    //Copy the output image from device memory to host memory
    cudaMemcpy (output , d_output , blockSize*sizeImage*sizeImage*sizeof(float) , cudaMemcpyDeviceToHost);
    
    cudaFree(d_output);
    cudaFree(d_temp);
    cudaFree(d_back);
    cudaFree(d_count);
    cudaFree(d_flat);
    cudaFree(d_backcounts);
    
    cudaDeviceReset();
    
  }
}
