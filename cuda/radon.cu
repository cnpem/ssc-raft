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
  __global__ void radon_kernel(float *output, float *input,
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
      //output[tz * nrays * nangles + tx * nangles + ty] = linesum * dt;	
      output[tz * nrays * nangles + ty * nrays + tx] = linesum * dt;	
    }
  }
}


extern "C" {
  __global__ void radon_local_kernel(float* output, float *input,
				     int sizeImage, int nrays, int nangles,
				     int blockSize, int centerx, int centery)
  {
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
  
    if ( (tx<nrays) && (ty < nangles) && (tz<blockSize)  ){
      
      int k, X, Y;

      float a, s, x, y, linesum, ctheta, stheta, theta, t;
      float dxy = 2.0/(sizeImage-1); //, ds = 2.0*sqrtf(2.0)/(nrays-1);
      float dtheta = PI/(nangles-1);
      float xc, yc, T, S;

      a = ( nrays/2.0 ) * dxy;
    
      theta = ty * dtheta;
      ctheta =cosf(theta);
      stheta =sinf(theta);
      
      t = - a + tx * dxy;
      
      xc = -1 + centerx * dxy;
      yc = -1 + centery * dxy;
      T =  xc * ctheta + yc * stheta;
      S = -xc * stheta + yc * ctheta;
      
      linesum = 0;
      for( k = 0; k < sizeImage; k++ ) {
	s = - 1 + k * dxy;
	x = (t-T) * ctheta - (s-S) * stheta;
	y = (t-T) * stheta + (s-S) * ctheta;
	X = (int) ((x + 1)/dxy);
	Y = (int) ((y + 1)/dxy);
	if ((X >= 0) & (X<sizeImage) & (Y>=0) & (Y<sizeImage) )
	  linesum += input[ tz * sizeImage * sizeImage + Y * sizeImage + X ];
      }

      output[tz * nrays * nangles + ty * nrays + tx] = linesum * dxy;
    }
  }
}
                              


extern "C" {
  void radonp_gpu(float* h_output, float* h_input, int sizeImage, int nrays, int nangles, int device, int blocksize, float a)
  {
    cudaSetDevice(device);
    
    float *d_output, *d_input;
    
    // Allocate GPU buffers for the output sinogram
    cudaMalloc(&d_output, sizeof(float) * nrays * nangles * blocksize);
    cudaMalloc(&d_input, sizeof(float) * sizeImage * sizeImage * blocksize);
    cudaMemcpy(d_input, h_input, sizeof(float) * sizeImage * sizeImage * blocksize, cudaMemcpyHostToDevice);	
    
    //

    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlock((int)ceil((nrays)/threadsPerBlock.x)+1,
		   (int)ceil((nangles)/threadsPerBlock.y)+1,
		   (int)ceil(blocksize/threadsPerBlock.z)+1);
        
    radon_kernel<<<gridBlock, threadsPerBlock>>>(d_output, d_input, sizeImage, nrays, nangles, blocksize, a);
    
    cudaDeviceSynchronize();
    
    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(h_output, d_output, sizeof(float) * nrays * nangles * blocksize, cudaMemcpyDeviceToHost);
    
    //cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input);
    cudaDeviceReset();

    return;
  }
}

extern "C" {
  float radonp_ray(float *h_output, float* h_input,
		   int sizeImage, int nrays, int nangles,
		   float a, int i, int j)
  {
    int k, X, Y;
    float s, x, y, linesum, ctheta, stheta, theta, t;  
    float dt = 2.0*a/(nrays-1), ds = 2.0*sqrtf(2.0)/(nrays-1);
    float dtheta = PI/(nangles-1);
    float output;
    
    theta = i*dtheta;
    ctheta =cosf(theta);
    stheta =sinf(theta);
    
    t = - a + j * dt; 
    
    linesum = 0;
    for( k = 0; k < nrays; k++ )
      {
	s = -sqrtf(2.0) + k * ds; //- a + k * dt;
	x = t * ctheta - s * stheta;
	y = t * stheta + s * ctheta;
	X = (int) ((x + a)/dt);
	Y = (int) ((y + a)/dt);	     
	if ((X > -1) & (X<sizeImage) & (Y>-1) & (Y<sizeImage) ){
	  h_output[Y * sizeImage + X] = 1;
	  linesum += h_input[ Y * sizeImage + X ];
	}
      }
    output = linesum * dt;

    return output;
  }
}


extern "C" {
  void radonp_local_gpu(float* h_output, float* h_input, int sizeImage,
			int nrays, int nangles, int device, int blocksize, float a,
			int centerx, int centery)

  {
    cudaSetDevice(device);

    float *d_output, *d_input;

    // Allocate GPU buffers for the output sinogram
    cudaMalloc(&d_output, sizeof(float) * nrays * nangles * blocksize );
    cudaMalloc(&d_input, sizeof(float) * sizeImage * sizeImage * blocksize );
    cudaMemcpy(d_input, h_input, sizeof(float) * sizeImage * sizeImage * blocksize, cudaMemcpyHostToDevice);

    //
    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlock((int)ceil((nrays)/threadsPerBlock.x)+1,
		   (int)ceil((nangles)/threadsPerBlock.y)+1,
		   (int)ceil(blocksize/threadsPerBlock.z)+1);

    radon_local_kernel<<<gridBlock, threadsPerBlock>>>(d_output, d_input, sizeImage, nrays, nangles, blocksize, centerx, centery);

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(h_output, d_output, sizeof(float) * nrays * nangles * blocksize, cudaMemcpyDeviceToHost);

    //cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_input);
    cudaDeviceReset();

    return;
  }
}
