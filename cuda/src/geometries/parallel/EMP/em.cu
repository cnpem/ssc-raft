// #include <cuda.h>
// #include <cuda_runtime_api.h>
// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <cublas_v2.h>

#include "../../../../inc/sscraft.h"

#define PI 3.141592653589793238462643383279502884

#define TPBX 16
#define TPBY 16
#define TPBZ 4
#define TPBE 256

#define SQR(x) ((x)*(x))
#define SIGN(x) ((x > 0) ? 1 : ((x < 0) ? -1 : 0))
#define APPROXINVX(x,e) ((SIGN(x))/(sqrtf( SQR(e) + SQR(x) )))

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
  // difference in the Feature domain: square images of order 'sizeImage'
  // inplace: y = y - x
  
  __global__ void kernel_difference_F(float *y, float *x,
				      int sizeImage, int blockSize)
  {
    int tx = threadIdx.x + blockIdx.x*blockDim.x;
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
  
    if ( (tx<sizeImage) && (ty < sizeImage) && (tz<blockSize)  )
      {
	      int voxel = tz * sizeImage * sizeImage + ty * sizeImage + tx;

	      // inplace!
       	y[voxel] = y[voxel] - x[voxel];
      }
  }
}


extern "C" {
  __global__ void kernel_backprojection(float *image, float *blocksino, float *angles,
					int sizeImage, int nrays, int nangles,  int blockSize)
  {
    int i, j, k, T, z;
    float t, cs, x, y, cosk, sink;
    float xymin = -1.0;
    float dxy = 2.0 / (sizeImage - 1);
    float dt = 2.0 / (nrays - 1);
    // float dth = PI / nangles;
    float dth;
    float tmin = -1.0;
    
    i = (blockDim.x * blockIdx.x + threadIdx.x);
    j = (blockDim.y * blockIdx.y + threadIdx.y);
    z = (blockDim.z * blockIdx.z + threadIdx.z);
    
    // printf("nangles = %d \n",nangles);

    if ( (i<sizeImage) && (j < sizeImage) && (z<blockSize)  ){
    
      cs = 0;
    
      x = xymin + i * dxy;
      y = xymin + j * dxy;

      for(k=0; k < (nangles); k++)
      {
        // __sincosf(k * dth, &sink, &cosk);
        __sincosf(angles[k], &sink, &cosk);

        if ( k == (nangles - 1) ){
          dth = angles[k] - angles[k-1];
        }else{
          dth = angles[k+1] - angles[k];
        }
        
        // printf("dth[%d] = %e \n",k,dth);

        t = x * cosk + y * sink;
      
        T = (int) ((t - tmin)/dt);	     

        if ( (T > -1) && (T<nrays) )
          {
            cs += blocksino[ z * nrays * nangles + k * nrays + T] * dth;
          }
      }
      image[z * sizeImage * sizeImage + j * sizeImage + i]  = cs; 
    }
  }
}

extern "C" {
  __global__ void kernel_radon(float *output, float *input, float *angles,
			       int sizeImage, int nrays, int nangles,
			       int blockSize, float a)
  {
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
    

    if ( (tx<nrays) && (ty < nangles) && (tz<blockSize)  ){
 
      int k, X, Y;

      float s, x, y, linesum, ctheta, stheta, t;  
      float dt = 2.0*a/(nrays-1);
      // float dtheta = PI/(nangles-1);
 
      // theta = ty * dtheta;
      ctheta =cosf(angles[ty]);
      stheta =sinf(angles[ty]);
      
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
      // printf("out[%d] = %e \n",tz * nrays * nangles + ty * nrays + tx,output[tz * nrays * nangles + ty * nrays + tx]);
    }
  }
}

//---------------------------
// transmission-EM algorithm
//---------------------------

extern "C" {

  void tEM(float *output, float *count, float *flat, float *angles, 
	  int sizeImage, int nrays, int nangles, int blockSize, int device, int niter)
  {
    cudaSetDevice(device);
    int k;
    float *d_output, *d_count, *d_flat, *d_backcounts, *d_temp, *d_back, *d_angles;
    
    // Allocate GPU memory for the output image
    cudaMalloc(&d_output, sizeof(float) *sizeImage *sizeImage*blockSize);
    cudaMalloc(&d_temp, sizeof(float)  * nrays * nangles*blockSize);
    cudaMalloc(&d_back, sizeof(float) *sizeImage *sizeImage*blockSize);
    cudaMalloc(&d_backcounts, sizeof(float) * sizeImage * sizeImage*blockSize);
    
    // Allocate GPU memory for input image and copy
    cudaMalloc(&d_count, sizeof(float) * nrays * nangles*blockSize);
    cudaMemcpy(d_count, count, sizeof(float) * nrays * nangles*blockSize, cudaMemcpyHostToDevice);	

    cudaMalloc(&d_flat, sizeof(float) * nrays * nangles*blockSize);
    cudaMemcpy(d_flat, flat, sizeof(float) * nrays * nangles*blockSize, cudaMemcpyHostToDevice);	

    cudaMalloc(&d_angles, sizeof(float) * nangles);
    cudaMemcpy(d_angles, angles, sizeof(float) * nangles, cudaMemcpyHostToDevice);	

    // printf("Printando essa porra + %d, %d, %e\n",device, nangles, count[1024]);
    // for(int i=0; i < nangles; i++) printf("Angle[%d] = %e \n",i,angles[i]);

    //GRID and BLOCKS SIZE
    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlockD((int)ceil((nrays)/threadsPerBlock.x)+1,
		   (int)ceil((nangles)/threadsPerBlock.y)+1,
		   (int)ceil(blockSize/threadsPerBlock.z)+1);

    dim3 gridBlockF((int)ceil((sizeImage)/threadsPerBlock.x)+1,
		   (int)ceil((sizeImage)/threadsPerBlock.y)+1,
		   (int)ceil(blockSize/threadsPerBlock.z)+1);

    
    kernel_ones<<<gridBlockF, threadsPerBlock>>>(d_output, sizeImage, nrays, nangles, blockSize);

    kernel_backprojection<<<gridBlockF, threadsPerBlock>>>(d_backcounts, d_count, d_angles, sizeImage, nrays, nangles, blockSize);
 
    for( k=0; k < niter; k++ )
      {
      // printf("Aqui iter = %d \n",k);
      kernel_radon<<<gridBlockD, threadsPerBlock>>>(d_temp, d_output, d_angles, sizeImage, nrays, nangles, blockSize, 1.0);
      
      kernel_flatTimesExp<<<gridBlockD, threadsPerBlock>>>(d_temp, d_flat, sizeImage, nrays, nangles, blockSize);
      
      kernel_backprojection<<<gridBlockF, threadsPerBlock>>>(d_back, d_temp, d_angles, sizeImage, nrays, nangles, blockSize);
      
      kernel_update<<<gridBlockF, threadsPerBlock>>>(d_output, d_back, d_backcounts, sizeImage, nrays, nangles, blockSize);
      
      cudaDeviceSynchronize();
    }
    
    //Copy the output image from device memory to host memory
    cudaMemcpy (output , d_output , blockSize*sizeImage*sizeImage*sizeof(float) , cudaMemcpyDeviceToHost);
    
    // printf("Printando out + %d, %d, %e\n",device, nangles, output[1024]);

    cudaFree(d_output);
    cudaFree(d_temp);
    cudaFree(d_back);
    cudaFree(d_count);
    cudaFree(d_flat);
    cudaFree(d_backcounts);
    cudaFree(d_angles);
    
    cudaDeviceSynchronize();
    // cudaDeviceReset();
    
  }
}

//----------------------
// emission-EM algorithm
//----------------------

extern "C" {
  __global__ void kernel_radonWithDivision(float *output, float *input, float *sino,  float *angles,
					   int sizeImage, int nrays, int nangles,
					   int blockSize, float a)
  {
    float TOLZERO;
    
    int tx = threadIdx.x + blockIdx.x*blockDim.x; 
    int ty = threadIdx.y + blockIdx.y*blockDim.y; 
    int tz = threadIdx.z + blockIdx.z*blockDim.z;
  
    if ( (tx<nrays) && (ty < nangles) && (tz<blockSize)  ){
 
      int k, X, Y, voxel;

      float s, x, y, linesum, ctheta, stheta, t;  
      float dt = 2.0*a/(nrays-1);
      // float dtheta = PI/(nangles-1);
      float value;
      
      // theta = ty * dtheta;
      ctheta =cosf(angles[ty]);
      stheta =sinf(angles[ty]);
      
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

      value = linesum * dt;

      voxel = tz * nrays * nangles + ty * nrays + tx;

      TOLZERO = 0.0001;
      output[ voxel ] = sino[ voxel] * APPROXINVX(value, TOLZERO);  

      //enforcing positivity
      if (output[voxel] < 0)
	      output[voxel] = 0.0;

      
      /*
      if ( fabs(value) > TOLZERO ) 
	      output[ voxel ] = sino[voxel] / value;	
      else
	      output[ voxel ] = 0.0;
      */
    }
  }
}

extern "C" {
  __global__ void kernel_backprojectionWithUpdate(float *image, float *blocksino, float *backones, float *angles,
						  int sizeImage, int nrays, int nangles,  int blockSize)
  {
    int i, j, k, T, z, voxel;
    float t, cs, x, y, cosk, sink;
    float xymin = -1.0;
    float dxy = 2.0 / (sizeImage - 1);
    float dt = 2.0 / (nrays - 1);
    float dth; // = PI / nangles;
    float tmin = -1.0;
    float value; 
    
    i = (blockDim.x * blockIdx.x + threadIdx.x);
    j = (blockDim.y * blockIdx.y + threadIdx.y);
    z = (blockDim.z * blockIdx.z + threadIdx.z);
  
    if ( (i<sizeImage) && (j < sizeImage) && (z<blockSize)  ){
    
      cs = 0;
    
      x = xymin + i * dxy;
      y = xymin + j * dxy;
    
      for(k=0; k < (nangles); k++)
      {
        // __sincosf(k * dth, &sink, &cosk);
        __sincosf(angles[k], &sink, &cosk);
        
        if ( k == (nangles - 1) ){
          dth = angles[k] - angles[k-1];
        }else{
          dth = angles[k+1] - angles[k];
        }

        t = x * cosk + y * sink;
      
        T = (int)((t - tmin)/dt);	     
         
        if ( (T > -1) && (T<nrays) )
        {
          cs += blocksino[ z * nrays * nangles + k * nrays + T] * dth;
        }
      }

      voxel = z * sizeImage * sizeImage + j * sizeImage + i;

      value = cs;
      
      image[ voxel ]  = image[ voxel ] * value / backones[ voxel ]; 
    }
  }
}

extern "C" {
  __global__ void kernel_backprojectionOfOnes(float *backones, float *angles,
					      int sizeImage, int nrays, int nangles,  int blockSize)
  {
    int i, j, k, T, z, voxel;
    float t, cs, x, y, cosk, sink;
    float xymin = -1.0;
    float dxy = 2.0 / (sizeImage - 1);
    float dt = 2.0 / (nrays - 1);
    float dth; // = PI / nangles;
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
        // __sincosf(k * dth, &sink, &cosk);
        __sincosf(angles[k], &sink, &cosk);
        
        if ( k == (nangles - 1) ){
          dth = angles[k] - angles[k-1];
        }else{
          dth = angles[k+1] - angles[k];
        }

        t = x * cosk + y * sink;
      
        T = (int) ((t - tmin)/dt);	     

        if ( (T > -1) && (T<nrays) )
          {
            cs += dth; //blocksino[ z * nrays * nangles+ k * nrays + T];
          }
      }
      
      voxel = z * sizeImage * sizeImage + j * sizeImage + i;
      
      backones[ voxel ]  = cs; 
    }
  }
}

extern "C" {

  void eEM(float *output, float *sino, float *angles, 
	   int sizeImage, int nrays, int nangles, int blockSize, int device, int niter)
  {
    cudaSetDevice(device);
    int k;
    float *d_output, *d_sino, *d_backones, *d_temp, *d_ones, *d_angles;
    
    // Allocate GPU memory for the output image
    cudaMalloc(&d_output, sizeof(float) *sizeImage *sizeImage*blockSize);
    cudaMalloc(&d_temp, sizeof(float)  * nrays * nangles * blockSize);
    cudaMalloc(&d_backones, sizeof(float) * sizeImage * sizeImage*blockSize);
    cudaMalloc(&d_ones, sizeof(float) * nrays * nangles * blockSize);
    
    // Allocate GPU memory for input image and copy
    cudaMalloc(&d_sino, sizeof(float) * nrays * nangles*blockSize);
    cudaMemcpy(d_sino, sino, sizeof(float) * nrays * nangles*blockSize, cudaMemcpyHostToDevice);	

    cudaMalloc(&d_angles, sizeof(float) * nangles);
    cudaMemcpy(d_angles, angles, sizeof(float) * nangles, cudaMemcpyHostToDevice);	
    
    // printf("Printando essa porra + %d, %d, %e\n",device, nangles, sino[1024]);

    //GRID and BLOCKS SIZE
    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlockD((int)ceil((nrays)/threadsPerBlock.x)+1,
		   (int)ceil((nangles)/threadsPerBlock.y)+1,
		   (int)ceil(blockSize/threadsPerBlock.z)+1);

    dim3 gridBlockF((int)ceil((sizeImage)/threadsPerBlock.x)+1,
		   (int)ceil((sizeImage)/threadsPerBlock.y)+1,
		   (int)ceil(blockSize/threadsPerBlock.z)+1);

    kernel_ones<<<gridBlockF, threadsPerBlock>>>(d_output, sizeImage, nrays, nangles, blockSize);

    kernel_backprojectionOfOnes<<<gridBlockF, threadsPerBlock>>>(d_backones, d_angles, sizeImage, nrays, nangles, blockSize);
 
    for( k=0; k < niter; k++ ) {

      kernel_radonWithDivision<<<gridBlockD, threadsPerBlock>>>(d_temp, d_output, d_sino, d_angles, sizeImage, nrays, nangles, blockSize, 1.0);
            
      kernel_backprojectionWithUpdate<<<gridBlockF, threadsPerBlock>>>(d_output, d_temp, d_backones, d_angles, sizeImage, nrays, nangles, blockSize);
      
      cudaDeviceSynchronize();
    }
    
    //Copy the output image from device memory to host memory
    cudaMemcpy (output , d_output , blockSize*sizeImage*sizeImage*sizeof(float) , cudaMemcpyDeviceToHost);
    
    cudaFree(d_output);
    cudaFree(d_temp);
    cudaFree(d_sino);
    cudaFree(d_backones);
    cudaFree(d_angles);
    
    cudaDeviceSynchronize();
    // cudaDeviceReset();
    
  }
}

//-----------------------------------------
// emission-EM + Total Variation  algorithm
//------------------------------------------

extern "C" {
  // y : next iterate
  // x : current iterate
  
  __global__ void kernel_updateTV(float *y, float *x, float *backones,
				  int sizeImage, int blockSize,
				  float reg, float epsilon)
  {
    int i, j, z, v, vip1, vjp1, vjm1, vim1, vjm1ip1, vjp1im1;
    float sqrtA, sqrtB, sqrtD;
    float A, B, C, D, rhs;
    
    i = (blockDim.x * blockIdx.x + threadIdx.x);
    j = (blockDim.y * blockIdx.y + threadIdx.y);
    z = (blockDim.z * blockIdx.z + threadIdx.z);
  
    if ( ((i+1)<sizeImage) && ((j+1) < sizeImage) && (z<blockSize) &&
	 ((i-1)<sizeImage) && ((j-1) < sizeImage) ){

      // i: column (axis=1 from python)
      // j: row    (axis=0 from python)

      v    = z * sizeImage * sizeImage + j * sizeImage + i;
      
      vip1 = z * sizeImage * sizeImage + j * sizeImage + (i+1);
      vjp1 = z * sizeImage * sizeImage + (j+1) * sizeImage + i;
      
      sqrtA = epsilon + SQR( y[vjp1] - y[v]) + SQR( y[vip1] - y[v] ); //ok
      A     = - reg * backones[v] * sqrtf(sqrtA);

      vjm1  = z * sizeImage * sizeImage + (j-1) * sizeImage + i;
      vjm1ip1 = z * sizeImage * sizeImage + (j-1) * sizeImage + (i+1);
   
      sqrtB = epsilon * SQR( y[v] - y[vjm1]) + SQR( y[vjm1ip1] - y[vjm1]); //ok
      B     = reg * backones[v] * sqrtf(sqrtB);
      C     = A;

      vjp1im1 =  z * sizeImage * sizeImage + (j+1) * sizeImage + (i-1);
      vjm1    =  z * sizeImage * sizeImage + (j-1) * sizeImage + i;
      vim1    = z * sizeImage * sizeImage + j * sizeImage + (i-1);

      sqrtD = epsilon * SQR( y[vjp1im1] - y[vim1]) + SQR(y[v] - y[vim1]); //ok
      D     = reg * backones[v] * sqrtf(sqrtD); 
      
      rhs = x[v] - y[v] * ( y[vjp1]/A - y[vjm1]/B + y[vip1]/C - y[vim1]/D );
      
      //update!
      float TOLZERO = 1e-6; 
      y[ v ] =  rhs * APPROXINVX( y[v] * ( -1.0/A + 1.0/B - 1.0/C + 1.0/D ) + 1.0, TOLZERO );
    }
    
  }
}

extern "C" {
  // L2-error in the feature domain
  void getError_F(float *error, float *x, float *y, int N, int blockSize, int device)
  {
    cublasHandle_t handle;
    cublasStatus_t stat;

    //GRID and BLOCKS SIZE
    dim3 threads(TPBX,TPBY,TPBZ);
    dim3 gridF((int)ceil((N)/threads.x)+1,
	       (int)ceil((N)/threads.y)+1,
	       (int)ceil(blockSize/threads.z)+1);
    
    
    // inplace: x = x - y
    kernel_difference_F<<<gridF, threads>>>(x, y, N, blockSize);

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("ssc-raft: CUBLAS initialization failed\n");
        exit(EXIT_FAILURE);
    }
    
    stat = cublasSnrm2(handle, N * N * blockSize, x, 1, error);
    if(stat != CUBLAS_STATUS_SUCCESS){
      printf("ssc-raft: Error code %d, line(%d)\n", stat, __LINE__);
      exit(EXIT_FAILURE);
    }
    
    cublasDestroy(handle);
  }
  
}

extern "C" {
  void iterEM( float *em,
	       float *sino, float *sinotmp, float *backones, float *angles,
	       int sizeImage, int nrays, int nangles, int blockSize, int device )
  {
    //GRID and BLOCKS SIZE
    dim3 threads(TPBX,TPBY,TPBZ);
    dim3 gridD((int)ceil((nrays)/threads.x)+1,
	       (int)ceil((nangles)/threads.y)+1,
	       (int)ceil(blockSize/threads.z)+1);

    dim3 gridF((int)ceil((sizeImage)/threads.x)+1,
	       (int)ceil((sizeImage)/threads.y)+1,
	       (int)ceil(blockSize/threads.z)+1);
    
    kernel_radonWithDivision<<<gridD, threads>>>(sinotmp, em, sino, angles, sizeImage, nrays, nangles, blockSize, 1.0);	

    kernel_backprojectionWithUpdate<<<gridF, threads>>>(em, sinotmp, backones, angles, sizeImage, nrays, nangles, blockSize);
  }
}

extern "C" {
  void iterTV( float *y, float *x, float *backones,
	       int sizeImage, int blockSize, int device, float reg, float epsilon)
  {
    //GRID and BLOCKS SIZE
    dim3 threads(TPBX,TPBY,TPBZ);
    dim3 gridF((int)ceil((sizeImage)/threads.x)+1,
		    (int)ceil((sizeImage)/threads.y)+1,
		    (int)ceil(blockSize/threads.z)+1);
    
    kernel_updateTV<<<gridF, threads>>>(y, x, backones, sizeImage, blockSize, reg, epsilon);
  }
}

extern "C" {
  void EMTV(float *output, float *sino, float *angles, 
	    int sizeImage, int nrays, int nangles, int blockSize, int device, int niter,
	    int niter_em, int niter_tv, float reg, float epsilon)
  {
    cudaSetDevice(device);
 
    int m, k;
    float *d_em, *d_x, *d_y, *d_backones, *d_sino, *d_sinotmp, *d_angles;
    float error, _error_;
    
    // Allocate GPU memory for the output image
    cudaMalloc(&d_em, sizeof(float) *sizeImage *sizeImage*blockSize);
    cudaMalloc(&d_x,  sizeof(float) *sizeImage *sizeImage*blockSize);
    cudaMalloc(&d_y,  sizeof(float) *sizeImage *sizeImage*blockSize);
    cudaMalloc(&d_backones, sizeof(float) * sizeImage * sizeImage*blockSize);
    
    cudaMalloc(&d_sinotmp, sizeof(float)  * nrays * nangles * blockSize);
    cudaMalloc(&d_sino, sizeof(float) * nrays * nangles*blockSize);
    cudaMemcpy(d_sino, sino, sizeof(float) * nrays * nangles*blockSize, cudaMemcpyHostToDevice);	

    cudaMalloc(&d_angles, sizeof(float) * nangles);
    cudaMemcpy(d_angles, angles, sizeof(float) * nangles, cudaMemcpyHostToDevice);	
    
    //GRID and BLOCKS SIZE
    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlockD((int)ceil((nrays)/threadsPerBlock.x)+1,
		    (int)ceil((nangles)/threadsPerBlock.y)+1,
		    (int)ceil(blockSize/threadsPerBlock.z)+1);

    dim3 gridBlockF((int)ceil((sizeImage)/threadsPerBlock.x)+1,
		    (int)ceil((sizeImage)/threadsPerBlock.y)+1,
		    (int)ceil(blockSize/threadsPerBlock.z)+1);
    
    
    kernel_ones<<<gridBlockF, threadsPerBlock>>>(d_em, sizeImage, nrays, nangles, blockSize);

    // temp assignment to d_x in order to computer initial error estimate!
    cudaMemcpy(d_x, d_em, sizeof(float) * sizeImage * sizeImage * blockSize, cudaMemcpyDeviceToDevice);
    
    kernel_backprojectionOfOnes<<<gridBlockF, threadsPerBlock>>>(d_backones, d_angles, sizeImage, nrays, nangles, blockSize);

    iterEM( d_em, d_sino, d_sinotmp, d_backones, d_angles, 
	    sizeImage, nrays, nangles, blockSize, device);

    getError_F(&error, d_x, d_em, sizeImage, blockSize, device);
    
    for (m = 0; m < niter; m++)
      {
	//get d_x pointer
	cudaMemcpy(d_x, d_em, sizeof(float) * sizeImage * sizeImage * blockSize, cudaMemcpyDeviceToDevice);

	//EM iterations
	for( k = 0; k < niter_em; k++ )
	  {
	    iterEM( d_em, d_sino, d_sinotmp, d_backones, d_angles, 
		    sizeImage, nrays, nangles, blockSize, device);
	    
	    cudaDeviceSynchronize();
	  }
	
	//TV iterations
	cudaMemcpy(d_y, d_em, sizeof(float) * sizeImage * sizeImage * blockSize, cudaMemcpyDeviceToDevice);
	
	for (k = 0; k < niter_tv; k++ )
	  {
	    iterTV( d_y, d_em, d_backones, 
		    sizeImage, blockSize, device, reg, epsilon);

	    cudaDeviceSynchronize();
	  }

	cudaMemcpy( d_em, d_y, sizeof(float) * sizeImage * sizeImage * blockSize, cudaMemcpyDeviceToDevice);
       
	getError_F(&_error_, d_x, d_em, sizeImage, blockSize, device);
       
	fprintf(stdout,"EM+TV: %lf %lf\n", _error_, error);
	if (_error_ < error)
	  error = _error_;
	else
	  break;
	
    }
    
    //Copy the output image from device memory to host memory
    cudaMemcpy (output , d_em , blockSize*sizeImage*sizeImage*sizeof(float) , cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_em);
    cudaFree(d_backones);
    cudaFree(d_sinotmp);
    cudaFree(d_sino);
    cudaFree(d_angles);
    
    cudaDeviceSynchronize();
    // cudaDeviceReset();
    
  }
}

//----------------------
// emission-EM Threads Block algorithm
//----------------------

extern "C"{   

  void eEMgpu(float *output, float *sino, float *angles, 
        int sizeImage, int nrays, int nangles, int nslices, int gpu, int niter)
  {

      size_t blocksize = min((size_t)nslices,32ul);

      for(size_t b = 0; b < nslices; b += blocksize){
          
          blocksize = min(size_t(nslices) - b, blocksize);
          // printf("Nslices: %d, blocksize: %ld, Iter: %ld \n", nslices,blocksize,b);


          eEM(output + (size_t)b*sizeImage*sizeImage, sino + (size_t)b*nrays*nangles, angles, sizeImage, nrays, nangles, blocksize, gpu, niter);          
      }

      cudaDeviceSynchronize();

  }

  void eEMblock(float *output, float *sino, float *angles, 
    int sizeImage, int nrays, int nangles, int nslices, int ngpus, int niter, int* gpus)
  {
      int t;
      int blockgpu = (nslices + ngpus - 1) / ngpus;
      
      std::vector<std::future<void>> threads;

      for(t = 0; t < ngpus; t++){ 
          
          blockgpu = min(nslices - blockgpu * t, blockgpu);

          threads.push_back(std::async( std::launch::async, eEMgpu, output + (size_t)t * blockgpu * sizeImage*sizeImage, 
          sino + (size_t)t * blockgpu * nrays * nangles, angles, sizeImage, nrays, nangles, blockgpu,  gpus[t], niter
          ));
      }
  
      for(auto& t : threads)
          t.get();
  }

}

//----------------------
// transmission-EM Threads Block algorithm
//----------------------

extern "C"{   

  void tEMgpu(float *output, float *count, float *flat, float *angles, 
        int sizeImage, int nrays, int nangles, int nslices, int gpu, int niter)
  {

      size_t blocksize = min((size_t)nslices,32ul);

      for(size_t b = 0; b < nslices; b += blocksize){
          blocksize = min(size_t(nslices) - b, blocksize);
          // printf("Nslices: %d, blocksize: %ld, Iter: %ld \n", nslices,blocksize,b);

          tEM(output + (size_t)b*sizeImage*sizeImage, count + (size_t)b*nrays*nangles, flat + (size_t)b*nrays*nangles, angles, sizeImage, nrays, nangles, blocksize, gpu, niter);          
      }

      cudaDeviceSynchronize();

  }

  void tEMblock(float *output, float *count, float *flat, float *angles, 
    int sizeImage, int nrays, int nangles, int nslices, int ngpus, int niter, int* gpus)
  {
      int t;
      int blockgpu = (nslices + ngpus - 1) / ngpus;
      
      std::vector<std::future<void>> threads;

      for(t = 0; t < ngpus; t++){ 
          
          blockgpu = min(nslices - blockgpu * t, blockgpu);

          threads.push_back(std::async( std::launch::async, tEMgpu, output + (size_t)t * blockgpu * sizeImage*sizeImage, 
          count + (size_t)t * blockgpu * nrays * nangles, flat + (size_t)t * blockgpu * nrays * nangles, angles, 
          sizeImage, nrays, nangles, blockgpu,  gpus[t], niter
          ));
      }
  
      for(auto& t : threads)
          t.get();
  }

}