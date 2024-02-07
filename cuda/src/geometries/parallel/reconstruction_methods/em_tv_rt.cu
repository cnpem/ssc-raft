#include "../../../../inc/geometries/parallel/em.h"

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
    dim3 gridBlockD((int)ceil((nrays)     / threadsPerBlock.x) + 1,
		                (int)ceil((nangles)   / threadsPerBlock.y) + 1,
		                (int)ceil( blockSize  / threadsPerBlock.z) + 1
                    );

    dim3 gridBlockF((int)ceil((sizeImage) / threadsPerBlock.x) + 1,
		                (int)ceil((sizeImage) / threadsPerBlock.y) + 1,
		                (int)ceil( blockSize  / threadsPerBlock.z) + 1
                    );
    
    kernel_ones<<<gridBlockF, threadsPerBlock>>>(d_em, sizeImage, nrays, nangles, blockSize);

    // temp assignment to d_x in order to computer initial error estimate!
    cudaMemcpy(d_x, d_em, sizeof(float) * sizeImage * sizeImage * blockSize, cudaMemcpyDeviceToDevice);
    
    kernel_backprojectionOfOnes<<<gridBlockF, threadsPerBlock>>>(d_backones, d_angles, sizeImage, nrays, nangles, blockSize);

    iterEM( d_em, d_sino, d_sinotmp, d_backones, d_angles, 
	    sizeImage, nrays, nangles, blockSize, device);

    getError_F(&error, d_x, d_em, sizeImage, blockSize, device);
    
    for (m = 0; m < niter; m++){
	    //get d_x pointer
	    cudaMemcpy(d_x, d_em, sizeof(float) * sizeImage * sizeImage * blockSize, cudaMemcpyDeviceToDevice);

      //EM iterations
      for( k = 0; k < niter_em; k++ ){

	      iterEM( d_em, d_sino, d_sinotmp, d_backones, d_angles, 
                sizeImage, nrays, nangles, blockSize, device
              );
	    
	      cudaDeviceSynchronize();
	    }
	
      //TV iterations
      cudaMemcpy(d_y, d_em, sizeof(float) * sizeImage * sizeImage * blockSize, cudaMemcpyDeviceToDevice);
    
      for (k = 0; k < niter_tv; k++ ){
        
        iterTV( d_y, d_em, d_backones, 
                sizeImage, blockSize, device, reg, epsilon
              );

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
    cudaMemcpy (output, d_em, blockSize*sizeImage*sizeImage*sizeof(float), cudaMemcpyDeviceToHost);

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



