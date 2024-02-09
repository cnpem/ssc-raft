#include "geometries/parallel/em.hpp"

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
