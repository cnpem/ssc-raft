#include "../../../inc/include.h"
#include "../../../inc/common/kernel_operators.hpp"
#include "../../../inc/common/complex.hpp"
#include "../../../inc/common/types.hpp"
#include "../../../inc/common/operations.hpp"
#include "../../../inc/common/logerror.hpp"

extern "C"{   

   void reconstruction_pipeline()
  {
      /* Only GPU */

      // rings()

      // padding()

      // fdk()

  }

   void _reconstruction_pipeline(int gpu)
   {
      /* Send to GPU (device) memory */
      
      /* Initialize GPU device */
      HANDLE_ERROR(cudaSetDevice(gpu))

      /* send memory to device */
 
      reconstruction_pipeline();

      /* send memory to host */

      /* Free device memory */
      
      cudaDeviceSynchronize();
   }

   void reconstruction_pipeline_block(float *dataIn, float dataOut, /* something */, int* gpus, int ngpus)
   {	/* Extern C function  */
      int t, maximumgpu;

      /* Variables for time profile */
      cudaEvent_t start, stop;
      float milliseconds;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);

      /* Multiples devices */
      cudaGetDeviceCount(&maximumgpu);

      /* If devices input are larger than actual devices on GPU, exit */
      for(i = 0; i < ngpus; i++) 
            assert(gpus[i] < maximumgpu && "Invalid device number.");

      /* Start time */
      cudaEventRecord(start);

      /* Struct to parameters (PAR param) and Profiling (PROF prof - not finished) */
      PAR param; PROF prof;

      /* Set all parameters necessary to Rebinning */
      set_gc_parameters(&param, parameters, volumesize, gpus);

      int blockgpu = ( param.nslices + ngpus - 1) / ngpus;
      
      std::vector<std::future<void>> threads;

      for(t = 0; t < ngpus; t++){ 
         
         blockgpu = min(nslices - blockgpu * t, blockgpu);

         threads.push_back(std::async( std::launch::async, _reconstruction_pipeline, 
         /* variables */
         ));
      }

      for(auto& t : threads)
         t.get();

      /* Record Total time*/
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);

      cudaDeviceSynchronize();
   }

}