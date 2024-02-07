#include "../../../../inc/geometries/parallel/em.h"

//---------------------------
// transmission-EM algorithm
//---------------------------

extern "C" {  

  void get_tEM_RT_MultiGPU(float* recon, float* count, float *flat, float* angles, 
  float *paramf, int *parami, int* gpus, int ngpus)
  {
    int i, Maxgpudev;
		
		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		CFG configs; GPU gpu_parameters;

    setEMParameters(&configs, paramf, parami);

    setGPUParameters(&gpu_parameters, configs.tomo.npad, ngpus, gpus);

		int subvolume = (configs.tomo.size.z + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			get_tEM_RT_GPU(configs, gpu_parameters, recon, count, flat, angles, subvolume, gpus[0]);

		}else{
      /* Launch async Threads for each device.
        See future c++ async launch.
      */

			std::vector<std::future<void>> threads = {};

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(configs.tomo.size.z - ptr, subvolume);

				threads.push_back( std::async( std::launch::async, 
												get_tEM_RT_GPU, 
												configs, gpu_parameters, 
                        recon + (size_t)configs.recon.size.x * configs.recon.size.y * ptr,
												count + (size_t)configs.tomo.size.x  * configs.tomo.size.y  * ptr, 
                        flat  + (size_t)configs.tomo.size.x                         * ptr, 
												angles, 
                        subblock,
												gpus[i]
												));

                /* Update pointer */
				ptr = ptr + subblock;		

			}
		
			// Log("Synchronizing all threads...\n");
		
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}	

		HANDLE_ERROR(cudaDeviceSynchronize());
  }

  void get_tEM_RT_GPU(CFG configs, GPU gpus, float *recon, float *count, float *flat, float *angles, 
  int sizez, int ngpu)
  {
    int i; 
		int blocksize = min(sizez,32);
    int ind_block = (int)ceil( (float) sizez / blocksize );

    HANDLE_ERROR(cudaSetDevice(ngpu));
    
    float *drecon, *dcount, *dflat, *dangles;
    
    /* Allocate GPU memory for the input and output image */ 
    HANDLE_ERROR(cudaMalloc((void **)&drecon ,sizeof(float) * (size_t)configs.recon.size.x * configs.recon.size.y * blocksize));
    HANDLE_ERROR(cudaMalloc((void **)&dcount ,sizeof(float) * (size_t)configs.tomo.size.x  * configs.tomo.size.y  * blocksize));
    HANDLE_ERROR(cudaMalloc((void **)&dflat  ,sizeof(float) * (size_t)configs.tomo.size.x                         * blocksize));
    HANDLE_ERROR(cudaMalloc((void **)&dangles,sizeof(float) * configs.tomo.nangles));

    HANDLE_ERROR(cudaMemcpy(dangles, angles, sizeof(float) * configs.tomo.nangles, cudaMemcpyHostToDevice));	

    /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0, ptr_block_recon = 0, ptr_block_flat = 0;

    for (i = 0; i < ind_block; i++){

			subblock        = min(sizez - ptr, blocksize);
      ptr_block_tomo  = (size_t)configs.tomo.size.x  * configs.tomo.size.y  * ptr;
      ptr_block_recon = (size_t)configs.recon.size.x * configs.recon.size.y * ptr;
      ptr_block_flat  = (size_t)configs.recon.size.x * configs.numflats     * ptr;

      /* Update pointer */
			ptr = ptr + subblock;

      HANDLE_ERROR(cudaMemcpy(dcount, count  + ptr_block_tomo, sizeof(float) * (size_t)configs.tomo.size.x * configs.tomo.size.y * subblock, cudaMemcpyHostToDevice));	
      HANDLE_ERROR(cudaMemcpy(dflat , flat   + ptr_block_flat, sizeof(float) * (size_t)configs.tomo.size.x                       * subblock, cudaMemcpyHostToDevice));	

      get_tEM_RT( configs, gpus, drecon, dcount, dflat, dangles, subblock);                           
    
      HANDLE_ERROR(cudaMemcpy(recon + ptr_block_recon, drecon, (size_t)configs.recon.size.x * configs.recon.size.y * subblock * sizeof(float), cudaMemcpyDeviceToHost));
    }

    HANDLE_ERROR(cudaFree(drecon));
    HANDLE_ERROR(cudaFree(dcount));
    HANDLE_ERROR(cudaFree(dflat));
    HANDLE_ERROR(cudaFree(dangles));
    HANDLE_ERROR(cudaDeviceSynchronize());

  }

  void get_tEM_RT(CFG configs, GPU gpus, float *output, float *count, float *flat, float *angles, int blockSize)
  {
    int k;
    int niter     = configs.em_iterations;
    int sizeImage = configs.recon.size.x;
    int nrays     = configs.tomo.size.x;
    int nangles   = configs.tomo.nangles;

    float *backcounts, *temp, *back;

    HANDLE_ERROR(cudaMalloc((void **)&back      ,sizeof(float) * (size_t)sizeImage * sizeImage * blockSize));
    HANDLE_ERROR(cudaMalloc((void **)&backcounts,sizeof(float) * (size_t)sizeImage * sizeImage * blockSize));
    HANDLE_ERROR(cudaMalloc((void **)&temp      ,sizeof(float) * (size_t)nrays     * nangles   * blockSize));

    kernel_ones<<<gpus.Grd,gpus.BT>>>(output, sizeImage, nrays, nangles, blockSize);

    kernel_backprojection<<<gpus.Grd,gpus.BT>>>(backcounts, count, angles, sizeImage, nrays, nangles, blockSize);

    for( k = 0; k < niter; k++ ){

      kernel_radon<<<gpus.Grd,gpus.BT>>>(temp, output, angles, sizeImage, nrays, nangles, blockSize, 1.0);
      
      kernel_flatTimesExp<<<gpus.Grd,gpus.BT>>>(temp, flat, sizeImage, nrays, nangles, blockSize);
      
      kernel_backprojection<<<gpus.Grd,gpus.BT>>>(back, temp, angles, sizeImage, nrays, nangles, blockSize);
      
      kernel_update<<<gpus.Grd,gpus.BT>>>(output, back, backcounts, sizeImage, nrays, nangles, blockSize);
      
      HANDLE_ERROR(cudaDeviceSynchronize());
    }
    HANDLE_ERROR(cudaFree(temp));
    HANDLE_ERROR(cudaFree(back));
    HANDLE_ERROR(cudaFree(backcounts));
  }
}

//----------------------
// emission-EM algorithm
//----------------------

extern "C"{   

  void get_eEM_RT_MultiGPU(float* recon, float* tomogram, float* angles, 
  float *paramf, int *parami, int* gpus, int ngpus)
  {
    int i, Maxgpudev;
		
		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		CFG configs; GPU gpu_parameters;

    setEMParameters(&configs, paramf, parami);

    setGPUParameters(&gpu_parameters, configs.tomo.npad, ngpus, gpus);

		int subvolume = (configs.tomo.size.z + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			get_eEM_RT_GPU(configs, gpu_parameters, recon, tomogram, angles, subvolume, gpus[0]);

		}else{
      /* Launch async Threads for each device.
        See future c++ async launch
      */
 
			std::vector<std::future<void>> threads = {};

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(configs.tomo.size.z - ptr, subvolume);

				threads.push_back( std::async( std::launch::async, 
												get_eEM_RT_GPU, 
												configs, gpu_parameters, 
                        recon    + (size_t)configs.recon.size.x * configs.recon.size.y * ptr,
												tomogram + (size_t)configs.tomo.size.x  * configs.tomo.size.y  * ptr, 
												angles, 
                        subblock,
												gpus[i]
												));

        /* Update pointer */
				ptr = ptr + subblock;		

			}
		
			// Log("Synchronizing all threads...\n");
		
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}	

		HANDLE_ERROR(cudaDeviceSynchronize());
  }

  void get_eEM_RT_GPU(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, 
  int sizez, int ngpu)
  {
    int i; 
		int blocksize = min(sizez,32);
    int ind_block = (int)ceil( (float) sizez / blocksize );

    HANDLE_ERROR(cudaSetDevice(ngpu));
    
    float *drecon, *dtomo, *dangles;
    
    /* Allocate GPU memory for the input and output image */ 
    HANDLE_ERROR(cudaMalloc((void **)&drecon ,sizeof(float) * (size_t)configs.recon.size.x * configs.recon.size.y * blocksize));  
    HANDLE_ERROR(cudaMalloc((void **)&dtomo  ,sizeof(float) * (size_t)configs.tomo.size.x  * configs.tomo.size.y  * blocksize));
    HANDLE_ERROR(cudaMalloc((void **)&dangles,sizeof(float) * configs.tomo.nangles));

    HANDLE_ERROR(cudaMemcpy(dangles, angles, sizeof(float) * configs.tomo.nangles, cudaMemcpyHostToDevice));	

    /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0, ptr_block_recon = 0;

    for (i = 0; i < ind_block; i++){

			subblock        = min(sizez - ptr, blocksize);
      ptr_block_tomo  = (size_t)configs.tomo.size.x  * configs.tomo.size.y  * ptr;
      ptr_block_recon = (size_t)configs.recon.size.x * configs.recon.size.y * ptr;
      
      /* Update pointer */
			ptr = ptr + subblock;

      HANDLE_ERROR(cudaMemcpy(dtomo, tomogram + ptr_block_tomo, sizeof(float) * (size_t)configs.tomo.size.x * configs.tomo.size.y * subblock, cudaMemcpyHostToDevice));	
      
      get_eEM_RT( configs, gpus, drecon, dtomo, dangles, subblock);                           
    
      HANDLE_ERROR(cudaMemcpy(recon + ptr_block_recon, drecon, (size_t)configs.recon.size.x * configs.recon.size.y * subblock * sizeof(float), cudaMemcpyDeviceToHost));
    }
    HANDLE_ERROR(cudaFree(drecon));
    HANDLE_ERROR(cudaFree(dtomo));
    HANDLE_ERROR(cudaFree(dangles));
    HANDLE_ERROR(cudaDeviceSynchronize());
  }

}

extern "C" {

  void get_eEM_RT(CFG configs, GPU gpus, float *output, float *tomo, float *angles, int blockSize)
  {
    int k;
    int niter     = configs.em_iterations;
    int sizeImage = configs.recon.size.x;
    int nrays     = configs.tomo.size.x;
    int nangles   = configs.tomo.nangles;
    
    float *backones, *temp;

    HANDLE_ERROR(cudaMalloc((void **)&backones  ,sizeof(float) * (size_t)sizeImage * sizeImage * blockSize));
    HANDLE_ERROR(cudaMalloc((void **)&temp      ,sizeof(float) * (size_t)nrays     * nangles   * blockSize));
    
    kernel_ones<<<gpus.Grd,gpus.BT>>>(output, sizeImage, nrays, nangles, blockSize);

    kernel_backprojectionOfOnes<<<gpus.Grd,gpus.BT>>>(backones, angles, sizeImage, nrays, nangles, blockSize);
 
    for( k = 0; k < niter; k++ ){

      kernel_radonWithDivision<<<gpus.Grd,gpus.BT>>>(temp, output, tomo, angles, sizeImage, nrays, nangles, blockSize, 1.0);
            
      kernel_backprojectionWithUpdate<<<gpus.Grd,gpus.BT>>>(output, temp, backones, angles, sizeImage, nrays, nangles, blockSize);
      
      HANDLE_ERROR(cudaDeviceSynchronize());
    }
    HANDLE_ERROR(cudaFree(temp));
    HANDLE_ERROR(cudaFree(backones));
  }
}
