#include "processing/filters.hpp"
#include "processing/processing.hpp"
#include "common/opt.hpp"

extern "C" {

    void setPhaseParameters(CFG *configs, float *parameters_float, int *parameters_int)
    {
        /* Set Tomogram (or detector) variables */
        configs->tomo.size     = dim3(parameters_int[0],parameters_int[1],parameters_int[2]);  

        /* Set padding */
        
        /* Pad is the integer number such that the total padding is = ( pad + 1 ) * dimension 
        Example: 
            - Data have dimension on x-axis of nx = 2048;
            - The padx = 1;
            - The new dimension is nx_pad = nx * (1 + padx) = 4096
        */
        configs->tomo.pad      = dim3(parameters_int[3],parameters_int[4],0); //dim3(parameters_int[3],parameters_int[4],parameters_int[5]);

        /* Padsize is the final dimension with padding. 
        Example:
            - Data have dimension on x-axis of nx = 2048 and padx = 1
            - padsizex = nx_pad = nx * (1 + padx) = 4096
            - See Pad example above. 
        */
        configs->tomo.padsize = dim3(configs->tomo.size.x * ( 1 + configs->tomo.pad.x),configs->tomo.size.y * ( 1 + configs->tomo.pad.y),configs->tomo.size.z);

        /* GPU blocksize */
        configs->blocksize = parameters_int[7];

        /* Set Phase  */
		configs->phase_type                = parameters_int[6]; /* Phase type */
		configs->delta_beta                = parameters_float[0]; /* Phase  delta/beta parameter */

        /* Set Geometry */
        // configs->geometry.geometry         = parameters_int[9];
        configs->geometry.detector_pixel_x = parameters_float[1];
        configs->geometry.detector_pixel_y = parameters_float[2];
        configs->geometry.energy           = parameters_float[3];
        configs->geometry.z2x              = parameters_float[4];
        configs->geometry.z2y              = parameters_float[5];
        configs->geometry.magnitude_x      = parameters_float[6];
        configs->geometry.magnitude_y      = parameters_float[7];
        configs->geometry.wavelenght       = ( plank * vc ) / configs->geometry.energy;

        /* Set magnitude [(z1+z2)/z1] according to the beam geometry */
		// switch (configs->geometry.geometry){
		// 	case 0: /* Parallel */	
		// 		configs->geometry.magnitude_x = 1.0;
		// 		configs->geometry.magnitude_y = 1.0;
		// 		break;
		// 	case 1: /* Conebeam */
		// 		// configs->geometry.magnitude_x = ( configs->geometry.z1x + configs->geometry.z2x ) / configs->geometry.z1x;
		// 		// configs->geometry.magnitude_y = ( configs->geometry.z1y + configs->geometry.z2y ) / configs->geometry.z1y;
		// 		break;
		// 	case 2: /* Fanbeam */		
		// 		// configs->geometry.magnitude_x = ( configs->geometry.z1x + configs->geometry.z2x ) / configs->geometry.z1x;
		// 		configs->geometry.magnitude_y = 1.0;
		// 		break;
		// 	default:
		// 		// printf("Parallel case as default! \n");
		// 		configs->geometry.magnitude_x = 1.0;
		// 		configs->geometry.magnitude_y = 1.0;
		// 		break;
		// }

        configs->geometry.obj_pixel_x = configs->geometry.detector_pixel_x / configs->geometry.magnitude_x;
        configs->geometry.obj_pixel_y = configs->geometry.detector_pixel_y / configs->geometry.magnitude_y;

        configs->geometry.z2x /= configs->geometry.magnitude_x;
        configs->geometry.z2y /= configs->geometry.magnitude_y;

        /* Compute memory in bytes of a single frame for Measurements and its padded version for FFT */
        configs->tomo.lenght_memory_bytes     = static_cast<float>(sizeof(float)) * configs->tomo.size.x;
        configs->tomo.width_memory_bytes      = static_cast<float>(sizeof(float)) * configs->tomo.size.y;

        configs->tomo.frame_memory_bytes      = configs->tomo.lenght_memory_bytes * configs->tomo.width_memory_bytes;
        configs->tomo.frame_padd_memory_bytes = static_cast<float>(sizeof(float)) * configs->tomo.padsize.x * configs->tomo.padsize.y;

        /* Compute total memory used of Phase Filter method on a single frame */
        configs->total_required_mem_per_frame_bytes = (
                configs->tomo.frame_memory_bytes // Projection
                ); 
    }

    void printPhaseParameters(CFG *configs)
    {
        printf("Tomo size: %d, %d, %d \n",configs->tomo.size.x,configs->tomo.size.y,configs->tomo.size.z);
        printf("Tomo Pad: %d, %d, %d \n",configs->tomo.pad.x,configs->tomo.pad.y,configs->tomo.pad.z);
        printf("Tomo Padsize: %d, %d, %d \n",configs->tomo.padsize.x,configs->tomo.padsize.y,configs->tomo.padsize.z);
        printf("Phase type: %d \n", configs->phase_type);
        printf("Phase delta / beta: %e \n", configs->delta_beta);
        printf("z2: %e \n", configs->geometry.z2x);
        printf("pixeldet: %e \n", configs->geometry.detector_pixel_x);
        printf("energy: %e \n", configs->geometry.energy);
        printf("magn: %e \n", configs->geometry.magnitude_x );
    }

	void applyPhase(CFG configs, GPU gpus, 
    float *projections, dim3 tomo, dim3 tomo_pad)
	{
		switch (configs.phase_type){
			case 0:
				/* Paganin */
				_paganin_gpu(configs, gpus, projections, tomo, tomo_pad, configs.tomo.pad);
				break;
            case 1:
				/* Paganin tomopy */
				_paganin_gpu_tomopy(configs, gpus, projections, tomo, tomo_pad, configs.tomo.pad);
				break;
            case 2:
				/* Paganin v0 */
				_paganin_gpu_v0(configs, gpus, projections, tomo, tomo_pad, configs.tomo.pad);
				break;
			default:
                // printf("Using default Paganin phase filter. \n");
				_paganin_gpu(configs, gpus, projections, tomo, tomo_pad, configs.tomo.pad);
				break;
		}	

	}

	void getPhase(CFG configs, GPU gpus, 
    float *projections, dim3 size, dim3 size_pad)
	{	
		/* Plan for Fourier transform - cufft */
		int n[] = {(int)size_pad.x,(int)size_pad.x};
		HANDLE_FFTERROR(cufftPlanMany(&gpus.mplan, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, size.z));

		applyPhase(configs, gpus, projections, size, size_pad);
	
		// cudaDeviceSynchronize();

		/* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(gpus.mplan));
	}

	void getPhaseGPU(CFG configs, GPU gpus, 
	float *projections, int sizez, int ngpu)
	{	
		/* Initialize GPU device */
		HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Projection data sizes */
        int nrays      = configs.tomo.size.x;
        int nslices    = configs.tomo.size.y;
        int nrayspad   = configs.tomo.padsize.x;
        int nslicespad = configs.tomo.padsize.y;

		int i; 
        int blocksize = configs.blocksize;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, configs.total_required_mem_per_slice_bytes, true, A100_MEM);
            blocksize          = min(sizez, blocksize_aux);
        }

        int ind_block = (int)ceil( (float) sizez / blocksize );

		float *dprojections = opt::allocGPU<float>((size_t) nrays * nslices * blocksize);

        		/* Plan for Fourier transform - cufft */
		int n[] = {nrays,nslices};
		HANDLE_FFTERROR(cufftPlanMany(&gpus.mplan, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize));

		/* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block = 0;

		for (i = 0; i < ind_block; i++){

			subblock    = min(configs.tomo.size.z - ptr, blocksize);
			ptr_block = (size_t)nrays * nslices * ptr;

			/* Update pointer */
			ptr = ptr + subblock;

            if( subblock != blocksize){
				HANDLE_FFTERROR(cufftDestroy(gpus.mplan));
				HANDLE_FFTERROR(cufftPlanMany(&gpus.mplan, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, subblock));
			}

            opt::CPUToGPU<float>(projections + ptr_block, dprojections, 
                            (size_t)nrays * nslices * subblock);

			getPhase( configs, gpus, projections,
                    dim3(nrays, nslices, subblock), 
                    dim3(nrayspad, nslicespad, subblock)
                    );

			opt::GPUToCPU<float>(projections + ptr_block, dprojections, 
                                (size_t)nrays * nslices * subblock);

		}
		// HANDLE_ERROR(cudaDeviceSynchronize());
        
        /* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(gpus.mplan));

        /* Free memory */
		HANDLE_ERROR(cudaFree(dprojections));

	}

    void getPhaseMultiGPU(int *gpus, int ngpus, 
    float *projections, float *paramf, int *parami)
	{	
		int i, Maxgpudev;
		
		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		CFG configs; DIM tomo; GPU gpu_parameters;

        setPhaseParameters(&configs, paramf, parami);

        setGPUParameters(&gpu_parameters, configs.tomo.padsize, ngpus, gpus);

        printPhaseParameters(&configs);
        printGPUParameters(&gpu_parameters);

		int subvolume = (tomo.size.z + ngpus - 1) / ngpus;
		int subblock, ptr = 0; size_t ptr_volume = 0;

		if (ngpus == 1){ /* 1 device */

			getPhaseGPU(configs, gpu_parameters, projections, subvolume, gpus[0]);

		}else{
		/* Launch async Threads for each device.
			Each device solves a block of 'nrays * nangles' size.
		*/
			// See future c++ async launch
			std::vector<std::future<void>> threads = {};
            threads.reserve(ngpus);

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(tomo.size.z - ptr, subvolume);
				ptr_volume = (size_t)tomo.size.x * tomo.size.y * ptr;

				/* Update pointer */
				ptr = ptr + subblock;
				
				threads.push_back( std::async(  std::launch::async, 
												getPhaseGPU,
                                                configs, 
												gpu_parameters, 
												projections + ptr_volume, 
												subblock, gpus[i]
												));		

			}
			// Log("Synchronizing all threads...\n");
		
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}	

		HANDLE_ERROR(cudaDeviceSynchronize());
	}

	// void setPhaseFilterKernel(CFG *configs, GPU gpus, 
    // float *kernel, int phase_type, float phase_reg,
	// dim3 size_pad)
	// {	
	// 	// cublasHandle_t handle = NULL;
    //     // cublasCreate(&handle);
    //     // cublasStatus_t stat;

	// 	/* Compute phase filter kernel */ 
	// 	switch (phase_type){
	// 			case 0:
	// 				/* code */
	// 				printf("No filter was selected!");
	// 				break;
	// 			case 1:
	// 				/* code */
	// 				paganinKernel<<<gpus.Grd,gpus.BT>>>(kernel, phase_reg, size_pad);
	// 				break;
	// 			case 2:
	// 				/* code */
	// 				bronnikovKernel<<<gpus.Grd,gpus.BT>>>(kernel, phase_reg, size_pad);
	// 				break;
	// 			case 3:
	// 				/* code */
	// 				bornKernel<<<gpus.Grd,gpus.BT>>>(kernel, phase_reg, size_pad);
	// 				break;
	// 			case 4:
	// 				/* code */
	// 				rytovKernel<<<gpus.Grd,gpus.BT>>>(kernel, phase_reg, size_pad);
	// 				break;
	// 			default:
	// 				printf("Using default Paganin phase filter. \n");
	// 				paganinKernel<<<gpus.Grd,gpus.BT>>>(kernel, phase_reg, size_pad);
	// 				break;
	// 		}

    //     /* Normalize kernel by maximum value */ 
 	// 	// int max;
    //     // stat = cublasIsamax(handle, (int)size_pad.x * size_pad.y, kernel, 1, &max);

    //     // if (stat != CUBLAS_STATUS_SUCCESS)
    //     //     printf("Cublas Max failed\n");

	// 	// float maximum;
	// 	// HANDLE_ERROR(cudaMemcpy(&maximum, kernel + max, sizeof(float), cudaMemcpyDeviceToHost));
    //     // Normalize<<<gpus.Grd,gpus.BT>>>(kernel, maximum, size_pad);

	// 	// cublasDestroy(handle);
	// }
}

