#include "processing/filters.hpp"

extern "C" {

void setPhaseFilterParameters(GEO *geometry, DIM *tomo, 
	float *parameters_float, int *parameters_int)
	{
		/* Set Geometry */
		geometry->geometry = parameters_int[0];
		
		/* Set Tomogram (or detector) variables (h for horizontal (nrays) and v for vertical (nslices)) */
		tomo->size         = dim3(parameters_int[1],parameters_int[2],parameters_int[3]); 

		tomo->x            = parameters_float[0];
		tomo->y            = parameters_float[1];
		tomo->z            = parameters_float[2];
		tomo->dx           = parameters_float[3];
		tomo->dy           = parameters_float[4];
		tomo->dz           = parameters_float[5];

		/* Set Padding */
		tomo->pad          = dim3(parameters_int[4],parameters_int[5],parameters_int[6]); 

		int npadx          = tomo->size.x * ( 1 + 2 * tomo->pad.x ); 
		int npady          = tomo->size.y * ( 1 + 2 * tomo->pad.y ); 
		int npadz          = tomo->size.z * ( 1 + 2 * tomo->pad.z );

		tomo->padsize         = dim3(npadx,npady,npadz); 

		/* Set General reconstruction variables*/
		geometry->energy   = parameters_float[6];

		geometry->lambda   = ( plank * vc          ) / geometry->energy;
		geometry->wave     = ( 2.0   * float(M_PI) ) / geometry->lambda;

		geometry->z1x      = parameters_float[7];
		geometry->z1y      = parameters_float[8];
		geometry->z2x      = parameters_float[9];
		geometry->z2y      = parameters_float[10];

		/* Set magnitude [(z1+z2)/z1] according to the beam geometry */
		switch (geometry->geometry){
			case 0: /* Parallel */	
				geometry->magnitude_x = 1.0;
				geometry->magnitude_y = 1.0;
				break;
			case 1: /* Conebeam */
				geometry->magnitude_x = ( geometry->z1x + geometry->z2x ) / geometry->z1x;
				geometry->magnitude_y = ( geometry->z1y + geometry->z2y ) / geometry->z1y;
				break;
			case 2: /* Fanbeam */		
				geometry->magnitude_x = ( geometry->z1x + geometry->z2x ) / geometry->z1x;
				geometry->magnitude_y = 1.0;
				break;
			default:
				printf("Parallel case as default! \n");
				geometry->magnitude_x = 1.0;
				geometry->magnitude_y = 1.0;
				break;
		}

	}


	void getPhaseFilterMultiGPU(int *gpus, int ngpus, 
    float *projections, float *paramf, int *parami, 
	int phase_type, float phase_reg)
	{	
		int i, Maxgpudev;
		
		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		GEO geometry; DIM tomo; GPU gpu_parameters;

        setPhaseFilterParameters(&geometry, &tomo, paramf, parami);

        setGPUParameters(&gpu_parameters, tomo.padsize, ngpus, gpus);

		int subvolume = (tomo.size.z + ngpus - 1) / ngpus;
		int subblock, ptr = 0; size_t ptr_volume = 0;

		if (ngpus == 1){ /* 1 device */

			getPhaseFilterGPU(  gpu_parameters, geometry, tomo, 
                                projections, 
                                phase_type, phase_reg, gpus[0]);

		}else{
		/* Launch async Threads for each device.
			Each device solves a block of 'nrays * nangles' size.
		*/
			// See future c++ async launch
			std::vector<std::future<void>> threads = {};

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(tomo.size.z - ptr, subvolume);
				ptr_volume = (size_t)tomo.size.x * tomo.size.y * ptr;

				/* Update pointer */
				ptr = ptr + subblock;
				
				threads.push_back( std::async(  std::launch::async, 
												getPhaseFilterGPU, 
												gpu_parameters, 
                                                geometry, tomo,
												projections + ptr_volume, 
												phase_type, phase_reg, 
												gpus[i]
												));		

			}
		
			// Log("Synchronizing all threads...\n");
		
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}	

		HANDLE_ERROR(cudaDeviceSynchronize());
	}

	void getPhaseFilterGPU(GPU gpus, GEO geometry, DIM tomo,
	float *projections, int phase_type, float phase_reg, int ngpu)
	{	
		/* Initialize GPU device */
		HANDLE_ERROR(cudaSetDevice(ngpu));

		int i; 
		int blocksize = min(tomo.size.z,32);
		int ind_block = (int)ceil( (float) tomo.size.z / blocksize );

		float *dprojections; 
		HANDLE_ERROR(cudaMalloc((void **)&dprojections, sizeof(float) * (size_t)tomo.size.x * tomo.size.y * blocksize )); 

		/* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block = 0;

		for (i = 0; i < ind_block; i++){

			subblock    = min(tomo.size.z - ptr, blocksize);
			ptr_block = (size_t)tomo.size.x * tomo.size.y * ptr;

			/* Update pointer */
			ptr = ptr + subblock;

			getPhaseFilter( gpus, geometry, projections, phase_type, phase_reg,
							 dim3(tomo.size.x, tomo.size.y, subblock), 
							 dim3(tomo.npad.x, tomo.npad.y, subblock)
							 );

			HANDLE_ERROR(cudaMemcpy(projections, dprojections + ptr_block, (size_t)tomo.size.x * tomo.size.y * subblock * sizeof(float), cudaMemcpyDeviceToHost));

		}
		HANDLE_ERROR(cudaDeviceSynchronize());

		HANDLE_ERROR(cudaFree(dprojections));

	}

	void getPhaseFilter(GPU gpus, GEO geometry, float *projections, 
	int phase_type, int phase_reg, dim3 size, dim3 size_pad)
	{	
		float *phase_kernel;
		HANDLE_ERROR(cudaMalloc((void **)&phase_kernel, sizeof(float) * (size_t)size_pad.x * size_pad.y ));
		
		setPhaseFilterKernel(gpus, geometry, phase_kernel, size_pad, phase_type, phase_reg);

		/* Plan for Fourier transform - cufft */
		int n[] = {(int)size_pad.x,(int)size_pad.x};
		HANDLE_FFTERROR(cufftPlanMany(&gpus.mplan, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, size.z));

		applyPhaseFilter(gpus, projections, phase_kernel, phase_type, size, size_pad);
	
		// cudaDeviceSynchronize();

		/* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(gpus.mplan));
		HANDLE_ERROR(cudaFree(phase_kernel));
	}

	void applyPhaseFilter(GPU gpus, float *projections, float *kernel, 
	int phase_type, dim3 tomo, dim3 tomo_pad)
	{
		switch (phase_type){
			case 0:
				/* code */
				printf("No filter was selected!");
				break;
			case 1:
				/* code */
				_paganin_gpu(gpus, projections, kernel, tomo, tomo_pad);
				break;
			case 2:
				/* code */
				_bronnikov_gpu(gpus, projections, kernel, tomo, tomo_pad);
				break;
			case 3:
				/* code */
				_born_gpu(gpus, projections, kernel, tomo, tomo_pad);
				break;
			case 4:
				/* code */
				_rytov_gpu(gpus, projections, kernel, tomo, tomo_pad);
				break;

			default:
				_paganin_gpu(gpus, projections, kernel, tomo, tomo_pad);
				break;
		}	

	}

	void setPhaseFilterKernel(GPU gpus, GEO geometry, float *kernel, 
	dim3 size_pad, int phase_type, float phase_reg)
	{	
		cublasHandle_t handle = NULL;
        cublasCreate(&handle);
        cublasStatus_t stat;

		/* Compute phase filter kernel */ 
		switch (phase_type){
				case 0:
					/* code */
					printf("No filter was selected!");
					break;
				case 1:
					/* code */
					paganinKernel<<<gpus.Grd,gpus.BT>>>(geometry, kernel, size_pad, phase_reg);
					break;
				case 2:
					/* code */
					bronnikovKernel<<<gpus.Grd,gpus.BT>>>(geometry, kernel, size_pad, phase_reg);
					break;
				case 3:
					/* code */
					bornKernel<<<gpus.Grd,gpus.BT>>>(geometry, kernel, size_pad, phase_reg);
					break;
				case 4:
					/* code */
					rytovKernel<<<gpus.Grd,gpus.BT>>>(geometry, kernel, size_pad, phase_reg);
					break;
				default:
					printf("Using default Paganin phase filter. \n");
					paganinKernel<<<gpus.Grd,gpus.BT>>>(geometry, kernel, size_pad, phase_reg);
					break;
			}

        /* Normalize kernel by maximum value */ 
 		int max;
        stat = cublasIsamax(handle, (int)size_pad.x * size_pad.y, kernel, 1, &max);

        if (stat != CUBLAS_STATUS_SUCCESS)
            printf("Cublas Max failed\n");

		float maximum;
		HANDLE_ERROR(cudaMemcpy(&maximum, kernel + max, sizeof(float), cudaMemcpyDeviceToHost));
        Normalize<<<gpus.Grd,gpus.BT>>>(kernel, maximum, size_pad);

		cublasDestroy(handle);
	}

}

