#include <cublas_v2.h>

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
        configs->tomo.padsize = dim3(configs->tomo.size.x * ( 1 + configs->tomo.pad.x ),configs->tomo.size.y * ( 1 + configs->tomo.pad.y ),configs->tomo.size.z);

        /* GPU blocksize */
        configs->blocksize = parameters_int[7];

        /* Set Phase  */
		configs->phase_type                = parameters_int[6]; /* Phase type */
		configs->beta_delta                = parameters_float[0]; /* Phase  beta/delta parameter */

        /* Set Geometry */
        configs->geometry.detector_pixel_x = parameters_float[1];
        configs->geometry.detector_pixel_y = parameters_float[2];
        configs->geometry.energy           = parameters_float[3];
        configs->geometry.z2x              = parameters_float[4];
        configs->geometry.z2y              = parameters_float[5];
        configs->geometry.magnitude_x      = parameters_float[6];
        configs->geometry.magnitude_y      = parameters_float[7];
        configs->geometry.wavelength       = (configs->beta_delta == 0.0 ? 1.0:( ( plank * vc ) / configs->geometry.energy ) );

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
                4 * configs->tomo.frame_padd_memory_bytes // Projection
                ); 
    }

    void printPhaseParameters(CFG *configs)
    {
        printf("Tomo size: %d, %d, %d \n",configs->tomo.size.x,configs->tomo.size.y,configs->tomo.size.z);
        printf("Tomo Pad: %d, %d, %d \n",configs->tomo.pad.x,configs->tomo.pad.y,configs->tomo.pad.z);
        printf("Tomo Padsize: %d, %d, %d \n",configs->tomo.padsize.x,configs->tomo.padsize.y,configs->tomo.padsize.z);
        printf("Phase type: %d \n", configs->phase_type);
        printf("Phase beta / delta: %e \n", configs->beta_delta);
        printf("z2: %e \n", configs->geometry.z2x);
        printf("pixeldet: %e \n", configs->geometry.detector_pixel_x);
        printf("energy: %e \n", configs->geometry.energy);
        printf("magn: %e \n", configs->geometry.magnitude_x );
    }

    void compute_contrast_kernel(CFG configs, float *kernel)
    {
        /* Data sizes */
        int sizex        = configs.tomo.padsize.x;
        int sizey        = configs.tomo.padsize.y;

        float z2         = configs.geometry.z2x;
        float pixel_objx = configs.geometry.obj_pixel_x;
        float pixel_objy = configs.geometry.obj_pixel_y;
        float wavelength = configs.geometry.wavelength;
        float beta_delta = configs.beta_delta;

		cublasHandle_t handle = NULL;
        cublasCreate(&handle);
        cublasStatus_t stat;

        dim3 threadsPerBlock(TPBX,TPBY,1);
        dim3 gridBlock( (int)ceil( sizex / threadsPerBlock.x ) + 1, 
                        (int)ceil( sizey / threadsPerBlock.y ) + 1, 1);

		switch (configs.phase_type){
            case 0:
                /* code */
                contrast_enhance::paganinKernel<<<gridBlock,threadsPerBlock>>>(kernel, beta_delta, wavelength, 
                pixel_objx, pixel_objy, z2, dim3(sizex,sizey,1));
                break;
            case 1:
                /* code */
                
                break;
            case 2:
                /* code */
                
                break;
            case 3:
                /* code */
                
                break;
            case 4:
                /* code */
                
                break;

            default:
                contrast_enhance::paganinKernel<<<gridBlock,threadsPerBlock>>>(kernel, beta_delta, wavelength, 
                pixel_objx, pixel_objy, z2, dim3(sizex,sizey,1));
                break;
        }

        // Normalize kernel by maximum value
 		int max = 0;
        stat = cublasIsamax(handle, sizex * sizey, kernel, 1, &max);

        if (stat != CUBLAS_STATUS_SUCCESS)
            printf("Cublas Max failed in Phase Constrast Kernels\n");

        HANDLE_ERROR(cudaDeviceSynchronize());

		float scale = 0;
		HANDLE_ERROR(cudaMemcpy(&scale, kernel + max, sizeof(float), cudaMemcpyDeviceToHost));

        opt::scale<<<gridBlock,threadsPerBlock>>>(kernel, dim3(sizex,sizey,1), scale);

        // opt::fftshift2D<<<gridBlock,threadsPerBlock>>>(kernel, dim3(sizex,sizey,1));

        HANDLE_ERROR(cudaDeviceSynchronize());
    }

	void getPhase(CFG configs, GPU gpus, 
    float *projections, float *kernel, dim3 size, dim3 size_pad)
	{	
        contrast_enhance::apply_contrast_filter(configs, gpus, projections, kernel, size, size_pad, configs.tomo.pad);
    
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

        /* Kernel Computation */

        size_t nsize   = nrayspad * nslicespad;
		float *kernel  = opt::allocGPU<float>(nsize);

        compute_contrast_kernel(configs, kernel);

		int i; 
        int blocksize = configs.blocksize;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, configs.total_required_mem_per_frame_bytes, true, BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(sizez, blocksize_aux);
        }

        int ind_block = (int)ceil( (float) sizez / blocksize );

		float *dprojections = opt::allocGPU<float>((size_t) nrays * nslices * blocksize);

        /* Plan for Fourier transform - cufft */
		int n[] = {nslicespad,nrayspad};
		HANDLE_FFTERROR(cufftPlanMany(&gpus.mplan, 2, n, n, 1, nslicespad*nrayspad, n, 1, nslicespad*nrayspad, CUFFT_C2C, blocksize));

		/* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block = 0;

		for (i = 0; i < ind_block; i++){

			subblock  = min(sizez - ptr, blocksize);
			ptr_block = (size_t)nrays * nslices * ptr;

			/* Update pointer */
			ptr = ptr + subblock;

            if( subblock != blocksize){
                HANDLE_ERROR(cudaDeviceSynchronize());
				HANDLE_FFTERROR(cufftDestroy(gpus.mplan));
				HANDLE_FFTERROR(cufftPlanMany(&gpus.mplan, 2, n, n, 1, nslicespad*nrayspad, n, 1, nslicespad*nrayspad, CUFFT_C2C, subblock));
			}

            opt::CPUToGPU<float>(projections + ptr_block, dprojections, 
                            (size_t)nrays * nslices * subblock);

			getPhase( configs, gpus, dprojections, kernel,
                    dim3(nrays, nslices, subblock), 
                    dim3(nrayspad, nslicespad, subblock)
                    );

			opt::GPUToCPU<float>(projections + ptr_block, dprojections, 
                                (size_t)nrays * nslices * subblock);

		}
		HANDLE_ERROR(cudaDeviceSynchronize());
        
        /* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(gpus.mplan));

        /* Free memory */
		HANDLE_ERROR(cudaFree(dprojections));
        HANDLE_ERROR(cudaFree(kernel));

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

		CFG configs; GPU gpu_parameters;

        setPhaseParameters(&configs, paramf, parami);
        setGPUParameters(&gpu_parameters, configs.tomo.padsize, ngpus, gpus);

        // printPhaseParameters(&configs);
        // printGPUParameters(&gpu_parameters);

		int subvolume = (configs.tomo.size.z + ngpus - 1) / ngpus;
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
				
				subblock   = min(configs.tomo.size.z - ptr, subvolume);
				ptr_volume = (size_t)configs.tomo.size.x * configs.tomo.size.y * ptr;

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
}

