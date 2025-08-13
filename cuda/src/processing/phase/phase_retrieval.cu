#include <cublas_v2.h>

#include "processing/filters.hpp"
#include "processing/processing.hpp"
#include "common/opt.hpp"

extern "C" {
    void compute_contrast_kernel(DIM tomo, GEO geometry, CEF ContrastFilter, float *kernel)
    {
        /* Data sizes */
        int sizex        = PDIM(tomo.size.x,tomo.pad.x); // tomo.size.x * (1 + tomo.pad.x);
        int sizey        = PDIM(tomo.size.y,tomo.pad.y); // tomo.size.y * (1 + tomo.pad.y);

        float z2         = geometry.z2.x / geometry.magnitude.x;
        float pixel_objx = geometry.obj_pixel.x;
        float pixel_objy = geometry.obj_pixel.y;
        float wavelength = geometry.wavelength;
        float beta_delta = ContrastFilter.beta_delta;
        float reg        = ContrastFilter.regularization;

		cublasHandle_t handle = NULL;
        cublasCreate(&handle);
        cublasStatus_t stat;

        dim3 threadsPerBlock(TPBX,TPBY,1);
        dim3 gridBlock = opt::setGridBlock(dim3(sizex,sizey,1), threadsPerBlock);

        printf("ContrastFilter.method: %d \n", ContrastFilter.method);
        fflush(stdout);
		switch (ContrastFilter.method){
            case contrast_enhance::ContrastEnhanceType::paganin:
                /* Paganin by frames, classic */
                printf("Using Paganin classico \n");
                fflush(stdout);
                contrast_enhance::paganinKernel<<<gridBlock,threadsPerBlock>>>(kernel, beta_delta, wavelength, 
                pixel_objx, pixel_objy, z2, dim3(sizex,sizey,1));
                break;
            case contrast_enhance::ContrastEnhanceType::contrast:
                printf("Using Contrast \n");
                fflush(stdout);
                contrast_enhance::contrast_paganin_based_Kernel<<<gridBlock,threadsPerBlock>>>(kernel, reg, dim3(sizex,sizey,1));
                break;
            default:
                contrast_enhance::paganinKernel<<<gridBlock,threadsPerBlock>>>(kernel, beta_delta, wavelength, 
                pixel_objx, pixel_objy, z2, dim3(sizex,sizey,1));
                break;
        }
        // Normalize kernel by maximum value
 		int max = 0;
        stat = cublasIsamax(handle, sizex * sizey, kernel, 1, &max);

        if (stat != CUBLAS_STATUS_SUCCESS){
            printf("Cublas Max failed in Phase Constrast Kernels\n");
            fflush(stdout);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());

		float scale = 1.0f;
		HANDLE_ERROR(cudaMemcpy(&scale, kernel + max, sizeof(float), cudaMemcpyDeviceToHost));

        opt::scale<<<gridBlock,threadsPerBlock>>>(kernel, dim3(sizex,sizey,1), scale);

        HANDLE_ERROR(cudaDeviceSynchronize());
    }

	void getContrastEnhencement(cufftHandle mplan, cufftComplex *proj, float *kernel, 
    dim3 size)
	{	
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);
        
        HANDLE_FFTERROR(cufftExecC2C(mplan, proj, proj, CUFFT_FORWARD));

        contrast_enhance::multiplication<<<gridBlock,threadsPerBlock>>>(proj, kernel, proj, size);

        HANDLE_FFTERROR(cufftExecC2C(mplan, proj, proj, CUFFT_INVERSE));
    }

	void getContrastEnhencementGPU(DIM tomo, GEO geometry, CEF ContrastFilter,
	float *projections, int sizez, int ngpu)
	{	
		/* Initialize GPU device */
		HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Projection data sizes */
        int nrays      = tomo.size.x;
        int nslices    = tomo.size.y;
        int nrayspad   = PDIM(  nrays,tomo.pad.x); // nrays * (1 + tomo.pad.x);
        int nslicespad = PDIM(nslices,tomo.pad.y); // nslices * (1 + tomo.pad.y);

        printf("Here 1 \n");
        fflush(stdout);
		int i, blocksize = tomo.blocksize;

        size_t total_required_mem_per_frame_bytes = 8 * calcPaddedSliceMemoryBytes(tomo);

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize( sizez, 
                                                        total_required_mem_per_frame_bytes, 
                                                        true, 
                                                        BYTES_TO_GB * getTotalDeviceMemory());

            blocksize          = min(sizez, blocksize_aux);
            blocksize          = min(32, blocksize);
        }
        int ind_block = (int)ceil( (float) sizez / blocksize );

        printf("ind_block: %d \n", ind_block);
        printf("sizez: %d \n", sizez);
        printf("blocksize: %d \n", blocksize);
        fflush(stdout);

        /* Kernel Computation */
        size_t nsize   = nrayspad * nslicespad;
		float *kernel  = opt::allocGPU<float>(nsize);

        compute_contrast_kernel(tomo, geometry, ContrastFilter, kernel);
        HANDLE_ERROR(cudaDeviceSynchronize());

		float *dprojections      = opt::allocGPU<float>((size_t) nrays * nslices * blocksize);
        HANDLE_ERROR(cudaDeviceSynchronize());
        cufftComplex *dataPadded = opt::allocGPU<cufftComplex>((size_t) nrayspad * nslicespad * blocksize);
        HANDLE_ERROR(cudaDeviceSynchronize());

        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(dim3(nrayspad,nslicespad,blocksize), threadsPerBlock);

        /* Plan for Fourier transform - cufft */
        cufftHandle mplan;
		int n[] = {nslicespad,nrayspad};
		HANDLE_FFTERROR(cufftPlanMany(&mplan, 2, n, n, 1, nslicespad*nrayspad, n, 1, nslicespad*nrayspad, CUFFT_C2C, blocksize));

		/* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block = 0;

		for (i = 0; i < ind_block; i++){

			subblock  = min(sizez - ptr, blocksize);
			ptr_block = (size_t)nrays * nslices * ptr;

			/* Update pointer */
			ptr = ptr + subblock;

            if( subblock != blocksize){
                HANDLE_ERROR(cudaDeviceSynchronize());
				HANDLE_FFTERROR(cufftDestroy(mplan));
				HANDLE_FFTERROR(cufftPlanMany(&mplan, 2, n, n, 1, nslicespad*nrayspad, n, 1, nslicespad*nrayspad, CUFFT_C2C, subblock));
			}

            opt::CPUToGPU<float>(projections + ptr_block, dprojections, 
                                (size_t)nrays * nslices * subblock);
            
            gridBlock.z = (int)ceil( subblock / TPBZ ) + 1;
            opt::paddR2C<<<gridBlock,threadsPerBlock>>>(dprojections, dataPadded, tomo.padding_mode, 
                                                        dim3(nrays,nslices,subblock), tomo.pad);

			getContrastEnhencement( mplan, dataPadded, kernel, dim3(nrayspad,nslicespad,subblock) );

            opt::remove_paddC2R<<<gridBlock,threadsPerBlock>>>(dataPadded, dprojections, dim3(nrays,nslices,subblock), tomo.pad);
            
            printf("ContrastFilter.post_process = %d \n",ContrastFilter.post_process);
            fflush(stdout);

            if( ContrastFilter.post_process == 1) getlog(dprojections, dim3(nrays,nslices,subblock));

			opt::GPUToCPU<float>(projections + ptr_block, dprojections, 
                                (size_t)nrays * nslices * subblock);
		}
		HANDLE_ERROR(cudaDeviceSynchronize());
        
        /* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(mplan));

        /* Free memory */
        HANDLE_ERROR(cudaFree(dataPadded));
		HANDLE_ERROR(cudaFree(dprojections));
        HANDLE_ERROR(cudaFree(kernel));
	}

    void getContrastEnhencementMultiGPU(DIM tomo, GEO geometry, CEF ContrastFilter,
    int *gpus, int ngpus, float *projections)
	{	
		int i, Maxgpudev;
		
		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		int subvolume = (tomo.size.z + ngpus - 1) / ngpus;
		int subblock, ptr = 0; size_t ptr_volume = 0;

		if (ngpus == 1){ /* 1 device */

			getContrastEnhencementGPU(tomo, geometry, ContrastFilter, projections, subvolume, gpus[0]);

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
				
				threads.push_back( std::async(  std::launch::async, 
												getContrastEnhencementGPU,
                                                tomo, geometry, 
                                                ContrastFilter,
												projections + ptr_volume, 
												subblock, gpus[i]
												));		
                /* Update pointer */
				ptr = ptr + subblock;
			}
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}	

		HANDLE_ERROR(cudaDeviceSynchronize());
	}
}

