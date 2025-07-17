#include <cublas_v2.h>

#include "processing/filters.hpp"
#include "processing/processing.hpp"
#include "common/opt.hpp"

extern "C" {

    void compute_contrast_kernel(DIM tomo, GEO geometry, CEF PaganinFilter, float *kernel)
    {
        /* Data sizes */
        int sizex        = tomo.size.x * (1 + tomo.pad.x);
        int sizey        = tomo.size.y * (1 + tomo.pad.y);

        float z2         = geometry.z2x;
        float pixel_objx = geometry.obj_pixel_x;
        float pixel_objy = geometry.obj_pixel_y;
        float wavelength = geometry.wavelength;
        float beta_delta = PaganinFilter.beta_delta;

		cublasHandle_t handle = NULL;
        cublasCreate(&handle);
        cublasStatus_t stat;

        dim3 threadsPerBlock(TPBX,TPBY,1);
        dim3 gridBlock = opt::setGridBlock(dim3(sizex,sizey,1), threadsPerBlock);

		switch (PaganinFilter.method){
            case contrast_enhance::ContrastEnhanceType::paganin:
                /* code */
                contrast_enhance::paganinKernel<<<gridBlock,threadsPerBlock>>>(kernel, beta_delta, wavelength, 
                pixel_objx, pixel_objy, z2, dim3(sizex,sizey,1));
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

	void getContrastEnhencement(cufftHandle mplan, 
    float *projections, float *kernel, dim3 size, dim3 size_pad, dim3 pad)
	{	
        contrast_enhance::apply_contrast_filter(mplan, projections, kernel, size, size_pad, pad);
    }

	void getContrastEnhencementGPU(DIM tomo, GEO geometry, CEF PaganinFilter,
	float *projections, int sizez, int ngpu)
	{	
		/* Initialize GPU device */
		HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Projection data sizes */
        int nrays      = tomo.size.x;
        int nslices    = tomo.size.y;
        int nrayspad   =   nrays * (1 + tomo.pad.x);
        int nslicespad = nslices * (1 + tomo.pad.y);

        /* Kernel Computation */

        size_t nsize   = nrayspad * nslicespad;
		float *kernel  = opt::allocGPU<float>(nsize);

        compute_contrast_kernel(tomo, geometry, PaganinFilter, kernel);

		int i; 
        int blocksize = tomo.blocksize;

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

		float *dprojections = opt::allocGPU<float>((size_t) nrays * nslices * blocksize);

        cufftHandle mplan;
        /* Plan for Fourier transform - cufft */
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

			getContrastEnhencement( mplan, dprojections, kernel,
                      dim3(nrays, nslices, subblock), 
                      dim3(nrayspad, nslicespad, subblock),
                      tomo.pad
                    );

			opt::GPUToCPU<float>(projections + ptr_block, dprojections, 
                                (size_t)nrays * nslices * subblock);

		}
		HANDLE_ERROR(cudaDeviceSynchronize());
        
        /* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(mplan));

        /* Free memory */
		HANDLE_ERROR(cudaFree(dprojections));
        HANDLE_ERROR(cudaFree(kernel));

	}

    void getContrastEnhencementMultiGPU(DIM tomo, GEO geometry, CEF PaganinFilter,
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

			getContrastEnhencementGPU(tomo, geometry, PaganinFilter, projections, subvolume, gpus[0]);

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
												getContrastEnhencementGPU,
                                                tomo, geometry, 
                                                PaganinFilter,
												projections + ptr_volume, 
												subblock, gpus[i]
												));		

			}
		
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}	

		HANDLE_ERROR(cudaDeviceSynchronize());
	}
}

