#include "../../../../inc/include.h"
#include "../../../../inc/common/types.hpp"
#include "../../../../inc/common/kernel_operators.hpp"
#include "../../../../inc/common/complex.hpp"
#include "../../../../inc/common/operations.hpp"
#include "../../../../inc/common/logerror.hpp"

# define vc 299792458           /* Velocity of Light [m/s] */ 
# define plank 4.135667662E-15  /* Plank constant [ev*s] */

extern "C" {

	void phase_filters(float *projections, float *paramf, size_t *parami, 
				int nrays, int nangles, int nslices,
				int *gpus, int ngpus)
	{	
		int i, Maxgpudev;
		
		/* Multiples devices */
		cudaGetDeviceCount(&Maxgpudev);

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		PAR param;

		set_phase_filters_parameters(&param, paramf, parami, nrays, nslices, nangles);

		int subvolume = (nangles + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 
		size_t ptr_volume = 0;

		if (ngpus == 1){ /* 1 device */

			_phase_filters_threads(param, projections, nrays, nangles, nslices, gpus[0]);

		}else{
		/* Launch async Threads for each device.
			Each device solves a block of 'nrays * nangles' size.
		*/
			// See future c++ async launch
			std::vector<std::future<void>> threads = {};

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(nangles - ptr, subvolume);
				ptr_volume = (size_t)nrays * nslices * ptr;
				
				// printf("Sub[%d] = %d () \n",i,subblock,ptr);

				/* Update pointer */
				ptr = ptr + subblock;
				
				threads.push_back( std::async( std::launch::async, _phase_filters_threads, param, projections + ptr_volume, (size_t)nrays, (size_t)subblock, (size_t)nslices, gpus[i]));		

			}
		
			// Log("Synchronizing all threads...\n");
		
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}	

		cudaDeviceSynchronize();
	}

	void _phase_filters_threads(PAR param, float *projections, size_t nrays, size_t nangles, size_t nslices, int ngpu)
	{	
		/* Initialize GPU device */
		HANDLE_ERROR(cudaSetDevice(ngpu))

		size_t npad = param.Npadx * param.Npady;
		float *d_kernel;
		cublasHandle_t handle = NULL;
        cublasCreate(&handle);
        cublasStatus_t stat;

		// Compute phase filter kernel
		HANDLE_ERROR(cudaMalloc((void **)&d_kernel, sizeof(float) * npad ));

		switch ((int)param.filter){
				case 0:
					/* code */
					printf("No filter was selected!");
					break;
				case 1:
					/* code */
					paganinKernel<<<param.Grd,param.BT>>>(d_kernel, param, param.Npadx, param.Npady, nangles);
					break;
				case 2:
					/* code */
					bronnikovKernel<<<param.Grd,param.BT>>>(d_kernel, param, param.Npadx, param.Npady, nangles);
					break;
				case 3:
					/* code */
					bornKernel<<<param.Grd,param.BT>>>(d_kernel, param, param.Npadx, param.Npady, nangles);
					break;
				case 4:
					/* code */
					rytovKernel<<<param.Grd,param.BT>>>(d_kernel, param, param.Npadx, param.Npady, nangles);
					break;

				default:
					paganinKernel<<<param.Grd,param.BT>>>(d_kernel, param, param.Npadx, param.Npady, nangles);
					break;
			}

        // Normalize kernel by maximum value
 		int max;
        stat = cublasIsamax(handle, (int)npad, d_kernel, 1, &max);

        if (stat != CUBLAS_STATUS_SUCCESS)
            printf("Cublas Max failed\n");

		float maximum;
		HANDLE_ERROR(cudaMemcpy(&maximum, d_kernel + max, sizeof(float), cudaMemcpyDeviceToHost));
        Normalize<<<param.Grd,param.BT>>>(d_kernel, maximum, param.Npadx, param.Npady);

		cudaDeviceSynchronize();

		size_t bz; 
		param.blocksize = min(nangles,param.blocksize);

		// printf("Filter: %d \n",(int)param.filter);
		// printf("Dims: %ld, %ld, %ld \n",param.Npadx,param.Npady,param.blocksize);
		// printf("Dims: %e, %e, %e, %e, %e, %e \n",param.z1x,param.z1y,param.z2x,param.z2y,param.energy,param.lambda);
		
		/* Plan for Fourier transform - cufft */
		int n[] = {(int)param.Npadx,(int)param.Npady};
		HANDLE_FFTERROR(cufftPlanMany(&param.mplan, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, (int)param.blocksize));

		size_t zblock = param.blocksize;

		size_t ind_block = (size_t)ceil( (float) nangles / param.blocksize );
		int ptr = 0;
		size_t ptr_block = 0;

		/* Loop for each batch of size 'batch' in threads */

		// printf("GPU: %d; zblock = %ld; indBlock = %ld \n", ngpu, zblock, ind_block);

		for (bz = 0; bz < ind_block; bz++){

			zblock    = min(nangles - ptr, param.blocksize);
			ptr_block = (size_t)nrays * nslices * ptr;

			// printf("zblock[%d,%ld] = %ld, ptr = %d (%d) \n", ngpu, bz, zblock, ptr, nangles - ptr);

			/* Update pointer */
			ptr = ptr + zblock;

			if( zblock != param.blocksize){

				HANDLE_FFTERROR(cufftDestroy(param.mplan));

				HANDLE_FFTERROR(cufftPlanMany(&param.mplan, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, (int)zblock));
			}
				
			switch ((int)param.filter){
				case 0:
					/* code */
					printf("No filter was selected!");
					break;
				case 1:
					/* code */
					_paganin_gpu(param, projections + ptr_block, d_kernel, nrays, zblock, nslices);
					break;
				case 2:
					/* code */
					_bronnikov_gpu(param, projections + ptr_block, d_kernel, nrays, zblock, nslices);
					break;
				case 3:
					/* code */
					_born_gpu(param, projections + ptr_block, d_kernel, nrays, zblock, nslices);
					break;
				case 4:
					/* code */
					_rytov_gpu(param, projections + ptr_block, d_kernel, nrays, zblock, nslices);
					break;

				default:
					printf("Using paganin as default phase filter!");
					_paganin_gpu(param, projections + ptr_block, d_kernel, nrays, zblock, nslices);
					break;
			}	
		}
		cudaDeviceSynchronize();

		/* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(param.mplan));
		cudaFree(d_kernel);
		cublasDestroy(handle);

		cudaDeviceSynchronize();
	}

	void set_phase_filters_parameters(PAR *param, float *paramf, size_t *parami, size_t sizex, size_t sizey, size_t sizez)
	{
		/* Initialize paganin parameters */
		param->z1x       = paramf[0]; // z1x;
		param->z1y       = paramf[1]; // z1y;
		param->z2x       = paramf[2]; // z2x;
		param->z2y       = paramf[3]; // z2y;
		param->energy    = paramf[4]; // energy;
		param->alpha     = paramf[5]; // alpha;		

		/* Dimensions */
		param->padx      = parami[0]; // padx;
		param->pady      = parami[1]; // pady;
		param->blocksize = parami[2]; // blocksize;

		param->filter    = (int)parami[3]; // filter type;

		param->lambda    = ( plank * vc ) / param->energy;
		param->wave      = ( 2.0 * float(M_PI) ) / param->lambda;

		param->Npadx     = sizex + 2.0 * param->padx; 
		param->Npady     = sizey + 2.0 * param->pady; 


		/* GPUs */
		/* Initialize Device sizes variables */
		size_t Nsx      = 16;
		size_t Nsy      = 16; 
		size_t Nsz      = 1;

		param->BT       = dim3(Nsx,Nsy,Nsz);
        const int bx    = ( param->Npadx + Nsx - 1 ) / Nsx;	
		const int by    = ( param->Npady + Nsy - 1 ) / Nsy;
		const int bz    = ( sizez        + Nsz - 1 ) / Nsz;
		param->Grd      = dim3(bx,by,bz);
	}

}

