#include "../../../../inc/include.h"
#include "../../../../inc/common/types.hpp"
#include "../../../../inc/common/kernel_operators.hpp"
#include "../../../../inc/common/complex.hpp"
#include "../../../../inc/common/operations.hpp"
#include "../../../../inc/common/logerror.hpp"

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

		if (ngpus == 1){ /* 1 device */
			_phase_filters_threads(param, projections, nrays, nangles, nslices, gpus[0]);
		}else{
		/* Launch async Threads for each device.
			Each device solves a block of 'nrays * nslices' size.
		*/
			// See future c++ async launch
			std::vector<std::future<void>> threads = {};

			for (i = 0; i < ngpus; i++){
				
				if(subvolume*(i+1) > nangles)
					subvolume = nangles - subvolume*i;

				if(subvolume < 1)
					continue;
				
				threads.push_back( std::async( std::launch::async, _phase_filters_threads, param, projections + (size_t)nrays * nslices * subvolume * i, (size_t)nrays, (size_t)subvolume, (size_t)nslices, gpus[i]));		
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
		
		size_t bz; 
		param.blocksize = min(nangles,param.blocksize);

		printf("Filter: %d \n",(int)param.filter);
		printf("Dims: %ld, %ld, %ld \n",param.Npadx,param.Npady,param.blocksize);
		printf("Dims: %e, %e, %e, %e, %e, %e \n",param.z1x,param.z1y,param.z2x,param.z2y,param.energy,param.lambda);
		
		/* Plan for Fourier transform - cufft */
		int n[] = {(int)param.Npady,(int)param.Npadx};
		HANDLE_FFTERROR(cufftPlanMany(&param.mplan , 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, param.blocksize)); //wrong

		size_t zblock = param.blocksize;

		/* Loop for each batch of size 'batch' in threads */

		for (bz = 0; bz < nangles; bz += param.blocksize){

			zblock = min((size_t)fabs(nangles - bz),param.blocksize);

			if( zblock != param.blocksize){

				HANDLE_FFTERROR(cufftDestroy(param.mplan));

				HANDLE_FFTERROR(cufftPlanMany(&param.mplan , 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, zblock)); //wrong
			}

			switch ((int)param.filter){
				case 0:
					/* code */
					printf("No filter was selected!");
					break;
				case 1:
					/* code */
					_paganin_gpu(param, projections + nrays * nslices * bz, nrays, zblock, nslices);
					break;
				case 2:
					/* code */
					_bronnikov_gpu(param, projections + nrays * nslices * bz, nrays, zblock, nslices);
					break;
				case 3:
					/* code */
					_born_gpu(param, projections + nrays * nslices * bz, nrays, zblock, nslices);
					break;
				case 4:
					/* code */
					_rytov_gpu(param, projections + nrays * nslices * bz, nrays, zblock, nslices);
					break;

				default:
					printf("Using paganin as default phase filter!");
					_paganin_gpu(param, projections + nrays * nslices * bz, nrays, zblock, nslices);
					break;
			}	
			
		}

		/* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(param.mplan));

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

		param->Npadx     = sizex + 2.0 * param->padx; 
		param->Npady     = sizey + 2.0 * param->pady; 

		param->lambda    = ( plank * vc ) / param->energy;
		param->wave      = ( 2.0 * float(M_PI) ) / param->lambda;

		/* GPUs */
		/* Initialize Device sizes variables */
		size_t Nsx      = 128;
		size_t Nsy      = 1; 
		size_t Nsz      = 1;

		param->BT       = dim3(Nsx,Nsy,Nsz);
        const int bx    = ( param->Npadx + Nsx - 1 ) / Nsx;	
		const int by    = ( param->Npady + Nsy - 1 ) / Nsy;
		const int bz    = ( sizez + Nsz - 1 ) / Nsz;
		param->Grd      = dim3(bx,by,bz);
	}

}
