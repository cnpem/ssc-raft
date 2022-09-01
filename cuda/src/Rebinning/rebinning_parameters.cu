#include "../../inc/include.h"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"

extern "C"{
	void set_rebinning_parameters_cpu(PAR *param, float *parameters, size_t *volumesize)
	{
		/* Dimensions */
		param->Nx          = volumesize[0]; 
		param->Ny          = volumesize[1]; 
		param->Nz          = volumesize[2];
		param->sizeofblock = volumesize[3]; 
 
		param->slice       = param->Nx * param->Ny; // slice size
		param->blocksize   = param->Nz;

		/* Distances */
		param->z1x         = parameters[0];
		param->z1y         = parameters[1];
		param->z2x         = parameters[2];
		param->z2y         = parameters[3];
		param->d1x         = parameters[4];
		param->d1y         = parameters[5];
		param->d2x         = parameters[6];
		param->d2y         = parameters[7];
		param->pixelDetx   = parameters[8];
		param->pixelDety   = parameters[9];

		param->magnx       = ( param->z1x + param->z2x ) / param->z1x;
		param->magny       = ( param->z1y + param->z2y ) / param->z1y;
		param->mx          = ( param->d1x + param->d2x ) / param->d1x;
		param->my          = ( param->d1y + param->d2y ) / param->d1y;
		
		param->effa_pixel  = param->pixelDetx / param->magnx; 
		param->effb_pixel  = param->pixelDety / param->magny;

	}
}

extern "C"{
	void set_rebinning_parameters_gpu(PAR *param, float *parameters, size_t *volumesize, int *gpus)
	{
		/* Dimensions */
		param->Nx          = volumesize[0]; 
		param->Ny          = volumesize[1]; 
		param->Nz          = volumesize[2]; 
		param->sizeofblock = volumesize[3]; 

		param->slice       = param->Nx * param->Ny; // slice size
		param->blocksize   = param->Nz;

		/* Distances */
		param->z1x         = parameters[0];
		param->z1y         = parameters[1];
		param->z2x         = parameters[2];
		param->z2y         = parameters[3];
		param->d1x         = parameters[4];
		param->d1y         = parameters[5];
		param->d2x         = parameters[6];
		param->d2y         = parameters[7];
		param->pixelDetx   = parameters[8];
		param->pixelDety   = parameters[9];

		param->magnx       = ( param->z1x + param->z2x ) / param->z1x;
		param->magny       = ( param->z1y + param->z2y ) / param->z1y;
		param->mx          = ( param->d1x + param->d2x ) / param->d1x;
		param->my          = ( param->d1y + param->d2y ) / param->d1y;
		
		param->effa_pixel  = ( param->pixelDetx / param->magnx ) / param->mx; 
		param->effb_pixel  = ( param->pixelDety / param->magny ) / param->my;

		/* GPUs */
		param->gpus        = gpus;

		size_t Nsx         = 128; 
		size_t Nsy         = 1;
		size_t Nsz         = 1;
		/* Initialize Device sizes variables */	
		param->BT          = dim3(Nsx,Nsy,Nsz);
		const int bx       = ( param->Nx + Nsx ) / Nsx + 1;	
		const int by       = ( param->Ny + Nsy ) / Nsy + 1;
		const int bz       = ( param->Nz + Nsz ) / Nsz + 1;
		param->Grd         = dim3(bx,by,bz);
	}
}

extern "C"{
	RDAT *allocate_gpu_rebinning(RDAT *workspace, PAR param)
	{   /* Allocate struct RDAT for conebeam rebinning with all GPU the variables */
		RDAT *reb = workspace;

		/* GPU */
		size_t n = param.Nx * param.Ny * param.Nz;
		/* Float */
		HANDLE_ERROR(cudaMalloc((void **)&reb->dctomo, sizeof(float) * n)); 
		HANDLE_ERROR(cudaMalloc((void **)&reb->dtomo , sizeof(float) * n)); 

		return reb;
	}
}

extern "C"{
	void free_gpu_rebinning(RDAT *reb)
	{   /* Free struct CDAT for conebeam with all the GPU variables */
		
		/* GPU */
		HANDLE_ERROR(cudaFree(reb->dctomo));
		HANDLE_ERROR(cudaFree(reb->dtomo ));	

	}
}