#include "../inc/include.h"
#include "../inc/common/complex.hpp"
#include "../inc/common/types.hpp"
#include "../inc/common/operations.hpp"
#include "../inc/common/logerror.hpp"

void set_rebinning_parameters_cpu(PAR *param, float *parameters, size_t *volumesize)
{
	/* Dimensions */
	param->Nx          = volumesize[0]; param->Ny = volumesize[1]; param->Nz = volumesize[2]; 
	param->slice       = param->Nx * param->Ny; // slice size
	param->sizeofblock = volumesize[3]; 
	param->blocksize   = param->Nz;

	/* Distances */
	param->z1 = parameters[0];
	param->z2 = parameters[1];
	param->zt = param->z1 + param->z2;

	/* Rebinning Parameters */
	param->ct = parameters[5 ]; param->cr = parameters[2 ]; 
	param->Dt = parameters[8 ]; param->Dr = parameters[9 ];
	param->Lt = parameters[10]; param->Lr = parameters[11];
	
	param->efft_pixel = parameters[12]; param->effr_pixel = parameters[13];

	param->rt = parameters[6];
	param->rr = parameters[3];
	param->st = parameters[7];
	param->sr = parameters[4];
}

void set_rebinning_parameters_gpu(PAR *param, float *parameters, size_t *volumesize, int *gpus)
{
	/* Dimensions */
	param->Nx          = volumesize[0]; param->Ny = volumesize[1]; param->Nz = volumesize[2]; 
	param->slice       = param->Nx * param->Ny; // slice size
	param->sizeofblock = volumesize[3]; 
	param->blocksize   = param->Nz;

	/* Distances */
	param->z1 = parameters[0];
	param->z2 = parameters[1];
	param->zt = param->z1 + param->z2;

	/* Rebinning Parameters */
	param->ct = parameters[5 ]; param->cr = parameters[2 ]; 
	param->Dt = parameters[8 ]; param->Dr = parameters[9 ];
	param->Lt = parameters[10]; param->Lr = parameters[11];
	
	param->efft_pixel = parameters[12]; param->effr_pixel = parameters[13];

	param->rt = parameters[6];
	param->rr = parameters[3];
	param->st = parameters[7];
	param->sr = parameters[4];

	/* GPUs */
	param->gpus = gpus;

	size_t Nsx = 128, Nsy = 1, Nsz = 1;
	/* Initialize Device sizes variables */	
	param->BT = dim3(Nsx,Nsy,Nsz);
	const int bx = (param->Nx + Nsx - 1)/Nsx;	
	const int by = (param->Ny + Nsy - 1)/Nsy;
	const int bz = (param->Nz + Nsz - 1)/Nsz;
	param->Grd = dim3(bx,by,bz);
}

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

void free_gpu_rebinning(RDAT *reb)
{   /* Free struct CDAT for conebeam with all the GPU variables */
    
    /* GPU */
	HANDLE_ERROR(cudaFree(reb->dctomo));
	HANDLE_ERROR(cudaFree(reb->dtomo ));	

}