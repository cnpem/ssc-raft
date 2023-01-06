#include "../../../inc/include.h"
#include "../../../inc/common/kernel_operators.hpp"
#include "../../../inc/common/complex.hpp"
#include "../../../inc/common/types.hpp"
#include "../../../inc/common/operations.hpp"
#include "../../../inc/common/logerror.hpp"

extern "C"{

	void set_gc_parameters(PAR *param, float *parameters, size_t *iterations, size_t *volumesize, int *devices)
	{
		/* GPUs */
		/* Initialize Device sizes variables */
		param->devices        = devices;	
		size_t Nsx            = 128;
		size_t Nsy            = 1; 
		size_t Nsz            = 1;
		param->BT             = dim3(Nsx,Nsy,Nsz);
		const int bx          = ( param->Npadx + Nsx - 1 ) / Nsx;	
		const int by          = ( param->Npady + Nsy - 1 ) / Nsy;
		const int bz          = ( param->Nz    + Nsz - 1 ) / Nsz;
		param->Grd            = dim3(bx,by,bz);
	}

}

extern "C"{

	GC *allocate_gc_parameters(GC *workspace, PAR param)
	{  /* Allocate struct GC with all GPU and CPU variables */
		GC *raft = workspace;
		size_t n = param.slice       * param.zblock;
		size_t m = param.slicePadded * param.zblock;

		/* GPU */
		/* Float */
		HANDLE_ERROR(cudaMalloc((void **)&raft->volume      , sizeof(float) * n )); 

      /* CPU */
		/* Float */
		HANDLE_ERROR(cudaMalloc((void **)&raft->volumePadded, sizeof(float) * m )); 

		return raft;
	}

}

extern "C"{

	void free_gc_parameters(GC *raft, PAR param)
	{  /* Free struct GC with all the GPU and CPU variables */
		
		/* GPU */
		HANDLE_ERROR(cudaFree(raft->volume       ));

      /* CPU */
		HANDLE_ERROR(cudaFree(prain->volumePadded));

	}
   
}

