#include "../../inc/include.h"
#include "../../inc/common/configs.h"
#include "../../inc/common/logerror.hpp"

extern "C"{
	WKP *allocateWorkspace(CFG configs, Process process)
	{  /* Allocate struct prain with all GPU the variables */
		WKP *workspace = (WKP *)malloc(sizeof(WKP));

		/* GPU */
		/* Float */
		HANDLE_ERROR(cudaMalloc((void **)&workspace->recon , sizeof(float) * process.n_recon                                      )); 
		HANDLE_ERROR(cudaMalloc((void **)&workspace->tomo  , sizeof(float) * process.n_tomo                                       )); 
		HANDLE_ERROR(cudaMalloc((void **)&workspace->flat  , sizeof(float) * (size_t)process.batch_size_tomo * configs.tomo.nrays ));
		HANDLE_ERROR(cudaMalloc((void **)&workspace->dark  , sizeof(float) * (size_t)process.batch_size_tomo * configs.tomo.nrays ));
        HANDLE_ERROR(cudaMalloc((void **)&workspace->angles, sizeof(float) * configs.tomo.nangles                                 ));

		return workspace;
	}
}


extern "C"{
	void freeWorkspace(WKP *workspace, CFG configs)
	{  /* Free struct prain with all the GPU variables */
		
		/* GPU */
        HANDLE_ERROR(cudaFree(workspace->recon ));
		HANDLE_ERROR(cudaFree(workspace->tomo  ));
		HANDLE_ERROR(cudaFree(workspace->flat  ));	
		HANDLE_ERROR(cudaFree(workspace->dark  ));
		HANDLE_ERROR(cudaFree(workspace->angles));

		free(workspace);

	}
}

