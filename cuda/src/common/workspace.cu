#include "../../inc/include.h"
#include "../../inc/common/config.h"
#include "../../inc/common/workspace.h"
#include "../../inc/pipeline/process.h"
#include "../../inc/common/logerror.hpp"



extern "C"{
	WKP *allocateWorkspace(CFG configs, Process process)
	{  /* Allocate struct prain with all GPU the variables */
		WKP *workspace = (WKP *)malloc(sizeof(WKP));

		size_t n = process.n_tomo;
		size_t m = process.n_recon;

		/* GPU */
		/* Float */
		HANDLE_ERROR(cudaMalloc((void **)&workspace->recon , sizeof(float) * m                   )); 
		HANDLE_ERROR(cudaMalloc((void **)&workspace->tomo  , sizeof(float) * n                   )); 
		HANDLE_ERROR(cudaMalloc((void **)&workspace->flat  , sizeof(float) * n                   ));
		HANDLE_ERROR(cudaMalloc((void **)&workspace->dark  , sizeof(float) * n                   ));
        HANDLE_ERROR(cudaMalloc((void **)&workspace->angles, sizeof(float) * configs.tomo.size.y ));

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

