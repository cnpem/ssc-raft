#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{
	WKP *allocateWorkspace(CFG configs, int blocksize)
	{  /* Allocate struct prain with all GPU the variables */
		WKP *workspace = (WKP *)malloc(sizeof(WKP));

		size_t n = configs.nangles * configs.nrays * blocksize;
		size_t m = configs.nx      * configs.ny    * blocksize;

		/* GPU */
		/* Float */
		HANDLE_ERROR(cudaMalloc((void **)&workspace->recon , sizeof(float) * m               )); 
		HANDLE_ERROR(cudaMalloc((void **)&workspace->tomo  , sizeof(float) * n               )); 
		HANDLE_ERROR(cudaMalloc((void **)&workspace->flat  , sizeof(float) * n               ));
		HANDLE_ERROR(cudaMalloc((void **)&workspace->dark  , sizeof(float) * n               ));
        HANDLE_ERROR(cudaMalloc((void **)&workspace->angles, sizeof(float) * configs.nangles ));

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

