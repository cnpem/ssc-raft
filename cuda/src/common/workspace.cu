#include "common/configs.hpp"
#include "common/logerror.hpp"


extern "C"{
	WKP *allocateWorkspace(CFG configs, Process process)
	{  /* Allocate struct prain with all GPU the variables */
		WKP *workspace = (WKP *)malloc(sizeof(WKP));

        /* configs.tomo = (nrays,nangles,nslices) */
		/* GPU */
		/* Float */
		HANDLE_ERROR(cudaMalloc((void **)&workspace->obj   , sizeof(float) *                             process.objptr_size )); 
		HANDLE_ERROR(cudaMalloc((void **)&workspace->tomo  , sizeof(float) *                            process.tomoptr_size )); 
		HANDLE_ERROR(cudaMalloc((void **)&workspace->flat  , sizeof(float) * (size_t)process.tomobatch_size * configs.tomo.x ));
		HANDLE_ERROR(cudaMalloc((void **)&workspace->dark  , sizeof(float) * (size_t)process.tomobatch_size * configs.tomo.x ));
        HANDLE_ERROR(cudaMalloc((void **)&workspace->angles, sizeof(float) *                                  configs.tomo.y ));
        
		return workspace;
	}
}


extern "C"{
	void freeWorkspace(WKP *workspace, CFG configs)
	{  /* Free struct prain with all the GPU variables */
		
		/* GPU */
        HANDLE_ERROR(cudaFree(workspace->obj   ));
		HANDLE_ERROR(cudaFree(workspace->tomo  ));
		HANDLE_ERROR(cudaFree(workspace->flat  ));	
		HANDLE_ERROR(cudaFree(workspace->dark  ));
		HANDLE_ERROR(cudaFree(workspace->angles));

		free(workspace);
	}
}

