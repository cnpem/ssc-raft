#include <cstddef>
#include "common/configs.hpp"
#include "common/logerror.hpp"


extern "C"{
	WKP *allocateWorkspace(CFG configs, size_t tomo_batch_size, size_t obj_batch_size)
	{  /* Allocate struct prain with all GPU the variables */
		WKP *workspace = (WKP *)malloc(sizeof(WKP));

        const size_t tomoptr_size = tomo_batch_size * configs.tomo.size.x * configs.tomo.size.y;
        const size_t objptr_size = obj_batch_size * configs.obj.size.x * configs.obj.size.y;
        const size_t frame_size = tomo_batch_size * configs.tomo.size.x;
        const size_t angles_size = configs.tomo.size.y;

        /* configs.tomo = (nrays,nangles,nslices) */
		/* GPU */
		/* Float */
		HANDLE_ERROR(cudaMalloc((void **)&workspace->obj   , sizeof(float) * objptr_size ));
		HANDLE_ERROR(cudaMalloc((void **)&workspace->tomo  , sizeof(float) * tomoptr_size));
		HANDLE_ERROR(cudaMalloc((void **)&workspace->flat   , sizeof(float) * frame_size ));
		HANDLE_ERROR(cudaMalloc((void **)&workspace->dark   , sizeof(float) * frame_size ));
        HANDLE_ERROR(cudaMalloc((void **)&workspace->angles , sizeof(float) * angles_size));

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

