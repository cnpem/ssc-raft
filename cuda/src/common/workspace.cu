#include <cstddef>
#include "common/configs.hpp"
#include "common/opt.hpp"
#include "common/logerror.hpp"


extern "C"{
	WKP *Initialize_workspace(CFG configs, size_t tomo_batch_size, size_t obj_batch_size)
	{  /* Allocate struct prain with all GPU the variables */
		WKP *workspace = (WKP *)malloc(sizeof(WKP));

        const size_t tomoptr_size =  tomo_batch_size * configs.tomo.size.x * configs.tomo.size.y;
        const size_t objptr_size  =   obj_batch_size *  configs.obj.size.x *  configs.obj.size.y;
        const size_t flatptr_size =   configs.nflats *     tomo_batch_size * configs.tomo.size.x;
        const size_t darkptr_size =  tomo_batch_size *                       configs.tomo.size.x;

        const int tomosizepadx = PDIM(configs.tomo.size.x,configs.tomo.pad.x); // configs.tomo.size.x * ( 1 + configs.tomo.pad.x );
        const int objsizepadx  = PDIM( configs.obj.size.x, configs.obj.pad.x); // configs.obj.size.x * ( 1 +  configs.obj.pad.x );
        const int objsizepady  = PDIM( configs.obj.size.y, configs.obj.pad.y); // configs.obj.size.y * ( 1 +  configs.obj.pad.y );

        const size_t tomoptr_padsize =  tomo_batch_size * tomosizepadx * configs.tomo.size.y;
        const size_t objptr_padsize  =   obj_batch_size *  objsizepadx *         objsizepady;

        const size_t angles_size  = configs.tomo.size.y;

        /* configs.tomo = (nrays,nangles,nslices) */
		/* GPU */
		/* Float */
        workspace->obj    = opt::allocGPU<float>( objptr_size);
        workspace->tomo   = opt::allocGPU<float>(tomoptr_size);
        workspace->flat   = opt::allocGPU<float>(flatptr_size);
        workspace->dark   = opt::allocGPU<float>(darkptr_size);
        workspace->angles = opt::allocGPU<float>( angles_size);

        workspace->objPadd  = opt::allocGPU<float>( objptr_padsize);
        workspace->tomoPadd = opt::allocGPU<float>(tomoptr_padsize);

		return workspace;
	}
}

extern "C"{
    void freeWorkspace(WKP *workspace)
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

