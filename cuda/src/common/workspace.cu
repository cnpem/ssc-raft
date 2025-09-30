#include <cstddef>
#include "common/configs.hpp"
#include "common/opt.hpp"
#include "common/logerror.hpp"


extern "C"{
	WKP *Initialize_workspace(dim3 size_tomo, dim3 size_obj, 
    dim3 size_flat, dim3 size_dark, 
    dim3 tomo_pad, dim3 obj_pad, int nangles)
	{  
        /* Allocate the local GPU variables:
        size_tomo: (nrays,nangles,nslices_gpu_block) = (size_tomo.x, size_tomo.y, size_tomo.z)
        size_obj:  (nrays,  nrays,nslices_gpu_block) = ( size_obj.x,  size_obj.y,  size_obj.z)

        size_data: (nrays,nslices_gpu_block,nangles) = (size_tomo.x, size_tomo.z, size_tomo.y)
        size_flat: (size_tomo.x, size_tomo.z, nflats) = (nrays, nslices_gpu_block, nflats)
        size_dark: (size_tomo.x, size_tomo.z,      1) = (nrays, nslices_gpu_block,      1)

        nangles: True dimension size of angles (related to excentric measurements)
        */
		WKP *workspace = (WKP *)malloc(sizeof(WKP));

        const size_t flatptr_size = size_flat.x * size_flat.y * size_flat.z;
        const size_t darkptr_size = size_dark.x * size_dark.y; 

        const size_t tomoptr_size = size_tomo.x * size_tomo.y * size_tomo.z;
        const int tomosizepadx    = PDIM(size_tomo.x,tomo_pad.x); 
        const size_t tomoptr_pad  = tomosizepadx * size_tomo.y * size_tomo.z;

        const size_t objptr_size  = 2 * size_obj.x *  2 * size_obj.y *  size_obj.z;
        const int objsizepadx     = PDIM( size_obj.x, size_obj.x); 
        const int objsizepady     = PDIM( size_obj.y, size_obj.y); 
        const size_t objptr_pad   = objsizepadx * objsizepady *  size_obj.z;

        const size_t angles_size  = nangles;

        workspace->obj      = opt::allocGPU<float>( objptr_size);
        workspace->tomo     = opt::allocGPU<float>(tomoptr_size);
        // workspace->data     = opt::allocGPU<float>(tomoptr_size);
        workspace->flat     = opt::allocGPU<float>(flatptr_size);
        workspace->dark     = opt::allocGPU<float>(darkptr_size);
        workspace->angles   = opt::allocGPU<float>( angles_size);

        workspace->objPadd  = opt::allocGPU<float>( objptr_pad);
        workspace->tomoPadd = opt::allocGPU<float>(tomoptr_pad);

		return workspace;
	}
}

extern "C"{
    void freeWorkspace(WKP *workspace)
	{  /* Deallocate the GPU variables */

		/* GPU */
        HANDLE_ERROR(cudaFree(workspace->obj     ));
        HANDLE_ERROR(cudaFree(workspace->objPadd ));
		HANDLE_ERROR(cudaFree(workspace->tomo    ));
        HANDLE_ERROR(cudaFree(workspace->tomoPadd));
        // HANDLE_ERROR(cudaFree(workspace->data    ));
		HANDLE_ERROR(cudaFree(workspace->flat    ));
		HANDLE_ERROR(cudaFree(workspace->dark    ));
		HANDLE_ERROR(cudaFree(workspace->angles  ));

		free(workspace);
	}
}

