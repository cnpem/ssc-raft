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

		size_t n        = configs.nangles * configs.nrays * blocksize;
		size_t m        = configs.nx * configs.ny * blocksize;

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

extern "C"{
	Process *setProcesses(CFG configs, int i, int n_total_processes, int *gpus, int ngpus)
	{  
        int index;

        Process *process = (Process *)malloc(sizeof(Process) * n_total_processes);
             
        if( configs.isconical == 1){
            for (index = 0; index < n_total_processes; index++)
                setProcessConical(configs, &process[index], index, n_total_processes, gpus, ngpus);
        }else{
            for (index = 0; index < n_total_processes; index++)
                setProcessParallel(configs, &process[index], index, n_total_processes, gpus, ngpus);
        } 

        return process; 
                
	}
}


extern "C"{
    void setProcessParallel(CFG configs, Process* process, int index, int n_total_processes, int *gpus, int ngpus)
    {   
        /* Declare variables */
        long long int  n_recon, n_tomo, ind_recon, ind_tomo;
        int indz, indz_max, nz_block;  

        /* Set indexes */
        nz_block             = (int) ( configs.nslices / n_total_processes ); 
        indz                 = index * nz_block;
        indz_max             = (int) std::min( ( index + 1 ) * nz_block, configs.nslices);

        /* Indexes for Reconstruction division - same as Tomogram division */
        n_recon              = (long long int) ( indz_max - indz ) * configs.nx * configs.ny;
        ind_recon            = (long long int) indz                * configs.nx * configs.ny;

        /* Indexes for Tomogram division - same as Reconstruction division */
        n_tomo               = (long long int) ( indz_max - indz ) * configs.nrays * configs.nangles;
        ind_tomo             = (long long int) indz                * configs.nrays * configs.nangles;

        /* Set process struct */

        /* Tomogram division */
        (*process).index     = index;
        (*process).index_gpu = (int) gpus[index%ngpus];  
        (*process).indv      = indz;
        (*process).ind_tomo  = ind_tomo;
        (*process).n_tomo    = n_tomo;

        /* Reconstruction division */
        (*process).n_recon   = n_recon;
        (*process).ind_recon = ind_recon;
        
    }
}

extern "C"{
    void setProcessConical(CFG configs, Process* process, int index, int n_total_processes, int *gpus, int ngpus)
    {   
        /* Declare variables */
        int nz_block;

        /* Reconstruction */
        long long int n_recon, ind_recon;
        int indz, indz_max; 
        float z, z_max;
        
        /* Tomogram and reconstruction filter (v of vertical) */
        long long int n_tomo, ind_tomo, n_filter, ind_filter;
        int indv, indv_max, indv_filter;
        float v, v_max, L;

        /* Reconstruction (or object) variables */ 
        nz_block    = (int) ( ( configs.slice_recon_end - configs.slice_recon_start ) / n_total_processes ); 
        indz        = configs.slice_recon_start + index * nz_block;
        indz_max    = (int) std::min(configs.slice_recon_start + (index + 1) * nz_block, configs.slice_recon_end); 

        n_recon     = (long long int) ( indz_max -                      indz ) * configs.nx * configs.ny;
        ind_recon   = (long long int) ( indz     - configs.slice_recon_start ) * configs.nx * configs.ny;

        z           = - configs.z +     indz * configs.dz;
        z_max       = - configs.z + indz_max * configs.dz;
                
        L           = sqrtf( configs.x * configs.x + configs.y * configs.y );
        
        /* Tomogram (or detector) and filter (with padding) variables */

        /* Tomogram */
        v           = std::max(- configs.v, std::min( configs.z12 *     z / (configs.z1 - L), configs.z12 *     z / (configs.z1 + L) ));
        v_max       = std::min(+ configs.v, std::max( configs.z12 * z_max / (configs.z1 + L), configs.z12 * z_max / (configs.z1 - L) )); 

        indv        = std::max(         0, (int) floor( (v     + configs.v) / configs.dv ));
        indv_max    = std::min(configs.nv, (int)  ceil( (v_max + configs.v) / configs.dv ));

        n_tomo      = (long long int) ( indv_max -                     indv) * ( configs.nangles * configs.nrays );
        ind_tomo    = (long long int) ( indv     - configs.slice_tomo_start) * ( configs.nangles * configs.nrays );

        /* Set process struct */

        /* Reconstruction Filter */
        n_filter    = (long long int) ( indv_max -                     indv ) * ( configs.nangles * configs.nprays );
        ind_filter  = (long long int) ( indv     - configs.slice_tomo_start ) * ( configs.nangles * configs.nprays );
        indv_filter = (          int) (                             n_filter) / ( configs.nangles * configs.nprays );

        /* Tomogram division */
        (*process).index       = index;
        (*process).index_gpu   = (int) gpus[index % ngpus];  
        (*process).indv        = indv;
        (*process).ind_tomo    = ind_tomo;
        (*process).n_tomo      = n_tomo;

        /* Reconstruction Filter division */
        (*process).n_filter    = n_filter;
        (*process).ind_filter  = ind_filter;
        (*process).indv_filter = indv_filter; 

        /* Reconstruction division */
        (*process).n_recon     = n_recon;
        (*process).ind_recon   = ind_recon;
        (*process).z           = z;
        (*process).z_det       = - configs.v + indv * configs.dv;
        
    }
}

extern "C"{
    int getTotalProcesses(CFG configs, int ngpus)
    {
        long double mem_gpu, mem_recon, mem_tomo;
        int n_total_process;
        long long int n_tomo, n_recon;
        
        n_tomo  = (long long int)(configs.nslices)*(long long int)(configs.nangles)*(long long int)(configs.nprays);
        n_recon = (long long int)(configs.nx)*(long long int)(configs.ny)*(long long int)(configs.nz);

        mem_gpu = 40;
        mem_tomo = 32*n_tomo*1.16*(pow(10,-10));
        mem_recon = 32*n_recon*1.16*(pow(10,-10));

        n_total_process = (int) std::ceil((mem_tomo + mem_recon)/mem_gpu);

        // divisão de processos 
        if(n_total_process < ngpus) n_total_process = ngpus;

        if(configs.nprays > 1800 && configs.nangles > 1800) n_total_process = 8;

        if (configs.nprays >= 4000 || configs.nangles >= 4000) n_total_process = 16;
        if (configs.nprays >= 8000 || configs.nangles >= 8000) n_total_process = 32;

        return n_total_process;
    }
}