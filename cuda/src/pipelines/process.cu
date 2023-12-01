#include "../../inc/sscraft.h"

extern "C"{
	Process *setProcesses(CFG configs, GPU gpus, int total_number_of_processes)
	{  
        int process_index;

        Process *process = (Process *)malloc(sizeof(Process) * total_number_of_processes);

        switch (configs.geometry.geometry){
            case 0:
                /* Parallel */
                for (process_index = 0; process_index < total_number_of_processes; process_index++)
                    setProcessParallel(configs, process, gpus, process_index, total_number_of_processes);
                break;
            case 1:
                /* Conebeam */
                for (process_index = 0; process_index < total_number_of_processes; process_index++)
                    setProcessConebeam(configs, process, gpus, process_index, total_number_of_processes);
                break;
            case 2:
                /* Fanbeam - the process division for Fanbeam geometry is the same as the Parallel one */
               for (process_index = 0; process_index < total_number_of_processes; process_index++)
                    setProcessParallel(configs, process, gpus, process_index, total_number_of_processes);
                break;
            default:
                printf("Nope.");
                break;
        }	

        return process; 
                
	}
}


extern "C"{
    void setProcessParallel(CFG configs, Process* process, GPU gpus, int index, int n_total_processes)
    {   
        /* Declare variables */
        long long int  n_recon, n_tomo, ind_recon, ind_tomo;
        int indz, indz_max, nz_block;  

        /* Set indexes */
        nz_block             = (int) ( configs.tomo.size.z / n_total_processes ); 

        indz                 = index * nz_block;

        indz_max             = (int) std::min( ( index + 1 ) * nz_block, (int)configs.tomo.size.z);

        /* Indexes for Reconstruction division - same as Tomogram division */
        n_recon              = (long long int) ( indz_max - indz ) * configs.recon.size.x * configs.recon.size.y;
        ind_recon            = (long long int)              indz   * configs.recon.size.x * configs.recon.size.y;

        /* Indexes for Tomogram division - same as Reconstruction division */
        n_tomo               = (long long int) ( indz_max - indz ) * configs.tomo.size.x * configs.tomo.size.y;
        ind_tomo             = (long long int)              indz   * configs.tomo.size.x * configs.tomo.size.y;

        /* Set process struct */
        (*process).index            = index;
        (*process).index_gpu        = (int)gpus.gpus[index % gpus.ngpus]; 
        (*process).batch_index      = (int)index % gpus.ngpus;
        (*process).batch_size_tomo  = (int)( indz_max - indz );
        (*process).batch_size_recon = (int)( indz_max - indz );

        /* Tomogram division */
        (*process).indv             = indz;
        (*process).ind_tomo         = ind_tomo;
        (*process).n_tomo           = n_tomo;

        /* Reconstruction division */
        (*process).n_recon          = n_recon;
        (*process).ind_recon        = ind_recon;
        
    }
}


extern "C"{
    void setProcessConebeam(CFG configs, Process* process, GPU gpus, int index, int n_total_processes)
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
        float v, v_max, lenght;

        /* Reconstruction (or object) variables */ 
        nz_block    = (int) ( ( configs.recon.end_slice - configs.recon.start_slice ) / n_total_processes ); 
        
        indz        = configs.recon.start_slice + index * nz_block;
        
        indz_max    = (int) std::min( configs.recon.start_slice + ( index + 1 ) * nz_block, configs.recon.end_slice ); 

        n_recon     = (long long int) ( indz_max -                      indz ) * configs.recon.size.x * configs.recon.size.y;
        ind_recon   = (long long int) ( indz     - configs.recon.start_slice ) * configs.recon.size.x * configs.recon.size.y;

        z           = - configs.recon.z +     indz * configs.recon.dz;
        z_max       = - configs.recon.z + indz_max * configs.recon.dz;
                
        lenght      = sqrtf( configs.recon.x * configs.recon.x + configs.recon.y * configs.recon.y );
        
        /* Tomogram (or detector) and filter (with padding) variables */
        float z12x  = configs.geometry.z1x + configs.geometry.z2x;
        float z12y  = configs.geometry.z1y + configs.geometry.z2y;

        /* Tomogram */
        v           = std::max(- configs.tomo.z, std::min( z12x * z     / ( configs.geometry.z1x - lenght ), z12x *     z / ( configs.geometry.z1x + lenght ) ) );
        v_max       = std::min(+ configs.tomo.z, std::max( z12x * z_max / ( configs.geometry.z1x + lenght ), z12x * z_max / ( configs.geometry.z1x - lenght ) ) ); 

        indv        = std::max(                        0, (int) floor( ( v     + configs.tomo.z ) / configs.tomo.dz ) );
        indv_max    = std::min( (int)configs.tomo.size.z, (int)  ceil( ( v_max + configs.tomo.z ) / configs.tomo.dz ) );

        n_tomo      = (long long int) ( indv_max -                     indv) * ( configs.tomo.size.x * configs.tomo.size.y );
        ind_tomo    = (long long int) ( indv     - configs.tomo.start_slice) * ( configs.tomo.size.x * configs.tomo.size.y );

        /* Set process struct */

        /* Reconstruction Filter */
        n_filter    = (long long int) ( indv_max -                     indv ) * ( configs.tomo.npad.x * configs.tomo.size.y );
        ind_filter  = (long long int) ( indv     - configs.tomo.start_slice ) * ( configs.tomo.npad.x * configs.tomo.size.y );
        indv_filter = (          int) (                            n_filter ) / ( configs.tomo.npad.x * configs.tomo.size.y );

        /* Set process struct */
        (*process).index            = index;
        (*process).index_gpu        = (int)gpus.gpus[index % gpus.ngpus]; 
        (*process).batch_index      = (int)index % gpus.ngpus;
        (*process).batch_size_tomo  = (int)( indv_max - indv );
        (*process).batch_size_recon = (int)( indz_max - indz );

        /* Tomogram division */
        (*process).indv             = indv;
        (*process).ind_tomo         = ind_tomo;
        (*process).n_tomo           = n_tomo;

        /* Reconstruction Filter division */
        (*process).n_filter         = n_filter;
        (*process).ind_filter       = ind_filter;
        (*process).indv_filter      = indv_filter; 

        /* Reconstruction division */
        (*process).n_recon          = n_recon;
        (*process).ind_recon        = ind_recon;
        (*process).z                = z;
        (*process).z_det            = - configs.tomo.z + indv * configs.tomo.dz;
        
    }
}

extern "C"{
    void setProcessFrames(CFG configs, Process* process, GPU gpus, int index, int n_total_processes)
    {   
                /* Declare variables */
        long long int  n_recon, n_tomo, ind_recon, ind_tomo;
        int indz, indz_max, nz_block;  

        /* Set indexes */
        nz_block             = (int) ( configs.tomo.size.z / n_total_processes ); 

        indz                 = index * nz_block;

        indz_max             = (int) std::min( ( index + 1 ) * nz_block, (int)configs.tomo.size.z);

        /* Indexes for Reconstruction division - same as Tomogram division */
        n_recon              = (long long int) ( indz_max - indz ) * configs.recon.size.x * configs.recon.size.y;
        ind_recon            = (long long int)              indz   * configs.recon.size.x * configs.recon.size.y;

        /* Indexes for Tomogram division - same as Reconstruction division */
        n_tomo               = (long long int) ( indz_max - indz ) * configs.tomo.size.x * configs.tomo.size.y;
        ind_tomo             = (long long int)              indz   * configs.tomo.size.x * configs.tomo.size.y;

        /* Set process struct */
        (*process).index            = index;
        (*process).index_gpu        = (int)gpus.gpus[index % gpus.ngpus]; 
        (*process).batch_index      = (int)index % gpus.ngpus;
        (*process).batch_size_tomo  = (int)( indz_max - indz );
        (*process).batch_size_recon = (int)( indz_max - indz );

        /* Tomogram division */
        (*process).indv             = indz;
        (*process).ind_tomo         = ind_tomo;
        (*process).n_tomo           = n_tomo;

        /* Reconstruction division */
        (*process).n_recon          = n_recon;
        (*process).ind_recon        = ind_recon;
        
    }
}


extern "C"{
    int getTotalProcesses(CFG configs, GPU gpus)
    {
        long double mem_gpu, mem_recon, mem_tomo;
        int n_total_process;
        long long int n_tomo, n_recon;
        
        n_tomo  = (long long int)(configs.tomo.npad.x)*(long long int)(configs.tomo.size.y)*(long long int)(configs.tomo.size.z);
        n_recon = (long long int)(configs.recon.size.y)*(long long int)(configs.recon.size.y)*(long long int)(configs.recon.size.y);

        mem_gpu = 40;
        mem_tomo = 32*n_tomo*1.16*(pow(10,-10));
        mem_recon = 32*n_recon*1.16*(pow(10,-10));

        n_total_process = (int) std::ceil((mem_tomo + mem_recon)/mem_gpu);

        // divisão de processos 
        if(n_total_process < gpus.ngpus) n_total_process = gpus.ngpus;

        if(configs.tomo.npad.x > 1800 && configs.tomo.nangles > 1800) n_total_process = 8;

        if (configs.tomo.npad.x >= 4000 || configs.tomo.nangles >= 4000) n_total_process = 16;
        if (configs.tomo.npad.x >= 8000 || configs.tomo.nangles >= 8000) n_total_process = 32;

        return n_total_process;
    }
}