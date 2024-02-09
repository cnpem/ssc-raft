#include "common/configs.hpp"

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
        /* Processes to parallelize the data z-axis by independent blocks */
        /* Declare variables */
        long long int  n_obj, n_tomo, ind_obj, ind_tomo;
        int ind, ind_max, block;  

        /* Set indexes */
        block    = (int) ( configs.tomo.size.z / n_total_processes ); 

        ind      = index * block;

        ind_max  = (int) std::min( ( index + 1 ) * block, (int)configs.tomo.size.z);

        /* Indexes for Reconstruction division - same as Tomogram division */
        n_obj    = (long long int) ( ind_max - ind ) * configs.obj.size.x * configs.obj.size.y;
        ind_obj  = (long long int)             ind   * configs.obj.size.x * configs.obj.size.y;

        /* Indexes for Tomogram division - same as Reconstruction division */
        n_tomo   = (long long int) ( ind_max - ind ) * configs.tomo.size.x * configs.tomo.size.y;
        ind_tomo = (long long int)             ind   * configs.tomo.size.x * configs.tomo.size.y;

        /* Set process struct */
        (*process).index          = index;
        (*process).index_gpu      = (int)gpus.gpus[index % gpus.ngpus]; 
        (*process).batch_index    = (int)index % gpus.ngpus;
        (*process).tomobatch_size = (int)( ind_max - ind );
        (*process).objbatch_size  = (int)( ind_max - ind );

        /* Tomogram division */
        (*process).tomo_index_z   = ind;
        (*process).tomoptr_index  = ind_tomo;
        (*process).tomoptr_size   = n_tomo;

        /* Reconstruction division */
        (*process).objptr_size    = n_obj;
        (*process).objptr_index   = ind_obj;
        
    }
}


extern "C"{
    void setProcessConebeam(CFG configs, Process* process, GPU gpus, int index, int n_total_processes)
    {   
        /* Processes to parallelize the data z-axis by independent blocks */
        /* Declare variables */
        int block;  

        /* Reconstruction */
        long long int n_obj, ind_obj;
        int indz, indz_max; 
        float posz, posz_max;
        
        /* Tomogram and reconstruction filter (v of vertical) */
        long long int n_tomo, ind_tomo, n_filter, ind_filter;
        int ind, ind_max, indz_filter;
        float pos, pos_max, lenght;

        /* Reconstruction (or object) variables */ 
        block    = (int) ( ( configs.obj.zslice1 - configs.obj.zslice0 ) / n_total_processes ); 
        
        indz        = configs.obj.zslice0 + index * block;
        
        indz_max    = (int) std::min( configs.obj.zslice0 + ( index + 1 ) * block, configs.obj.zslice1 ); 

        n_obj     = (long long int) ( indz_max -                indz ) * configs.obj.size.x * configs.obj.size.y;
        ind_obj   = (long long int) ( indz     - configs.obj.zslice0 ) * configs.obj.size.x * configs.obj.size.y;

        posz           = - configs.obj.z +     indz * configs.obj.dz;
        posz_max       = - configs.obj.z + indz_max * configs.obj.dz;
                
        lenght      = sqrtf( configs.obj.x * configs.obj.x + configs.obj.y * configs.obj.y );
        
        /* Tomogram (or detector) and filter (with padding) variables */
        float z12x  = configs.geometry.z1x + configs.geometry.z2x;
        float z12y  = configs.geometry.z1y + configs.geometry.z2y;

        /* Tomogram */
        pos           = std::max(- configs.tomo.z, std::min( z12x * posz     / ( configs.geometry.z1x - lenght ), z12x *     posz / ( configs.geometry.z1x + lenght ) ) );
        pos_max       = std::min(+ configs.tomo.z, std::max( z12x * posz_max / ( configs.geometry.z1x + lenght ), z12x * posz_max / ( configs.geometry.z1x - lenght ) ) ); 

        ind        = std::max(                        0, (int) floor( ( pos     + configs.tomo.z ) / configs.tomo.dz ) );
        ind_max    = std::min( (int)configs.tomo.size.z, (int)  ceil( ( pos_max + configs.tomo.z ) / configs.tomo.dz ) );

        n_tomo      = (long long int) ( ind_max -                  ind ) * ( configs.tomo.size.x * configs.tomo.size.y );
        ind_tomo    = (long long int) ( ind     - configs.tomo.zslice0 ) * ( configs.tomo.size.x * configs.tomo.size.y );

        /* Set process struct */

        /* Reconstruction Filter */
        n_filter    = (long long int) ( ind_max -                 ind  ) * ( configs.tomo.padsize.x * configs.tomo.size.y );
        ind_filter  = (long long int) ( ind     - configs.tomo.zslice0 ) * ( configs.tomo.padsize.x * configs.tomo.size.y );
        indz_filter = (          int) (                       n_filter ) / ( configs.tomo.padsize.x * configs.tomo.size.y );

        /* Set process struct */
        (*process).index           = index;
        (*process).index_gpu       = (int)gpus.gpus[index % gpus.ngpus]; 
        (*process).batch_index     = (int)index % gpus.ngpus;
        (*process).tomobatch_size  = (int)(  ind_max -  ind );
        (*process).objbatch_size   = (int)( indz_max - indz );

        /* Tomogram division */
        (*process).tomo_index_z    = ind;
        (*process).tomoptr_index   = ind_tomo;
        (*process).tomoptr_size    = n_tomo;

        /* Reconstruction Filter division */
        (*process).filterptr_size  = n_filter;
        (*process).filterptr_index = ind_filter;
        (*process).filter_index_z  = indz_filter; 

        /* Reconstruction division */
        (*process).objptr_size     = n_obj;
        (*process).objptr_index    = ind_obj;
        (*process).obj_posz        = posz;
        (*process).tomo_posz       = - configs.tomo.z + ind * configs.tomo.dz;
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