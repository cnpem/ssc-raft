#include <unistd.h>
#include <cstddef>
#include <iostream>
#include "common/configs.hpp"
#include "common/types.hpp"

extern "C"{
    int compute_GPU_blocksize_block(int nslices, float total_required_mem_per_slice,
    bool using_fft, float gpu_memory)
    {
        const float empiric_const = using_fft? 2.0 : 1.0; // the GPU needs some free memory to perform the FFTs.
        const float epsilon = 4.0;       // how much free memory we want to leave, in GB.
    
        long blocksize;

        float total_mem_per_slice_GB = BYTES_TO_GB * total_required_mem_per_slice;

        blocksize = static_cast<long>( - epsilon + ( gpu_memory / ( total_mem_per_slice_GB * empiric_const ) ) );
        
        // std::cout << "\t  total required mem per slice: " << total_mem_per_slice_GB << " GB" << std::endl;
        // std::cout << "\t  gpu_memory: " << gpu_memory << " GB" << std::endl;
        // std::cout << "\t  number of slices: " << nslices << std::endl;

        if (nslices < blocksize) blocksize = nslices;

        // std::cout << "\t  Blocksize: " << blocksize << std::endl;

        return blocksize;
    }

    int compute_GPU_blocksize(int nslices, float total_required_mem_per_slice,
    bool using_fft, float gpu_memory)
    {
        const float empiric_const = using_fft? 4.0 : 1.0; // the GPU needs some free memory to perform the FFTs.
        const float epsilon = 4.0;       // how much free memory we want to leave, in GB.
        
        // the values permitted for blocksize are powers of two.
        long raw_blocksize; // biggest blocksize feasible, although not necessarily: 
            // 1) a power of two; and 
            // 2) not a divisor of nslices (i.e., nslices % raw_blocksize != 0).

        long blocksize_exp = 1; // to store which power of 2 will be used. 
        long blocksize;

        float total_mem_per_slice_GB = BYTES_TO_GB * total_required_mem_per_slice;

        raw_blocksize = static_cast<long>( - epsilon + ( gpu_memory / ( total_mem_per_slice_GB * empiric_const ) ) );
        
        std::cout << "\t  total_required_mem_per_slice GB: " << total_mem_per_slice_GB << std::endl;
        std::cout << "\t  gpu_memory: " << gpu_memory << std::endl;
        std::cout << "\t  Raw blocksize: " << raw_blocksize << std::endl;

        if (nslices < raw_blocksize) {
            blocksize = nslices;
        } else {
            while (raw_blocksize >> blocksize_exp) {
                blocksize_exp++;
            }
            blocksize_exp--;
            blocksize = 1 << blocksize_exp;
        }

        std::cout << "\t  Blocksize: " << blocksize << std::endl;

        return blocksize;
    }
}

extern "C"{

	Process *setProcesses(CFG configs, GPU gpus, int total_number_of_processes)
	{
        Process *process = (Process *) malloc(sizeof(Process) * total_number_of_processes);

        if (isParallelOrFanbeamGeometry(configs)) {
            for (int p = 0; p < total_number_of_processes; p++)
                setProcessParallel(configs, process, gpus, p, total_number_of_processes);
        } else { //cone beam geometry
            for (int p = 0; p < total_number_of_processes; p++)
                setProcessConebeam(configs, process, gpus, p, total_number_of_processes);
        }

        return process;
	}
}

extern "C"{
    void setProcessParallel(CFG configs, Process* process, GPU gpus, int index, int n_total_processes)
    {
        /* 20/03/2025 - NEEDS TO FIX THIS FUNCTION BLOCKSIZE - LOOK INTO FDK */
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
        process[index].index          = index;
        process[index].index_gpu      = (int)gpus.gpus[index % gpus.ngpus];
        process[index].batch_index    = (int)index % gpus.ngpus;
        process[index].tomobatch_size = (int)( ind_max - ind );
        process[index].objbatch_size  = (int)( ind_max - ind );

        /* Tomogram division */
        process[index].tomo_index_z   = ind;
        process[index].tomoptr_index  = ind_tomo;
        process[index].tomoptr_size   = n_tomo;

        /* Reconstruction division */
        process[index].objptr_size    = n_obj;
        process[index].objptr_index   = ind_obj;
    }
}


extern "C"{
    void setProcessConebeam(CFG configs, Process* process, GPU gpus, int index, int n_total_processes)
    {   
        /* 20/03/2025 - NEEDS TO FIX THIS FUNCTION BLOCKSIZE - LOOK INTO FDK */
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

        posz           = - configs.obj.Lz +     indz * configs.obj.dz;
        posz_max       = - configs.obj.Lz + indz_max * configs.obj.dz;
                
        lenght      = sqrtf( configs.obj.Lx * configs.obj.Lx + configs.obj.Ly * configs.obj.Ly );
        
        /* Tomogram (or detector) and filter (with padding) variables */
        float z12x  = configs.geometry.z1x + configs.geometry.z2x;
        // float z12y  = configs.geometry.z1y + configs.geometry.z2y;

        /* Tomogram */
        pos           = std::max(- configs.tomo.Lz, std::min( z12x * posz     / ( configs.geometry.z1x - lenght ), z12x *     posz / ( configs.geometry.z1x + lenght ) ) );
        pos_max       = std::min(+ configs.tomo.Lz, std::max( z12x * posz_max / ( configs.geometry.z1x + lenght ), z12x * posz_max / ( configs.geometry.z1x - lenght ) ) ); 

        ind        = std::max(                        0, (int) floor( ( pos     + configs.tomo.Lz ) / configs.tomo.dz ) );
        ind_max    = std::min( (int)configs.tomo.size.z, (int)  ceil( ( pos_max + configs.tomo.Lz ) / configs.tomo.dz ) );

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
        (*process).tomo_posz       = - configs.tomo.Lz + ind * configs.tomo.dz;
    }
}

extern "C"{
    int getTotalProcesses(CFG configs, float gpu_memory, int sizeZ, bool using_fft)
    {
        const float total_required_mem_per_slice_bytes = calcTotalRequiredMemoryBytes(configs);
        const int blocksizeMax = compute_GPU_blocksize(sizeZ,
                                                total_required_mem_per_slice_bytes,
                                                using_fft,
                                                gpu_memory);

        const int n_total_processes = (int)ceil( (float) sizeZ / blocksizeMax );

        ssc_assert(n_total_processes > 0, "Invalid number of total processes");

        return n_total_processes;
    }
}
