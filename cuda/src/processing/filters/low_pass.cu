// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "common/opt.hpp"
#include "processing/filters.hpp"
#include "geometries/parallel/fbp.hpp"


extern "C"{
    void getFilterLowPass(CFG configs, GPU gpus, 
    float *tomogram, 
    dim3 tomo_size, dim3 tomo_pad)
    {
        int filter_type      = configs.reconstruction_filter_type;
        float paganin_reg    = configs.reconstruction_paganin;
        float regularization = configs.reconstruction_reg;
        float axis_offset    = 0.0;
        float pixel_x        = configs.geometry.obj_pixel_x;
        float pixel_y        = configs.geometry.obj_pixel_y;

        int nangles          = configs.tomo.size.y;

        Filter filter(filter_type, paganin_reg, regularization, axis_offset, pixel_x);

        if (filter.type != Filter::EType::none)
            filterFBP(gpus, filter, tomogram, tomo_size, tomo_pad, configs.tomo.pad);

        HANDLE_ERROR(cudaDeviceSynchronize());

    }
}

extern "C"{   
    void getFilterLowPassGPU(CFG configs, GPU gpus, 
    float *tomogram, int sizez, int ngpu)
    {
        HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Projection data sizes */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;
        int nrayspad = configs.tomo.padsize.x;

        int i;

        int blocksize = configs.blocksize;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, configs.total_required_mem_per_slice_bytes, true, A100_MEM);
            blocksize          = min(sizez, blocksize_aux);
        }

        int ind_block = (int)ceil( (float) sizez / blocksize );

        float *dtomo   = opt::allocGPU<float>((size_t)     nrays *    nangles * blocksize);

        /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0;

        for (i = 0; i < ind_block; i++){

			subblock       = min(sizez - ptr, blocksize);

			ptr_block_tomo = (size_t)nrays * nangles * ptr;

			/* Update pointer */
			ptr = ptr + subblock;
			
            opt::CPUToGPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)nrays * nangles * subblock);

            getFilterLowPass( configs, gpus, dtomo,  
                            dim3(nrays     ,    nangles, subblock),  /* Tomogram size */
                            dim3(nrayspad  ,    nangles, subblock)); /* Tomogram padded size */

            opt::GPUToCPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)nrays * nangles * subblock);

        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaFree(dtomo));
    }

    void getFilterLowPassMultiGPU(int* gpus, int ngpus, 
    float* tomogram, float *paramf, int *parami)
    {
        int i, Maxgpudev;

		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		CFG configs; GPU gpu_parameters;

        setFBPParameters(&configs, paramf, parami);
        // printFBPParameters(&configs);

        setGPUParameters(&gpu_parameters, configs.obj.size, ngpus, gpus);

        /* Projection data sizes */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;
        int nslices  = configs.tomo.size.z;

		int subvolume = (nslices + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			getFilterLowPassGPU(configs, gpu_parameters, tomogram, nslices, gpus[0]);

		}else{
		/* Launch async Threads for each device.
			Each device solves a block of 'nrays * nangles' size.
		*/
			// See future c++ async launch
			std::vector<std::future<void>> threads = {};
            threads.reserve(ngpus);

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(nslices - ptr, subvolume);

				threads.push_back( std::async( std::launch::async, 
                    getFilterLowPassGPU, 
                    configs, gpu_parameters, 
                    tomogram + (size_t)nrays * nangles * ptr, 
                    subblock,
                    gpus[i]));

                /* Update pointer */
				ptr = ptr + subblock;		

			}
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}
    }

}

