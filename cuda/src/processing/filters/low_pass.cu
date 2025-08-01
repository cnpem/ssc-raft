// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "common/opt.hpp"
#include "processing/filters.hpp"
#include "geometries/parallel/fbp.hpp"


extern "C"{
    void getFilterLowPass(GEO geometry, REC FilterParam, 
    float *tomogram, dim3 tomo_size)
    {
        int filter_type   = FilterParam.filter;
        float paganin_reg = FilterParam.paganin_slices;
        float filter_reg  = FilterParam.filter_reg;
        float axis_offset = 0.0;
        float pixel_x     = geometry.obj_pixel.x;

        Filter filter(filter_type, paganin_reg, filter_reg, axis_offset, pixel_x);

        cufftHandle mplan;
        cufftHandle mplanI;

        cufftPlan1d(&mplan , tomo_size.x, CUFFT_R2C, tomo_size.y);
        cufftPlan1d(&mplanI, tomo_size.x, CUFFT_C2R, tomo_size.y);

        if (filter.type != Filter::EType::none)
            filter_lowpass(mplan, mplanI, filter, tomogram, tomo_size);

        HANDLE_FFTERROR(cufftDestroy(mplan));
        HANDLE_FFTERROR(cufftDestroy(mplanI));
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

extern "C"{   
    void getFilterLowPassGPU(DIM tomo, GEO geometry, REC FilterParam, 
    float *tomogram, int sizez, int ngpu)
    {
        HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Projection data sizes */
        int nrays    = tomo.size.x;
        int nangles  = tomo.size.y;
        int nrayspad = PDIM(nrays,tomo.pad.x); // nrays * (1 + tomo.pad.x);

        int i;

        int blocksize = tomo.blocksize;

        /* Compute total memory used on a singles slice */
        size_t total_required_mem_per_slice_bytes = (
            calcSliceMemoryBytes(tomo)       + // Tomo slice
            calcPaddedSliceMemoryBytes(tomo)  // Tomo padded slice
            ); 

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize( sizez, 
                                                        2 * total_required_mem_per_slice_bytes, 
                                                        true, 
                                                        BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(sizez, blocksize_aux);
        }

        int ind_block = (int)ceil( (float) sizez / blocksize );

        float *dtomo    = opt::allocGPU<float>((size_t)   nrays * nangles * blocksize);
        float *dtomopad = opt::allocGPU<float>((size_t)nrayspad * nangles * blocksize);

        /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0;

        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock( (int)ceil(  nrayspad / TPBX ) + 1,
                            (int)ceil(   nangles / TPBY ) + 1,
                            (int)ceil( blocksize / TPBZ ) + 1);

        for (i = 0; i < ind_block; i++){

			subblock       = min(sizez - ptr, blocksize);

			ptr_block_tomo = (size_t)nrays * nangles * ptr;

			/* Update pointer */
			ptr = ptr + subblock;
			
            opt::CPUToGPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)nrays * nangles * subblock);
            
            /* Padding the tomogram data */
            TomogridBlock.z = (int)ceil( subblock / TPBZ ) + 1;
            opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(dtomo, dtomopad, tomo.padding_mode,
                                                                dim3(nrays, nangles, subblock), 
                                                                tomo.pad);

            getFilterLowPass( geometry, FilterParam, dtomo,  
                            dim3(nrayspad, nangles, subblock));  /* Tomogram padded size */

            /* Remove padd from the tomogram (reconstruction) */
            opt::remove_paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(dtomopad, dtomo, 
                                                                       dim3(nrays, nangles, subblock), 
                                                                       tomo.pad);

            opt::GPUToCPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)nrays * nangles * subblock);

        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaFree(dtomo));
    }

    void getFilterLowPassMultiGPU(DIM tomo, GEO geometry, REC FilterParam,
    int* gpus, int ngpus, float* tomogram)
    {
        int i, Maxgpudev;

		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

        /* Projection data sizes */
        int nrays    = tomo.size.x;
        int nangles  = tomo.size.y;
        int nslices  = tomo.size.z;

		int subvolume = (nslices + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			getFilterLowPassGPU(tomo, geometry, FilterParam, tomogram, nslices, gpus[0]);

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
                    tomo, 
                    geometry, 
                    FilterParam,
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

