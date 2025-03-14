// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "common/opt.hpp"
#include "processing/filters.hpp"
#include "geometries/parallel/fbp.hpp"

extern "C"{
    __global__ void BackProjection_SS(float *object, float *tomogram, 
    float *angles, float *sine, float *cosine, 
    float pixel_size_x, float pixel_size_y,
    dim3 obj_size, dim3 tomo_size)
    {
        int i, j, k, t_index, angle_index;
        float x, y, scale, t, sum;

        int nrays   = tomo_size.x;
        int nangles = tomo_size.y;

        /* Correction scale (angular correction) for 
            cases where there are more than 180 degrees.
            Specially for testes with Mogno conebeam data 
            that is acquired in 360 degrees rotation.
        */
        if ( angles[nangles - 1] > float(M_PI) ){
            scale = float(M_PI) / angles[nangles - 1];
        }else{  
            scale = 1.0f;
        }

        // float xmin = -1.0;
        // float ymin = -1.0;
        // float dx   = 2.0 / (obj_size.x - 1);
        // float dy   = 2.0 / (obj_size.y - 1);

        // float tmin = -1.0;
        // float dt   = 2.0 / (nrays - 1);

        float xmin = - pixel_size_x * obj_size.x / 2.0f;
        float ymin = - pixel_size_y * obj_size.y / 2.0f;
        float dx   =   pixel_size_x;
        float dy   =   pixel_size_y;

        float tmin = - pixel_size_x * nrays / 2.0f;
        float dt   = pixel_size_x;
        
        float dangle; // = angles[1] - angles[0];
        
        i = (blockDim.x * blockIdx.x + threadIdx.x);
        j = (blockDim.y * blockIdx.y + threadIdx.y);
        k = (blockDim.z * blockIdx.z + threadIdx.z);
    
        if ( ( i < obj_size.x ) && ( j < obj_size.y ) && ( k < obj_size.z ) ){
            sum = 0;
            
            x = xmin + i * dx;
            y = ymin + j * dy;
            
            for(angle_index = 0; angle_index < nangles; angle_index++){

                /* Compute angle step size (dangle)*/
                if ( angle_index == (nangles - 1) )
                    dangle = angles[angle_index] - angles[angle_index - 1];
                else
                    dangle = angles[angle_index + 1] - angles[angle_index];
                
                /* Compute t variable */
                t = x * cosine[angle_index] - y * sine[angle_index]; 
                                
                t_index = (int) ( ( t - tmin ) / dt);	     

                if ( ( t_index > -1 ) && ( t_index < nrays) )
                    sum += tomogram[ k * nrays * nangles  + angle_index * nrays + t_index] * dangle;
            }
            object[k * obj_size.y * obj_size.x + j * obj_size.x + i]  = sum * scale;
        }
    }

}

extern "C"{
    void getFBP(CFG configs, GPU gpus, 
    float *obj, float *tomogram, float *angles, 
    dim3 tomo_size, dim3 obj_size)
    {
        int filter_type      = configs.reconstruction_filter_type;
        float paganin_reg    = configs.reconstruction_paganin;
        float regularization = configs.reconstruction_reg;
        float axis_offset    = configs.rotation_axis_offset;
        float pixel_x        = configs.geometry.obj_pixel_x;
        float pixel_y        = configs.geometry.obj_pixel_y;
        int nangles          = tomo_size.y;

        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock( (int)ceil( obj_size.x / TPBX ) + 1,
                        (int)ceil( obj_size.y / TPBY ) + 1,
                        (int)ceil( obj_size.z / TPBZ ) + 1);

        /* Filter and Paganin by slices (filter) */
        Filter filter(filter_type, paganin_reg, regularization, axis_offset, pixel_x);

        if (filter.type != Filter::EType::none)
            filterFBP(gpus, filter, tomogram, tomo_size);

        /* Sin and Cos tables for backprojection */
        float *sintable = opt::allocGPU<float>(nangles);
        float *costable = opt::allocGPU<float>(nangles);

        int grid = (int)ceil( nangles / TPBY ) + 1;
        setSinCosTable<<<grid,TPBY>>>(sintable, costable, angles, nangles);

        /* Backprojection */
        BackProjection_SS<<<gridBlock,threadsPerBlock>>>(obj, tomogram, angles,
                                                        sintable, costable, 
                                                        pixel_x, pixel_y,
                                                        obj_size, tomo_size);

        HANDLE_ERROR(cudaDeviceSynchronize());
        
        HANDLE_ERROR(cudaFree(sintable));
        HANDLE_ERROR(cudaFree(costable));    
    }
}

extern "C"{   

    void getFBPGPU(CFG configs, GPU gpus, 
    float *obj, float *tomogram, float *angles, 
    int sizez, int ngpu)
    {
        HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Projection data sizes */
        /* Projection size */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;

        /* Projection padded size */
        int nrayspad = configs.tomo.padsize.x;

        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock( (int)ceil( configs.tomo.padsize.x / TPBX ) + 1,
                            (int)ceil( configs.tomo.padsize.y / TPBY ) + 1,
                            (int)ceil( configs.tomo.padsize.z / TPBZ ) + 1);

        /* Reconstruction sizes */
        /* Reconstruction size */
        int sizeImagex = configs.obj.size.x;
        int sizeImagey = configs.obj.size.y;

        /* Reconstruction padded size */
        int padImagex  = configs.obj.padsize.x;
        int padImagey  = configs.obj.padsize.y;

        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 ObjthreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 ObjgridBlock(  (int)ceil( configs.obj.padsize.x / TPBX ) + 1,
                            (int)ceil( configs.obj.padsize.y / TPBY ) + 1,
                            (int)ceil( configs.obj.padsize.z / TPBZ ) + 1);

        int i;

        int blocksize = configs.blocksize;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, configs.total_required_mem_per_slice_bytes, true, A100_MEM);
            blocksize          = min(sizez, blocksize_aux);
        }

        int ind_block = (int)ceil( (float) sizez / blocksize );

        float *dtomo   = opt::allocGPU<float>((size_t)     nrays *    nangles * blocksize);
        float *dobj    = opt::allocGPU<float>((size_t)sizeImagex * sizeImagey * blocksize);
        float *dangles = opt::allocGPU<float>( nangles );

        /* Padding */
        float *dtomoPadded = opt::allocGPU<float>((size_t) nrayspad *   nangles * blocksize);
        float *dobjPadded  = opt::allocGPU<float>((size_t)padImagex * padImagey * blocksize);

        opt::CPUToGPU<float>(angles, dangles, nangles);

        /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0, ptr_block_obj = 0;

        // printf("Size image %d, %d \n", sizeImagex, sizeImagey);
        // printf("Size image %d, %d, %d \n", configs.tomo.size.z, configs.tomo.size.y,configs.tomo.size.x);
        // printf("Size image %d, %d, %d \n", configs.tomo.padsize.z, configs.tomo.padsize.y,configs.tomo.padsize.x);

        // fflush(stdout);

        for (i = 0; i < ind_block; i++){

			subblock       = min(sizez - ptr, blocksize);

			ptr_block_tomo = (size_t)     nrays *    nangles * ptr;
            ptr_block_obj  = (size_t)sizeImagex * sizeImagey * ptr;
            // ptr_block_obj  = (size_t)nrayspad * nangles * ptr;

			/* Update pointer */
			ptr = ptr + subblock;
			
            opt::CPUToGPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)nrays * nangles * subblock);

            /* Padding the tomogram data */
            opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(dtomo, dtomoPadded, 
                                                                configs.tomo.size, 
                                                                configs.tomo.pad, 
                                                                0.0f);

            getFBP( configs, gpus, dobjPadded, dtomoPadded, dangles, 
                    dim3(nrayspad ,   nangles, subblock),  /* Tomogram padded size */
                    dim3(padImagex, padImagey, subblock)); /* Object (reconstruction) padded size */

            /* Remove padd from the object (reconstruction) */
            opt::remove_paddR2R<<<ObjgridBlock,ObjthreadsPerBlock>>>(dobjPadded, dobj, 
                                                                    configs.obj.size, 
                                                                    configs.obj.pad);

            opt::GPUToCPU<float>(obj + ptr_block_obj, dobj, 
                                (size_t)sizeImagex * sizeImagey * subblock);

            // opt::GPUToCPU<float>(obj + ptr_block_obj, dtomoPadded, 
            //                     (size_t)nrayspad * nangles * subblock);

        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaFree(dangles));
        HANDLE_ERROR(cudaFree(dtomo));
        HANDLE_ERROR(cudaFree(dobj));
        HANDLE_ERROR(cudaFree(dtomoPadded));
        HANDLE_ERROR(cudaFree(dobjPadded));
    }

    void getFBPMultiGPU(int* gpus, int ngpus, 
    float* obj, float* tomogram, float* angles, 
    float *paramf, int *parami)
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

        setGPUParameters(&gpu_parameters, configs.obj.padsize, ngpus, gpus);

        /* Projection data sizes */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;
        int nslices  = configs.tomo.size.z;

        /* Reconstruction sizes */
        int sizeImagex = configs.obj.size.x;
        int sizeImagey = configs.obj.size.y;

		int subvolume = (nslices + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			getFBPGPU(configs, gpu_parameters, obj, tomogram, angles, nslices, gpus[0]);

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
                    getFBPGPU, 
                    configs, gpu_parameters, 
                    obj      + (size_t)sizeImagex * sizeImagey * ptr,
                    tomogram + (size_t)     nrays *    nangles * ptr, 
                    angles, 
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

