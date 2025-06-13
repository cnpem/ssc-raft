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
                    dangle = abs(angles[angle_index] - angles[angle_index - 1]);
                else
                    dangle = abs(angles[angle_index + 1] - angles[angle_index]);
                
                /* Compute t variable */
                t = x * cosine[angle_index] - y * sine[angle_index]; 
                                
                t_index = (int) ( ( t - tmin ) / dt);	     

                if ( ( t_index > -1 ) && ( t_index < nrays) )
                    sum += tomogram[ k * nrays * nangles  + angle_index * nrays + t_index] * dangle;
            }
            object[k * obj_size.y * obj_size.x + j * obj_size.x + i]  = sum;
        }
    }

}

extern "C"{
    void getFBP(REC ReconParam, 
    float *obj, float *tomogram, float *angles, 
    dim3 tomo_size, dim3 obj_size, 
    float pixel_x, float pixel_y)
    {
        int filter_type   = ReconParam.filter;
        float paganin_reg = ReconParam.paganin_slices;
        float filter_reg  = ReconParam.filter_reg;
        float axis_offset = ReconParam.rotation_axis_offset;
        int nangles       = tomo_size.y;

        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock( (int)ceil( obj_size.x / TPBX ) + 1,
                        (int)ceil( obj_size.y / TPBY ) + 1,
                        (int)ceil( obj_size.z / TPBZ ) + 1);

        /* Filter and Paganin by slices (filter) */
        Filter filter(filter_type, paganin_reg, filter_reg, axis_offset, pixel_x);

        if (filter.type != Filter::EType::none)
            filterFBP(filter, tomogram, tomo_size);

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

    void getFBPGPU(DIM tomo, DIM obj, GEO geometry, REC ReconParam, 
    float *object, float *tomogram, float *angles, 
    int sizez, int ngpu)
    {
        HANDLE_ERROR(cudaSetDevice(ngpu));

        int i;

        int blocksize = tomo.blocksize;

        /* Compute total memory used of FBP method on a singles slice */
        size_t total_required_mem_per_slice_bytes = (
            calcSliceMemoryBytes(tomo)           + // Tomo slice
            calcSliceMemoryBytes(obj)            + // Reconstructed object slice
            calcPaddedSliceMemoryBytes(obj)      + // Reconstructed padded object slice
            2 * calcPaddedSliceMemoryBytes(tomo) + // Tomo padded slice
            tomo.size.y * sizeof(float)            // angles
            ); 

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, 
                                                    total_required_mem_per_slice_bytes, 
                                                    true, 
                                                    BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(sizez, blocksize_aux);
        }

        int ind_block = (int)ceil( (float) sizez / blocksize );

        /* Projection data sizes */
        /* Projection size */
        int nrays    = tomo.size.x;
        int nangles  = tomo.size.y;

        /* Projection padded size */
        int nrayspad = nrays * (1 + tomo.pad.x);

        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock( (int)ceil(  nrayspad / TPBX ) + 1,
                            (int)ceil(   nangles / TPBY ) + 1,
                            (int)ceil( blocksize / TPBZ ) + 1);

        /* Reconstruction sizes */
        /* Reconstruction size */
        int sizeImagex = obj.size.x;
        int sizeImagey = obj.size.y;

        /* Reconstruction padded size */
        int padImagex  = sizeImagex * (1 + obj.pad.x);
        int padImagey  = sizeImagey * (1 + obj.pad.y);

        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 ObjthreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 ObjgridBlock(  (int)ceil( padImagex / TPBX ) + 1,
                            (int)ceil( padImagey / TPBY ) + 1,
                            (int)ceil( blocksize / TPBZ ) + 1);

        float *dtomo   = opt::allocGPU<float>((size_t)     nrays *    nangles * blocksize);
        float *dobj    = opt::allocGPU<float>((size_t)sizeImagex * sizeImagey * blocksize);
        float *dangles = opt::allocGPU<float>( nangles );

        /* Padding */
        float *dtomoPadded = opt::allocGPU<float>((size_t) nrayspad *   nangles * blocksize);
        float *dobjPadded  = opt::allocGPU<float>((size_t)padImagex * padImagey * blocksize);

        opt::CPUToGPU<float>(angles, dangles, nangles);

        /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0, ptr_block_obj = 0;

        for (i = 0; i < ind_block; i++){

			subblock       = min(sizez - ptr, blocksize);

			ptr_block_tomo = (size_t)     nrays *    nangles * ptr;
            ptr_block_obj  = (size_t)sizeImagex * sizeImagey * ptr;

			/* Update pointer */
			ptr = ptr + subblock;
			
            opt::CPUToGPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)nrays * nangles * subblock);

            /* Padding the tomogram data */
            TomogridBlock.z = (int)ceil( subblock / TPBZ ) + 1;
            opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(dtomo, dtomoPadded, 
                                                                dim3(nrays, nangles, subblock), 
                                                                tomo.pad);

            getFBP( ReconParam, dobjPadded, dtomoPadded, dangles, 
                    dim3(nrayspad ,   nangles, subblock),  /* Tomogram padded size */
                    dim3(padImagex, padImagey, subblock)  /* Object (reconstruction) padded size */
                    geometry.obj_pixel_x, geometry.obj_pixel_x); 

            /* Remove padd from the object (reconstruction) */
            ObjgridBlock.z = TomogridBlock.z;
            opt::remove_paddR2R<<<ObjgridBlock,ObjthreadsPerBlock>>>(dobjPadded, dobj, 
                                                                    dim3(sizeImagex, sizeImagey, subblock), 
                                                                    obj.pad);

            opt::GPUToCPU<float>(object + ptr_block_obj, dobj, 
                                (size_t)sizeImagex * sizeImagey * subblock);

        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaFree(dangles));
        HANDLE_ERROR(cudaFree(dtomo));
        HANDLE_ERROR(cudaFree(dobj));
        HANDLE_ERROR(cudaFree(dtomoPadded));
        HANDLE_ERROR(cudaFree(dobjPadded));
    }

    void getFBPMultiGPU(DIM tomo, DIM obj, GEO geometry, REC ReconParam,
    int* gpus, int ngpus, float* object, float* tomogram, float* angles)
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

        /* Reconstruction sizes */
        int sizeImagex = obj.size.x;
        int sizeImagey = obj.size.y;

		int subvolume = (nslices + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			getFBPGPU(tomo, obj, geometry, ReconParam, object, tomogram, angles, nslices, gpus[0]);

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
                    tomo, obj, geometry, ReconParam, 
                    object   + (size_t)sizeImagex * sizeImagey * ptr,
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

