// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "common/opt.hpp"
#include "processing/filters.hpp"
#include "geometries/parallel/fbp.hpp"

extern "C"{
    __global__ void FBPBackProjectionRT(float *object, float *tomogram, 
    float *angles, float *sine, float *cosine, 
    float pixel_size, dim3 obj_size, dim3 tomo_size)
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

        float xmin = - pixel_size * obj_size.x / 2.0f;
        float ymin = - pixel_size * obj_size.y / 2.0f;
        float dx   =   pixel_size;
        float dy   =   pixel_size;

        float tmin = - pixel_size * nrays / 2.0f;
        float dt   =   pixel_size;
        
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
                    dangle = fabs(angles[angle_index] - angles[angle_index - 1]);
                else
                    dangle = fabs(angles[angle_index + 1] - angles[angle_index]);
                
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
    void getFBP(REC ReconParam, DIM tomo, DIM obj, 
    float *object, float *tomogram, float *angles, 
    float *objPadded, float *tomoPadded,
    int blocksize, float pixel)
    {
        /* Projection data sizes */
        /* Projection size */
        int nrays    = tomo.size.x;
        int nangles  = tomo.size.y;

        /* Projection padded size */
        int nrayspad = PDIM(nrays,tomo.pad.x); 

        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock = opt::setGridBlock(dim3(nrayspad,nangles,blocksize), TomothreadsPerBlock);

        /* Reconstruction sizes */
        /* Reconstruction size */
        int sizeImagex = obj.size.x;
        int sizeImagey = obj.size.y;

        /* Reconstruction padded size */
        int padImagex  = PDIM(sizeImagex,obj.pad.x); 
        int padImagey  = PDIM(sizeImagey,obj.pad.y); 
    
        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 ObjthreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 ObjgridBlock = opt::setGridBlock(dim3(padImagex,padImagey,blocksize), ObjthreadsPerBlock);

        /* Reconstruction parameters */
        int filter_type    = ReconParam.filter;
        float paganin_reg  = ReconParam.paganin_slices;
        float filter_reg   = ReconParam.filter_reg;
        int filter_pad     = ReconParam.filter_pad;
        int filter_padMode = ReconParam.filter_padMode;
        float axis_offset  = ReconParam.rotation_axis_offset;

        /* Padding the tomogram data */
        opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(tomogram, tomoPadded, 
                                                            tomo.padding_mode,
                                                            dim3(nrays, nangles, blocksize), 
                                                            tomo.pad);

        /* Filter and Paganin by slices (filter) */
        Filter filter(filter_type, paganin_reg, filter_reg, axis_offset, pixel, filter_pad, filter_padMode);

        cufftHandle mplan;
        cufftHandle mplanI;
        int filterXpad = PDIM(nrayspad,filter.pad); 

        cufftPlan1d(&mplan , filterXpad, CUFFT_R2C, nangles);
        cufftPlan1d(&mplanI, filterXpad, CUFFT_C2R, nangles);

        if (filter.type != Filter::EType::none)
            filter_lowpass(mplan, mplanI, filter, tomoPadded, dim3(nrayspad, nangles, blocksize));

        /* Sin and Cos tables for backprojection */
        float *sintable = opt::allocGPU<float>(nangles);
        float *costable = opt::allocGPU<float>(nangles);

        int grid = (int)ceil( nangles / TPBY ) + 1;
        setSinCosTable<<<grid,TPBY>>>(sintable, costable, angles, nangles);


        /* Backprojection */
        FBPBackProjectionRT<<<ObjgridBlock,ObjthreadsPerBlock>>>(   objPadded, tomoPadded, angles,
                                                                    sintable, costable, pixel,
                                                                    dim3(padImagex, padImagey, blocksize), 
                                                                    dim3(nrayspad, nangles, blocksize));
        /* Remove padd from the object (reconstruction) */
        opt::remove_paddR2R<<<ObjgridBlock,ObjthreadsPerBlock>>>(objPadded, object, 
                                                                 dim3(sizeImagex, sizeImagey, blocksize), 
                                                                 obj.pad);

        HANDLE_ERROR(cudaDeviceSynchronize());
        HANDLE_ERROR(cudaFree(sintable));
        HANDLE_ERROR(cudaFree(costable));  
        HANDLE_FFTERROR(cufftDestroy(mplan));
        HANDLE_FFTERROR(cufftDestroy(mplanI));  
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

extern "C"{   

    void getFBPGPU(DIM tomo, DIM obj, GEO geometry, REC ReconParam, 
    float *object, float *tomogram, float *angles, 
    int sizez, int ngpu)
    {
        HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Compute total memory used of FBP method on a singles slice */
        size_t total_required_mem_per_slice_bytes = (
            calcSliceMemoryBytes(tomo)           + // Tomo slice
            calcSliceMemoryBytes(obj)            + // Reconstructed object slice
            calcPaddedSliceMemoryBytes(obj)      + // Reconstructed padded object slice
            2 * calcPaddedSliceMemoryBytes(tomo) + // Tomo padded slice
            tomo.size.y * sizeof(float)            // angles
            ); 
        
        int blocksize = getGPUBlocksize(tomo.blocksize, sizez, total_required_mem_per_slice_bytes, 128, true);
        int ind_block = getNumberOfBlocks(sizez, blocksize); 

        /* Projection data sizes */
        /* Projection size */
        int nrays    = tomo.size.x;
        int nangles  = tomo.size.y;
        int nTomo    = nrays * nangles;

        /* Projection padded size */
        int nrayspad = PDIM(nrays,tomo.pad.x); 
        int nTomopad = nrayspad * nangles;

        /* Reconstruction sizes */
        /* Reconstruction size */
        int sizeImagex = obj.size.x;
        int sizeImagey = obj.size.y;
        int nImage     = sizeImagex * sizeImagey;

        /* Reconstruction padded size */
        int padImagex  = PDIM(sizeImagex,obj.pad.x); 
        int padImagey  = PDIM(sizeImagey,obj.pad.y); 
        int npadImage  = padImagex * padImagey;

        float *dtomo       = opt::allocGPU<float>((size_t)    nTomo * blocksize);
        float *dobj        = opt::allocGPU<float>((size_t)   nImage * blocksize);   
        float *dtomoPadded = opt::allocGPU<float>((size_t) nTomopad * blocksize);
        float *dobjPadded  = opt::allocGPU<float>((size_t)npadImage * blocksize);

        float *dangles     = opt::allocGPU<float>( nangles );  
    
        opt::CPUToGPU<float>(angles, dangles, nangles);
        
        int ptr = 0, subblock;
        for (int i = 0; i < ind_block; i++){

            subblock = getSubblock(sizez - ptr, blocksize); 

            opt::CPUToGPU<float>(tomogram + (size_t)nTomo * ptr, dtomo, (size_t)nTomo * subblock);

            getFBP( ReconParam, tomo, obj, 
                    dobj, dtomo, dangles, dobjPadded, dtomoPadded,
                    subblock, geometry.obj_pixel.x); 

            opt::GPUToCPU<float>(object + (size_t)nImage * ptr, dobj, (size_t)nImage * subblock);

            /* Update pointer */
            ptr = ptr + subblock;
        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaFree(dangles    ));
        HANDLE_ERROR(cudaFree(dtomo      ));
        HANDLE_ERROR(cudaFree(dobj       ));
        HANDLE_ERROR(cudaFree(dtomoPadded));
        HANDLE_ERROR(cudaFree(dobjPadded ));
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
			//See future c++ async launch
			std::vector<std::future<void>> threads = {};
            threads.reserve(ngpus);

			for (i = 0; i < ngpus; i++){
				
				subblock = getSubblock(nslices - ptr, subvolume);

				threads.push_back(std::async(   std::launch::async, 
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
			for (i = 0; i < ngpus; i++) threads[i].get();
		}
    }

}

