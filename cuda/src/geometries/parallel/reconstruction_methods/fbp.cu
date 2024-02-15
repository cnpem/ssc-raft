// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "processing/filters.hpp"
#include "geometries/parallel/fbp.hpp"

extern "C"{
    void getFBP(CFG configs, GPU gpus, 
    float *obj, float *tomogram, float *angles, 
    dim3 tomo_size, dim3 tomo_pad, dim3 obj_size)
    {
        int filter_type      = configs.reconstruction_filter_type;
        float paganin_reg    = configs.reconstruction_paganin_reg;
        float regularization = configs.reconstruction_reg;
        int axis_offset      = configs.rotation_axis_offset;

        int nangles          = configs.tomo.size.y;

        Filter filter(filter_type, regularization, paganin_reg, axis_offset);
        
        cufftComplex *filter_kernel = opt::allocGPU<cufftComplex>(tomo_pad.x);

        float *sintable = opt::allocGPU<float>(nangles);
        float *costable = opt::allocGPU<float>(nangles);

        setSinCosTable<<<gpus.Grd.y,gpus.BT.y>>>(sintable, costable, angles, nangles);

        filterFBP(gpus, filter, tomogram, filter_kernel, tomo_size, tomo_pad);

        BackProjection_RT<<<gpus.Grd,gpus.BT>>>(obj, tomogram, 
                                                sintable, costable, 
                                                obj_size, tomo_size);

        cudaDeviceSynchronize();
        
        cudaFree(sintable);
        cudaFree(costable);
        cudaFree(filter_kernel);
    }

}

extern "C"{   

    void getFBPGPU(CFG configs, GPU gpus, 
    float *obj, float *tomogram, float *angles, 
    int sizez, int ngpu)
    {
        cudaSetDevice(ngpu);

        int i; 
		int blocksize = min(sizez,32);
        int ind_block = (int)ceil( (float) sizez / blocksize );

        float *dtomo   = opt::allocGPU<float>((size_t)configs.tomo.size.x * configs.tomo.size.y * blocksize);
        float *dobj    = opt::allocGPU<float>((size_t) configs.obj.size.x *  configs.obj.size.y * blocksize);
        float *dangles = opt::allocGPU<float>( configs.tomo.size.y );

        opt::CPUToGPU<float>(angles, dangles, configs.tomo.size.y);

        /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0, ptr_block_obj = 0;

        for (i = 0; i < ind_block; i++){

			subblock        = min(sizez - ptr, blocksize);

			ptr_block_tomo = (size_t)configs.tomo.size.x * configs.tomo.size.y * ptr;
            ptr_block_obj  = (size_t) configs.obj.size.x *  configs.obj.size.y * ptr;

			/* Update pointer */
			ptr = ptr + subblock;
			
            opt::CPUToGPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)configs.tomo.size.x * configs.tomo.size.y * subblock);

            getFBP( configs, gpus, dobj, dtomo, dangles, 
                    dim3(configs.tomo.size.x   , configs.tomo.size.y, subblock), 
                    dim3(configs.tomo.padsize.x, configs.tomo.size.y, subblock),
                    dim3(configs.obj.size.x    , configs.obj.size.y , subblock));

            opt::GPUToCPU<float>(obj + ptr_block_obj, dobj, (size_t)configs.obj.size.x * configs.obj.size.y * subblock);

        }
        cudaDeviceSynchronize();

        cudaFree(dangles);
        cudaFree(dtomo);
        cudaFree(dobj);
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

        setGPUParameters(&gpu_parameters, configs.tomo.padsize, ngpus, gpus);

		int subvolume = (configs.tomo.size.z + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			getFBPGPU(configs, gpu_parameters, obj, tomogram, angles, subvolume, gpus[0]);

		}else{
		/* Launch async Threads for each device.
			Each device solves a block of 'nrays * nangles' size.
		*/
			// See future c++ async launch
			std::vector<std::future<void>> threads = {};

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(configs.tomo.size.z - ptr, subvolume);

				threads.push_back( std::async( std::launch::async, 
												getFBPGPU, 
												configs, gpu_parameters, 
                                                obj      + (size_t) configs.obj.size.x *  configs.obj.size.y * ptr,
												tomogram + (size_t)configs.tomo.size.x * configs.tomo.size.y * ptr, 
												angles, 
                                                subblock,
												gpus[i]
												));
                /* Update pointer */
				ptr = ptr + subblock;		

			}
		
			// Log("Synchronizing all threads...\n");
		
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}	

		cudaDeviceSynchronize();
    }

}

extern "C"{

    __global__ void BackProjection_RT(float* obj, const float *tomo, 
    const float* sintable, const float* costable,
    dim3 obj_size, dim3 tomo_size)
    {  
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = obj_size.x * j + i;
        size_t index = obj_size.y * k * obj_size.x + ind;

        float sum = 0, frac; 
        float x, y, t, norm;
        int T, angle;
        
        if ( (i >= obj_size.x) || (j >= obj_size.y) || (k >= obj_size.z) ) return;

        norm  = ( 0.5f * float(M_PI) ) / ( float(tomo_size.y) * float(tomo_size.x) ); 

        x     = - (float)obj_size.x/2.0f + i;
        y     = - (float)obj_size.y/2.0f + j;

        for(angle = 0; angle < (tomo_size.y); angle++){
        
            t = ( x * costable[angle] + y * sintable[angle] + tomo_size.x/2 );
            T = int(t);
        
            if ( ( T >= 0 ) && ( T < ( tomo_size.x - 1 ) ) ){
                frac = t-T;

                sum += tomo[tomo_size.y * tomo_size.x * k + angle * tomo_size.x + T] * (1.0f - frac) + tomo[angle * tomo_size.x + T + 1] * frac;
            }
        }        

        obj[index] = sum * norm;

    } 

    __global__ void BackProjection_SS(float *image, float *blocksino, float *angles,
    dim3 obj_size, dim3 tomo_size)
    {
        int i, j, k, T, z;
        float t, cs, x, y, cosk, sink;
        float xymin = -1.0;

        float dx = 2.0 / (obj_size.x - 1);
        float dy = 2.0 / (obj_size.y - 1);

        int nrays   = tomo_size.x;
        int nangles = tomo_size.y;
        int nslices = tomo_size.z;

        float dt    = 2.0 / (nrays - 1);
        float tmin = -1.0;

        float dth = angles[1] - angles[0];
        
        i = (blockDim.x * blockIdx.x + threadIdx.x);
        j = (blockDim.y * blockIdx.y + threadIdx.y);
        z = (blockDim.z * blockIdx.z + threadIdx.z);
    
        if ( (i<obj_size.x) && (j < obj_size.y) && (z<obj_size.z)  ){
        
            cs = 0;
            
            x = xymin + i * dx;
            y = xymin + j * dy;
            
            for(k=0; k < (nangles); k++){
                __sincosf(angles[k], &sink, &cosk);
                
                t = x * cosk + y * sink;
                
                T = (int) ((t - tmin)/dt);	     

                if ( (T > -1) && (T<nrays) ){
                    //cs = cs + blocksino[ T * nangles + k];
                    cs += blocksino[ z * nrays * nangles  + k * nrays + T];
                }
            }
        
            image[z * obj_size.y * obj_size.x + j * obj_size.x + i]  = (cs*dth); 
        }
    }

}
