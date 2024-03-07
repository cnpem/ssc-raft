// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

// #include <logger.hpp>
#include "common/opt.hpp"
#include "processing/filters.hpp"
#include "geometries/parallel/fbp.hpp"

extern "C"{
    __global__ void BackProjection_SS(float *object, float *tomogram, 
    float *angles, float *sine, float *cosine,
    dim3 obj_size, dim3 tomo_size)
    {
        int i, j, k;
        float x, y;
        int t_index, angle_index;
        float t, sum;  
        // float cosk, sink;
  
        float xmin = -1.0;
        float ymin = -1.0;
        float dx = 2.0 / (obj_size.x - 1);
        float dy = 2.0 / (obj_size.y - 1);

        int nrays   = tomo_size.x;
        int nangles = tomo_size.y;

        float dt    = 2.0 / (nrays - 1);
        float tmin = -1.0;

        float dangle; // = angles[1] - angles[0];
        
        i = (blockDim.x * blockIdx.x + threadIdx.x);
        j = (blockDim.y * blockIdx.y + threadIdx.y);
        k = (blockDim.z * blockIdx.z + threadIdx.z);
    
        if ( ( i < obj_size.x ) && ( j < obj_size.y ) && ( k < obj_size.z ) ){
        
            sum = 0;
            
            x = xmin + i * dx;
            y = ymin + j * dy;
            
            for(angle_index = 0; angle_index < nangles; angle_index++){

                // __sincosf(angles[angle_index], &sink, &cosk);

                /* Compute angle step size (dangle)*/
                if ( angle_index == (nangles - 1) )

                    dangle = angles[angle_index    ] - angles[angle_index - 1];

                else

                    dangle = angles[angle_index + 1] - angles[angle_index    ];
                
                t = x * cosine[angle_index] + y * sine[angle_index];
                
                // t = x * cosk + y * sink;
                
                t_index = (int) ( ( t - tmin ) / dt);	     

                if ( ( t_index > -1 ) && ( t_index < nrays) )
                    sum += tomogram[ k * nrays * nangles  + angle_index * nrays + t_index];
                
            }
        
            object[k * obj_size.y * obj_size.x + j * obj_size.x + i]  = ( sum * dangle ); 
        }
    }

}

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
        
        // cufftComplex *filter_kernel = opt::allocGPU<cufftComplex>(tomo_pad.x);

        float *sintable = opt::allocGPU<float>(nangles);
        float *costable = opt::allocGPU<float>(nangles);

        int gridBlock = (int)ceil( nangles / TPBY ) + 1;
        setSinCosTable<<<gridBlock,TPBY>>>(sintable, costable, angles, nangles);

        if (filter.type != Filter::EType::none)
            filterFBP(gpus, filter, tomogram, tomo_size, tomo_pad, configs.tomo.pad);

        /* Old version - Gio */
        // if (filter.type != Filter::EType::none)
        //     SinoFilter(tomogram, 
        //         (size_t)tomo_size.x, (size_t)tomo_size.y, (size_t)tomo_size.z, 
        //         0, true, filter, false, sintable);

        BackProjection_SS<<<gpus.Grd,gpus.BT>>>(obj, tomogram, angles,
                                                sintable, costable, 
                                                obj_size, tomo_size);

        HANDLE_ERROR(cudaDeviceSynchronize());
        
        HANDLE_ERROR(cudaFree(sintable));
        HANDLE_ERROR(cudaFree(costable));
        // HANDLE_ERROR(cudaFree(filter_kernel));
    }
}

extern "C"{   

    void getFBPGPU(CFG configs, GPU gpus, 
    float *obj, float *tomogram, float *angles, 
    int sizez, int ngpu)
    {
        HANDLE_ERROR(cudaSetDevice(ngpu));

        // if (ngpu == 0)
        //     ssc_event_start("getFBPGPU", {
        //             ssc_param_int("configs.tomo.size.x", configs.tomo.size.x),
        //             ssc_param_int("configs.tomo.size.y", configs.tomo.size.y),
        //             ssc_param_int("configs.tomo.size.z", configs.tomo.size.z),
        //             ssc_param_int("configs.obj.size.x", configs.obj.size.x),
        //             ssc_param_int("configs.obj.size.y", configs.obj.size.y),
        //             ssc_param_int("configs.tomo.pad.x", configs.tomo.pad.x),
        //             ssc_param_int("configs.tomo.pad.y", configs.tomo.pad.y),
        //             ssc_param_int("configs.tomo.pad.z", configs.tomo.pad.z),
        //             ssc_param_int("configs.tomo.padsize.x", configs.tomo.padsize.x),
        //             ssc_param_int("configs.tomo.padsize.y", configs.tomo.padsize.y),
        //             ssc_param_int("configs.tomo.padsize.z", configs.tomo.padsize.z),
        //             ssc_param_int("sizez (block)", sizez),
        //             ssc_param_int("ngpu", ngpu)
        //     });

        /* Projection data sizes */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;
        int nrayspad = configs.tomo.padsize.x;

        /* Reconstruction sizes */
        int sizeImagex = configs.obj.size.x;
        int sizeImagey = configs.obj.size.y;

        int i; 
		int blocksize = min(sizez,32);
        int ind_block = (int)ceil( (float) sizez / blocksize );

        float *dtomo   = opt::allocGPU<float>((size_t)     nrays *    nangles * blocksize);
        float *dobj    = opt::allocGPU<float>((size_t)sizeImagex * sizeImagey * blocksize);
        float *dangles = opt::allocGPU<float>( nangles );

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

            getFBP( configs, gpus, dobj, dtomo, dangles, 
                    dim3(nrays     ,    nangles, subblock),  /* Tomogram size */
                    dim3(nrayspad  ,    nangles, subblock),  /* Tomogram padded size */
                    dim3(sizeImagex, sizeImagey, subblock)); /* Object (reconstruction) size */

            opt::GPUToCPU<float>(obj + ptr_block_obj, dobj, 
                                (size_t)sizeImagex * sizeImagey * subblock);

        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaFree(dangles));
        HANDLE_ERROR(cudaFree(dtomo));
        HANDLE_ERROR(cudaFree(dobj));

        // if (ngpu == 0)
        //     ssc_event_stop(); /* getFBPGPU */
    }

    void getFBPMultiGPU(int* gpus, int ngpus, 
    float* obj, float* tomogram, float* angles, 
    float *paramf, int *parami)
    {
        // ssc_event_start("getFBPMultiGPU", {ssc_param_int("ngpus", ngpus)});

        int i, Maxgpudev;

		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		CFG configs; GPU gpu_parameters;

        setFBPParameters(&configs, paramf, parami);
        // printFBPParameters(&configs);

        setGPUParameters(&gpu_parameters, configs.tomo.padsize, ngpus, gpus);

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
        // ssc_event_stop(); /* getFBPMultiGPU */
    }

}

