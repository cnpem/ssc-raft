// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "../../../../inc/sscraft.h"


extern "C"{
    void getFBP(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, dim3 tomo_size, dim3 tomo_pad, dim3 recon_size)
    {
        int filter_type      = configs.reconstruction_filter_type;
        float paganin_reg    = configs.reconstruction_paganin_reg;
        float regularization = configs.reconstruction_reg;
        int axis_offset      = configs.rotation_axis_offset;

        Filter filter(filter_type, regularization, paganin_reg, axis_offset);
        
        cufftComplex *filter_kernel;
        HANDLE_ERROR(cudaMalloc((void **)&filter_kernel, sizeof(cufftComplex) * tomo_pad.x ));

        float *sintable, *costable;
        HANDLE_ERROR(cudaMalloc((void **)&sintable, configs.tomo.nangles * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&costable, configs.tomo.nangles * sizeof(float)));

        setSinCosTable<<<gpus.Grd.y,gpus.BT.y>>>(sintable, costable, angles, configs.tomo.nangles);

        filterFBP(gpus, filter, tomogram, filter_kernel, tomo_size, tomo_pad);

        BackProjection_RT<<<gpus.Grd,gpus.BT>>>(recon, tomogram, sintable, costable, recon_size, tomo_size);

        cudaDeviceSynchronize();
        
        cudaFree(sintable);
        cudaFree(costable);
        cudaFree(filter_kernel);
    }

    void getFBP_thresh(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, dim3 tomo_size, dim3 tomo_pad, dim3 recon_size)
    {
        int filter_type      = configs.reconstruction_filter_type;
        float paganin_reg    = configs.reconstruction_paganin_reg;
        float regularization = configs.reconstruction_reg;
        int axis_offset      = configs.rotation_axis_offset;

        Filter filter(filter_type, regularization, paganin_reg, axis_offset);
        EType datatype = EType((EType::TypeEnum)configs.datatype);

        cufftComplex *filter_kernel;
        HANDLE_ERROR(cudaMalloc((void **)&filter_kernel, sizeof(cufftComplex) * tomo_pad.x ));

        float *sintable, *costable;
        HANDLE_ERROR(cudaMalloc((void **)&sintable, configs.tomo.nangles * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&costable, configs.tomo.nangles * sizeof(float)));

        setSinCosTable<<<gpus.Grd.y,gpus.BT.y>>>(sintable, costable, angles, configs.tomo.nangles);

        filterFBP(gpus, filter, tomogram, filter_kernel, tomo_size, tomo_pad);

        BackProjection_RT_thresh<<<gpus.Grd,gpus.BT>>>(recon, tomogram, sintable, costable, recon_size, tomo_size, configs.threshold, datatype.type);

        cudaDeviceSynchronize();

        cudaFree(sintable);
        cudaFree(costable);
        cudaFree(filter_kernel);
    }

    __global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles)
    {
        size_t k = blockIdx.x*blockDim.x + threadIdx.x;

        if ( (k >= nangles) ) return;

        sintable[k] = __asinf(angles[k]);
        costable[k] = __acosf(angles[k]);
    }
}

extern "C"{   

    void getFBPGPU(CFG configs, GPU gpus, float *recon, float *tomogram, float *angles, 
    int sizez, int ngpu)
    {
        cudaSetDevice(ngpu);

        int i; 
		int blocksize = min(sizez,32);
        int ind_block = (int)ceil( (float) sizez / blocksize );

        float *dtomo, *dangles, *drecon;

        HANDLE_ERROR(cudaMalloc((void **)&dangles, sizeof(float) * configs.tomo.nangles )); 
		HANDLE_ERROR(cudaMalloc((void **)&dtomo  , sizeof(float) * (size_t)configs.tomo.size.x * configs.tomo.size.y * blocksize )); 
		HANDLE_ERROR(cudaMalloc((void **)&drecon , sizeof(float) * (size_t)configs.recon.size.x * configs.recon.size.y * blocksize )); 

        HANDLE_ERROR(cudaMemcpy(dangles, angles, configs.tomo.nangles * sizeof(float), cudaMemcpyHostToDevice));

        /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0, ptr_block_recon = 0;

        for (i = 0; i < ind_block; i++){

			subblock        = min(sizez - ptr, blocksize);

			ptr_block_tomo  = (size_t)configs.tomo.size.x  * configs.tomo.size.y  * ptr;
            ptr_block_recon = (size_t)configs.recon.size.x * configs.recon.size.y * ptr;

			/* Update pointer */
			ptr = ptr + subblock;
			
            HANDLE_ERROR(cudaMemcpy(dtomo, tomogram + ptr_block_tomo, (size_t)configs.tomo.size.x * configs.tomo.size.y * subblock * sizeof(float), cudaMemcpyHostToDevice));

            getFBP( configs, gpus, drecon, dtomo, dangles, 
                    dim3(configs.tomo.size.x , configs.tomo.size.y , subblock), 
                    dim3(configs.tomo.npad.x , configs.tomo.size.y , subblock),
                    dim3(configs.recon.size.x, configs.recon.size.y, subblock)
                  );
  
            HANDLE_ERROR(cudaMemcpy(recon + ptr_block_recon, drecon, (size_t)configs.recon.size.x * configs.recon.size.y * subblock * sizeof(float), cudaMemcpyDeviceToHost));                 

        }
        cudaDeviceSynchronize();
        cudaFree(dangles);
        cudaFree(dtomo);
        cudaFree(drecon);
    }

    void getFBPMultiGPU(float* recon, float* tomogram, float* angles, float *paramf, int *parami, int* gpus, int ngpus)
    {
        int i, Maxgpudev;
		
		/* Multiples devices */
		cudaGetDeviceCount(&Maxgpudev);

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		CFG configs; GPU gpu_parameters;

        setFBPParameters(&configs, paramf, parami);

        setGPUParameters(&gpu_parameters, configs.tomo.npad, ngpus, gpus);

		int subvolume = (tomo.size.z + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			getFBPGPU(configs, gpu_parameters, recon, tomogram, angles, gpus[0]);

		}else{
		/* Launch async Threads for each device.
			Each device solves a block of 'nrays * nangles' size.
		*/
			// See future c++ async launch
			std::vector<std::future<void>> threads = {};

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(tomo.size.z - ptr, subvolume);


				threads.push_back( std::async( std::launch::async, 
												getFBPGPU, 
												configs, gpu_parameters, 
                                                recon    + (size_t)configs.recon.size.x * configs.recon.size.y * ptr,
												tomogram + (size_t)configs.tomo.size.x  * configs.tomo.size.y  * ptr, 
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

    __global__ void BackProjection_RT(float* recon, const float *tomo, const float* sintable, const float* costable,
    dim3 recon_size, dim3 tomo_size)
    {  
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = recon_size.x * j + i;
        size_t index = recon_size.y * k * recon_size.x + ind;

        float sum = 0, frac; 
        float x, y, t, norm;
        int T, angle;
        
        if ( (i >= recon_size.x) || (j >= recon_size.y) || (k >= recon_size.z) ) return;

        norm  = ( 0.5f * float(M_PI) ) / ( float(tomo_size.y) * float(tomo_size.x) ); 

        x     = - (float)recon_size.x/2.0f + i;
        y     = - (float)recon_size.y/2.0f + j;

        for(angle = 0; angle < (tomo_size.y); angle++){
        
            t = ( x * costable[angle] + y * sintable[angle] + tomo_size.x/2 );
            T = int(t);
        
            if ( ( T >= 0 ) && ( T < ( tomo_size.x - 1 ) ) ){
                frac = t-T;

                sum += sino[tomo_size.y * tomo_size.x * k + angle * tomo_size.x + T] * (1.0f - frac) + sino[angle * tomo_size.x + T + 1] * frac;
            }
        }        

        recon[index] = sum * norm;

    } 

    __global__ void BackProjection_RT_thresh(float* recon, const float *tomo, const float* sintable, const float* costable,
    dim3 recon_size, dim3 tomo_size, float threshold, EType datatype)
    {  
        size_t i   = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j   = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k   = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind = j + k * recon_size.y;

        float sum = 0, frac; 
        float x, y, t, norm;
        int T, angle;
        
        if ( (i >= recon_size.x) || (j >= recon_size.y) || (k >= recon_size.z) ) return;

        norm  = ( 0.5f * float(M_PI) ) / ( float(tomo_size.y) * float(tomo_size.x) ); 

        x     = - (float)recon_size.x/2.0f + i;
        y     = - (float)recon_size.y/2.0f + j;

        for(angle = 0; angle < (tomo_size.y); angle++){
        
            t = ( x * costable[angle] + y * sintable[angle] + tomo_size.x / 2 );
            T = int(t);
        
            if ( ( T >= 0 ) && ( T < ( tomo_size.x - 1 ) ) ){
                frac = t-T;

                sum += sino[tomo_size.y * tomo_size.x * k + angle * tomo_size.x + T] * (1.0f - frac) + sino[angle * tomo_size.x + T + 1] * frac;
            }
        }        

        /* Normalizes and transforms the recon to 
        its desired range (controled by the threshold variable)
        and to its desired data type (uint8, unint16, and etc...)
        */
        BasicOps::set_pixel(recon, sum*norm, i, ind, recon_size.x, threshold, datatype);
    } 

    __global__ void KRadon_RT(float* restrict frames, const float* image, int nrays, int nangles)
    {
        int ray = (blockDim.x * blockIdx.x + threadIdx.x);
        int ang = (blockDim.y * blockIdx.y + threadIdx.y);
        const size_t sizef = nrays*nrays;
        
        if ( ray>=nrays || ang >= nangles )
            return;
        
        float btheta = float(M_PI)*float(ang + nangles/2)/nangles;
        
        float ct = nrays/2;
        float cos_t = cosf(btheta + 1E-5f);
        float sin_t = sinf(btheta + 1E-5f);

        float x = ct - 2.0f*ct*cos_t + (ray-nrays/2)*sin_t;
        float y = ct - 2.0f*ct*sin_t - (ray-nrays/2)*cos_t;

        float tx0 = -x/cos_t;
        float tx1 = (nrays-x)/cos_t;

        float ty0 = -y/sin_t;
        float ty1 = (nrays-y)/sin_t;

        float d1 = fmaxf(fminf(tx0,tx1), fminf(ty0,ty1));
        int d2 = int(ceil(fminf(fmaxf(tx0,tx1), fmaxf(ty0,ty1)) - d1)+0.1f);

        x += d1*cos_t;
        y += d1*sin_t;
        
        float radon = 0;
        for(int s=0; s<d2; s++)
        {
            radon += image[(int(y+0.5f)*nrays + int(x+0.5f))%sizef + blockIdx.z*sizef];

            x += cos_t;
            y += sin_t;
        }

        frames[blockIdx.z*size_t(nrays*nangles) + nrays*ang + ray] = radon;
    }

}

extern "C"
{
	void GRadon(int device, float* _frames, float* _image, int nrays, int nangles, int blocksize)
	{
		cudaSetDevice(device);

		rImage frames(nrays,nangles,blocksize);
		rImage image(_image,nrays,nrays,blocksize);

		KRadon_RT<<<dim3(nrays/64,nangles,blocksize),64>>>(frames.gpuptr, image.gpuptr, nrays, nangles);

		frames.CopyTo(_frames);
		cudaDeviceSynchronize();
	}

	void GBackprojection(int device, float* _recon, float* _sino, int nrays, int nangles, int blocksize)
	{
		cudaSetDevice(device);

		rImage sintable(nangles,1);
		rImage costable(nangles,1);

		for(int a=0; a<nangles; a++)
		{
			sintable.cpuptr[a] = sinf(float(M_PI)*a/float(nangles));
			costable.cpuptr[a] = cosf(float(M_PI)*a/float(nangles));
		}

		sintable.LoadToGPU();
		costable.LoadToGPU();

		rImage recon(nrays,nrays,blocksize);
		rImage sino(_sino,nrays,nangles,blocksize);

		KBackProjection_RT<<<dim3(nrays/64,nrays,blocksize),64>>>(
			(char*)recon.gpuptr, sino.gpuptr, nrays, nrays, nangles, EType::TypeEnum::FLOAT32, 0, sintable.gpuptr, costable.gpuptr);

		recon.CopyTo(_recon);
		cudaDeviceSynchronize();

	}
}