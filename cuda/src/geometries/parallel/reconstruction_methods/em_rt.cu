#include <cstddef>
#include "common/configs.hpp"
#include "common/opt.hpp"
#include "geometries/parallel/em.hpp"

//---------------------------
// transmission-EM algorithm
//---------------------------

extern "C" {  

    void get_tEM_RT_MultiGPU(int* gpus, int ngpus,
    float* obj, float* count, 
    float *flat, float* angles, 
    float *paramf, int *parami)
    {
        int i, Maxgpudev;
        
        /* Multiples devices */
        HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

        /* If devices input are larger than actual devices on GPU, exit */
        for(i = 0; i < ngpus; i++) 
            assert(gpus[i] < Maxgpudev && "Invalid device number.");

        CFG configs; GPU gpu_parameters;

        setEMRTParameters(&configs, paramf, parami);
        // printEMRTParameters(&configs);

        setGPUParameters(&gpu_parameters, configs.tomo.padsize, ngpus, gpus);
        // printGPUParameters(&gpu_parameters);

        /* Projection data sizes */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;
        int nslices  = configs.tomo.size.z;
        
        /* Object (reconstruction) data sizes */
        int nx       = configs.obj.size.x;
        int ny       = configs.obj.size.y;

        int blockgpu = (nslices + ngpus - 1) / ngpus;
        int subblock, ptr = 0; 

        if (ngpus == 1){ /* 1 device */

            get_tEM_RT_GPU(configs, gpu_parameters, obj, count, flat, angles, nslices, gpus[0]);

        }else{
        /* Launch async Threads for each device.
        See future c++ async launch.
        */

            std::vector<std::future<void>> threads = {};
            threads.reserve(ngpus);

            for (i = 0; i < ngpus; i++){
                
                subblock   = min(nslices - ptr, blockgpu);

                threads.push_back( std::async( std::launch::async, 
                                                get_tEM_RT_GPU, 
                                                configs, gpu_parameters, 
                                                obj   + (size_t)   nx *      ny * ptr,
                                                count + (size_t)nrays * nangles * ptr, 
                                                flat  + (size_t)nrays           * ptr, 
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

        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    void get_tEM_RT_GPU(CFG configs, GPU gpus, 
    float *obj, float *count, float *flat, float *angles, 
    int sizez, int ngpu)
    {
        int sizeImage = configs.obj.size.x;
        int nrays     = configs.tomo.size.x;
        int nangles   = configs.tomo.size.y;

        int i; 
        int blocksize        = configs.blocksize;
        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, configs.total_required_mem_per_slice_bytes, true, BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(sizez, blocksize_aux);
        }
        int ind_block = (int)ceil( (float) sizez / blocksize );

        HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Allocate GPU memory for the input and output image */

        float *dcount  = opt::allocGPU<float>((size_t)nrays * nangles * blocksize);
        float *dflat   = opt::allocGPU<float>((size_t)nrays * nangles);
        float *dobj    = opt::allocGPU<float>((size_t)sizeImage * sizeImage * blocksize);

        float *back    = opt::allocGPU<float>((size_t)sizeImage * sizeImage * blocksize);
        float *backcounts  = opt::allocGPU<float>((size_t)sizeImage * sizeImage * blocksize);
        float *temp    = opt::allocGPU<float>((size_t)nrays * nangles * blocksize);

        float *dangles = opt::allocGPU<float>( nangles );
        opt::CPUToGPU<float>(angles, dangles, nangles);

        HANDLE_ERROR( cudaPeekAtLastError() );

        /* Loop for each batch of size 'batch' in threads */
        int ptr = 0, subblock = 0;
        size_t ptr_block_tomo = 0, ptr_block_obj = 0, ptr_block_flat = 0;

        for (i = 0; i < ind_block; i++){

            subblock        = min(sizez - ptr, blocksize);
            ptr_block_tomo  = (size_t)nrays * nangles * ptr;
            ptr_block_obj   = (size_t)sizeImage * sizeImage * ptr;
            ptr_block_flat  = (size_t)nrays * ptr;

            /* Update pointer */
            ptr = ptr + subblock;

            opt::CPUToGPU<float>(count + ptr_block_tomo, dcount, (size_t)nrays * nangles * subblock);
            opt::CPUToGPU<float>(flat + ptr_block_flat, dflat, (size_t)nrays *subblock);
            opt::CPUToGPU<float>(obj + ptr_block_obj, dobj, (size_t)sizeImage * sizeImage * subblock);

            get_tEM_RT( configs, gpus, dobj, dcount, dflat, dangles,
                        backcounts, temp, back, subblock);

            opt::GPUToCPU<float>(obj + ptr_block_obj, dobj, (size_t)sizeImage * sizeImage * subblock);

        }

        HANDLE_ERROR(cudaFree(temp));
        HANDLE_ERROR(cudaFree(back));
        HANDLE_ERROR(cudaFree(backcounts));
        HANDLE_ERROR(cudaFree(dobj));
        HANDLE_ERROR(cudaFree(dcount));
        HANDLE_ERROR(cudaFree(dflat));
        HANDLE_ERROR(cudaFree(dangles));
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    void get_tEM_RT(CFG configs, GPU gpus, 
    float *output, float *count, float *flat, float *angles, 
    float *backcounts, float *temp, float *back,
    int blockSize)
    {
        int k;
        int niter     = configs.em_iterations;
        int sizeImage = configs.obj.size.x;
        int nrays     = configs.tomo.size.x;
        int nangles   = configs.tomo.size.y;

        //GRID and BLOCKS SIZE
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlockD((int)ceil((nrays)/threadsPerBlock.x)+1,
		                (int)ceil((nangles)/threadsPerBlock.y)+1,
		                (int)ceil(blockSize/threadsPerBlock.z)+1);

        dim3 gridBlockF((int)ceil((sizeImage)/threadsPerBlock.x)+1,
                        (int)ceil((sizeImage)/threadsPerBlock.y)+1,
		                (int)ceil(blockSize/threadsPerBlock.z)+1);

        /* Commented to add initial guess: `output` variable is also the initial guess */ 
        // kernel_ones<<<gridBlockF,threadsPerBlock>>>(output, sizeImage, nrays, nangles, blockSize); 
        
        kernel_backprojection<<<gridBlockF,threadsPerBlock>>>(backcounts, count, angles, sizeImage, nrays, nangles, blockSize);

        for( k = 0; k < niter; k++ ){

            kernel_radon<<<gridBlockD,threadsPerBlock>>>(temp, output, angles, sizeImage, nrays, nangles, blockSize, 1.0);
            
            kernel_flatTimesExp<<<gridBlockD,threadsPerBlock>>>(temp, flat, sizeImage, nrays, nangles, blockSize);
            
            kernel_backprojection<<<gridBlockF,threadsPerBlock>>>(back, temp, angles, sizeImage, nrays, nangles, blockSize);
            
            kernel_update<<<gridBlockF,threadsPerBlock>>>(output, back, backcounts, sizeImage, nrays, nangles, blockSize);
            
        }

    }
}

//----------------------
// emission-EM algorithm
//----------------------

extern "C"{   

    void get_eEM_RT_MultiGPU(int* gpus, int ngpus,
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

        setEMRTParameters(&configs, paramf, parami);
        // printEMRTParameters(&configs);

        setGPUParameters(&gpu_parameters, configs.tomo.padsize, ngpus, gpus);

        int subvolume = (configs.tomo.size.z + ngpus - 1) / ngpus;
        int subblock, ptr = 0; 

        if (ngpus == 1){ /* 1 device */

            get_eEM_RT_GPU(configs, gpu_parameters, obj, tomogram, angles, subvolume, gpus[0]);

        }else{
        /* Launch async Threads for each device.
        See future c++ async launch
        */

            std::vector<std::future<void>> threads = {};
            threads.reserve(ngpus);

            for (i = 0; i < ngpus; i++){
                
                subblock   = min(configs.tomo.size.z - ptr, subvolume);

                threads.push_back( std::async( std::launch::async, 
                                                get_eEM_RT_GPU, 
                                                configs, gpu_parameters, 
                                                obj    + (size_t)configs.obj.size.x * configs.obj.size.y * ptr,
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

        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    void get_eEM_RT_GPU(CFG configs, GPU gpus, 
    float *obj, float *tomogram, float *angles, 
    int sizez, int ngpu)
    {
        int i; 
        int blocksize        = configs.blocksize;
        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, configs.total_required_mem_per_slice_bytes, true, BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(sizez, blocksize_aux);
        }
        int ind_block = (int)ceil( (float) sizez / blocksize );

        HANDLE_ERROR(cudaSetDevice(ngpu));

        float *dobj, *dtomo, *dangles;

        /* Allocate GPU memory for the input and output image */ 
        HANDLE_ERROR(cudaMalloc((void **)&dobj   ,sizeof(float) * (size_t) configs.obj.size.x *  configs.obj.size.y * blocksize));  
        HANDLE_ERROR(cudaMalloc((void **)&dtomo  ,sizeof(float) * (size_t)configs.tomo.size.x * configs.tomo.size.y * blocksize));
        HANDLE_ERROR(cudaMalloc((void **)&dangles,sizeof(float) * configs.tomo.size.y));

        HANDLE_ERROR(cudaMemcpy(dangles, angles, sizeof(float) * configs.tomo.size.y, cudaMemcpyHostToDevice));	

        /* Loop for each batch of size 'batch' in threads */
        int ptr = 0, subblock; size_t ptr_block_tomo = 0, ptr_block_obj = 0;

        for (i = 0; i < ind_block; i++){

            subblock        = min(sizez - ptr, blocksize);
            ptr_block_tomo  = (size_t)configs.tomo.size.x  * configs.tomo.size.y  * ptr;
            ptr_block_obj = (size_t)configs.obj.size.x * configs.obj.size.y * ptr;
            
            /* Update pointer */
            ptr = ptr + subblock;

            HANDLE_ERROR(cudaMemcpy(dtomo, tomogram + ptr_block_tomo, sizeof(float) * (size_t)configs.tomo.size.x * configs.tomo.size.y * subblock, cudaMemcpyHostToDevice));	
            HANDLE_ERROR(cudaMemcpy(dobj, obj + ptr_block_obj, (size_t)configs.obj.size.x * configs.obj.size.y * subblock * sizeof(float), cudaMemcpyHostToDevice));
            
            get_eEM_RT( configs, gpus, dobj, dtomo, dangles, subblock);                           
            
            HANDLE_ERROR(cudaMemcpy(obj + ptr_block_obj, dobj, (size_t)configs.obj.size.x * configs.obj.size.y * subblock * sizeof(float), cudaMemcpyDeviceToHost));
        }
        HANDLE_ERROR(cudaFree(dobj));
        HANDLE_ERROR(cudaFree(dtomo));
        HANDLE_ERROR(cudaFree(dangles));
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

}

extern "C" {

    void get_eEM_RT(CFG configs, GPU gpus, 
    float *output, float *tomo, float *angles, int blockSize)
    {
        int k;
        int niter     = configs.em_iterations;
        int sizeImage = configs.obj.size.x;
        int nrays     = configs.tomo.size.x;
        int nangles   = configs.tomo.size.y;

        //GRID and BLOCKS SIZE
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlockD((int)ceil((nrays)/threadsPerBlock.x)+1,
		                (int)ceil((nangles)/threadsPerBlock.y)+1,
		                (int)ceil(blockSize/threadsPerBlock.z)+1);

        dim3 gridBlockF((int)ceil((sizeImage)/threadsPerBlock.x)+1,
                        (int)ceil((sizeImage)/threadsPerBlock.y)+1,
		                (int)ceil(blockSize/threadsPerBlock.z)+1);

        float *backones, *temp;

        HANDLE_ERROR(cudaMalloc((void **)&backones  ,sizeof(float) * (size_t)sizeImage * sizeImage * blockSize));
        HANDLE_ERROR(cudaMalloc((void **)&temp      ,sizeof(float) * (size_t)nrays     * nangles   * blockSize));

        /* Commented to add initial guess: `output` variable is also the initial guess */
        // kernel_ones<<<gridBlockF,threadsPerBlock>>>(output, sizeImage, nrays, nangles, blockSize);

        kernel_backprojectionOfOnes<<<gridBlockF,threadsPerBlock>>>(backones, angles, sizeImage, nrays, nangles, blockSize);

        for( k = 0; k < niter; k++ ){

            kernel_radonWithDivision<<<gridBlockD,threadsPerBlock>>>(temp, output, tomo, angles, sizeImage, nrays, nangles, blockSize, 1.0);
                
            kernel_backprojectionWithUpdate<<<gridBlockF,threadsPerBlock>>>(output, temp, backones, angles, sizeImage, nrays, nangles, blockSize);
            
            HANDLE_ERROR(cudaDeviceSynchronize());
        }
        HANDLE_ERROR(cudaFree(temp));
        HANDLE_ERROR(cudaFree(backones));
    }
}

