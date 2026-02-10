#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <cstdio>
#include "common/configs.hpp"
#include "common/opt.hpp"
#include "pipelines/pipeline.hpp"

#include <thread>

extern "C"{
    void ReconstructionPipeline(CFG configs, WKP *workspace, int tomoblock, int objblock)
    {
        cudaStream_t nstream = 0;

        // int nrays   = configs.tomo.size.x;
        // int nangles = configs.tomo.size.y;
        // int nflats  = configs.flat.size.z;

        // int nx      = configs.obj.size.x;
        // int ny      = configs.obj.size.y;

        getBackgroundCorrectionMethods(configs, workspace, tomoblock, nstream);

        getAlignMethods(configs, workspace, tomoblock, nstream);

        getRingsMethods(configs, workspace, tomoblock, nstream);

        getReconstructionMethods(configs, workspace, tomoblock, objblock, nstream);
    }
}

extern "C" {

    void ReconstructionPipeline_GPU(CFG configs, 
    float *object, float *data, 
    float *flats, float *darks, float *angles, 
    int sizez, int gpu_device)
    {
        /* Initialize GPU device */
        HANDLE_ERROR(cudaSetDevice(gpu_device));

        /* Compute total memory used on a singles slice */
        size_t total_required_mem_per_slice_bytes = (
            calcSliceMemoryBytes(configs.tomo)           + // Tomo slice
            calcSliceMemoryBytes(configs.obj)            + // Reconstructed object slice
            calcPaddedSliceMemoryBytes(configs.obj)      + // Reconstructed padded object slice
            2 * calcPaddedSliceMemoryBytes(configs.tomo) + // Tomo padded slice
            configs.tomo.size.y * sizeof(float)            // angles
            );
        
        int blocksize = getGPUBlocksize(configs.tomo.blocksize, sizez, total_required_mem_per_slice_bytes, 32, true);
        int ind_block = getNumberOfBlocks(sizez, blocksize); 
        
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nslices = configs.tomo.size.z;
        int nflats  = configs.flat.size.z;
        int nAngles = opt::get_angleList_dimension(configs.tomo.size, configs.AlignParam.method); /* True Angle value - excentric tomo */

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;

        /* Local GPUs Pointers: allocation */
        WKP *workspace = Initialize_workspace(  dim3(nrays,nangles,blocksize),
                                                dim3(nx,ny,blocksize),
                                                dim3(nrays,blocksize,nflats),
                                                dim3(nrays,blocksize,1),
                                                configs.tomo.pad,
                                                configs.obj.pad,
                                                nAngles);

        opt::CPUToGPU<float>(angles, workspace->angles, nAngles);

        printf("blocksize: %d\n",blocksize);
        printf("configs.tomo.blocksize: %d\n",configs.tomo.blocksize);
        printf("ind_block: %d\n",ind_block);
        printf("sizez: %d\n",sizez);
        fflush(stdout);
       
        int ptr = 0, subblock; 

        for (int i = 0; i < ind_block; i++){

            subblock = getSubblock(sizez - ptr, blocksize);

            /* Copy data from host to device */
            opt::CPUToGPU<float>( data + (size_t)ptr * nrays * nangles, workspace->tomo, (size_t)nrays *  nangles * subblock);
            opt::CPUToGPU<float>(flats + (size_t)ptr * nrays *  nflats, workspace->flat, (size_t)nrays * subblock *   nflats);
            opt::CPUToGPU<float>(darks + (size_t)ptr * nrays          , workspace->dark, (size_t)nrays * subblock           );

            /* Enter Reconstruction Pipeline */
            ReconstructionPipeline(configs, workspace, subblock, subblock);

            /* Copy Reconstructed data from device to host */
            opt::GPUToCPU<float>(object + (size_t)ptr * nx * ny, workspace->obj, (size_t)nx * ny * subblock);

            /* Copy Processed tomogram data from device to host */
            opt::GPUToCPU<float>(data + (size_t)ptr * nrays * nangles, workspace->tomo, (size_t)nrays * nangles * subblock);

            /* Update pointer */
            ptr = ptr + subblock;
        }
        /* Dealocate workspace variables */
        freeWorkspace(workspace);
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

extern "C"{
    void ReconstructionPipelineMultiGPU(CFG configs, int *gpus, int ngpus,
    float *object, float *data, float *flats, float *darks, float *angles)
    {
        int i, Maxgpu;

        /* Multiples devices */
        cudaGetDeviceCount(&Maxgpu);

        /* If devices input are larger than actual devices on GPU, exit */
        for(i = 0; i < ngpus; i++)
            assert(gpus[i] < Maxgpu && "Invalid device number.");

        int nrays     = configs.tomo.size.x;
        int nangles   = configs.tomo.size.y;
        int nslices   = configs.tomo.size.z;
        int nflats    = configs.flat.size.z;

        int nx        = configs.obj.size.x;
        int ny        = configs.obj.size.y;

        int subvolume = (nslices + ngpus - 1) / ngpus;
        int ptr       = 0; 
        int subblock;

        printf("tomo shape:\n");
        printDim(configs.tomo.size);
        printf("tomo pad shape:\n");
        printDim(configs.tomo.pad);
        printf("obj shape:\n");
        printDim(configs.obj.size);
        printf("obj pad shape:\n");
        printDim(configs.obj.pad);
        printf("nflats: %d\n",nflats);
        printf("ngpus: %d\n",ngpus);
        printf("configs.tomo.blocksize: %d\n",configs.tomo.blocksize);
        printf("subvolume: %d\n",subvolume);
        fflush(stdout);

        if (ngpus == 1){ /* 1 device */
            
            ReconstructionPipeline_GPU( configs, 
                                        object,
                                        data, 
                                        flats, 
                                        darks, 
                                        angles, 
                                        nslices,
                                        gpus[0]);

		}else{
            //See future c++ async launch
			std::vector<std::future<void>> threads = {};
            threads.reserve(ngpus);

            for (i = 0; i < ngpus; i++){
				
				subblock = getSubblock(nslices - ptr, subvolume);

				threads.push_back(  std::async( std::launch::async, 
                                    ReconstructionPipeline_GPU, 
                                    configs, 
                                    object + (size_t)ptr *    nx *      ny,
                                    data   + (size_t)ptr * nrays * nangles, 
                                    flats  + (size_t)ptr * nrays * nflats, 
                                    darks  + (size_t)ptr * nrays, 
                                    angles, 
                                    subblock,
                                    gpus[i]));

                /* Update pointer */
				ptr = ptr + subblock;		
			}
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}
        HANDLE_ERROR(cudaGetLastError());
    }
}

/* Launch independent processe each blocksize version */
// extern "C" {

//     void _ReconstructionPipeline_GPU(CFG configs, Process process,
//     float *object, float *data, 
//     float *flats, float *darks, float *angles)
//     {
//         /* Initialize GPU device */
//         HANDLE_ERROR(cudaSetDevice(process.gpu));

//         printf("Process number %d in gpu[%d] = %d \n", process.process, process.gpu_proc_ind, process.gpu);
//         fflush(stdout);

//         int nrays      = configs.tomo.size.x;
//         int nangles    = configs.tomo.size.y;
//         int nslices    = configs.tomo.size.z;
//         int tomoblock = process.tomobatch_size;

//         int nx         = configs.obj.size.x;
//         int ny         = configs.obj.size.y;
//         int objblock   = process.objbatch_size;

//         size_t ptr     = process.tomo_index * tomoblock * nrays;

//         /* Local GPUs Pointers: allocation */
//         WKP *workspace = Initialize_workspace(  dim3(nrays,nangles,tomoblock),
//                                                 dim3(nx,ny,objblock),
//                                                 dim3(nrays,tomoblock,configs.nflats),
//                                                 dim3(nrays,tomoblock,1),
//                                                 configs.tomo.pad,
//                                                 configs.obj.pad);

//         /* Copy data from host to device */
//         opt::CPUToGPU<float>(flats + ptr * configs.nflats         , workspace->flat  , (size_t)nrays * tomoblock * configs.nflats);
//         opt::CPUToGPU<float>(darks + ptr                          , workspace->dark  , (size_t)nrays * tomoblock                 );
//         opt::CPUToGPU<float>(angles                               , workspace->angles, nangles                                   );
//         opt::CPUToGPU<float>( data + (size_t)process.tomoptr_index, workspace->tomo  , (size_t)process.tomoptr_size              );

//         /* Enter Reconstruction Pipeline */
//         ReconstructionPipeline(configs, workspace, tomoblock, objblock);

//         /* Copy Reconstructed data from device to host */
//         opt::GPUToCPU<float>(object + (size_t)process.objptr_index, workspace->obj, (size_t)process.objptr_size);

//         /* Copy Processed tomogram data from device to host */
//         opt::GPUToCPU<float>(data + (size_t)process.tomoptr_index, workspace->tomo, (size_t)process.tomoptr_size);

//         /* Dealocate workspace variables */
//         freeWorkspace(workspace);
//         HANDLE_ERROR(cudaDeviceSynchronize());
//     }
// }

// extern "C"{
//     void ReconstructionPipelineProcessMultiGPU(CFG configs, int *gpus, int ngpus,
//         float *object, float *data, float *flats, float *darks, float *angles)
//     {
//         /* Multiples devices */
//         int Maxgpu;
//         cudaGetDeviceCount(&Maxgpu);

//         /* If devices input are larger than actual devices on GPU, exit */
//         for(int i = 0; i < ngpus; i++)
//             assert(gpus[i] < Maxgpu && "Invalid device number.");

//         /* Compute total memory used on a singles slice */
//         size_t total_required_mem_per_slice_bytes = (
//                                                         calcSliceMemoryBytes(configs.tomo)           + // Tomo slice
//                                                         calcSliceMemoryBytes(configs.obj)            + // Reconstructed object slice
//                                                         calcPaddedSliceMemoryBytes(configs.obj)      + // Reconstructed padded object slice
//                                                         2 * calcPaddedSliceMemoryBytes(configs.tomo) + // Tomo padded slice
//                                                         configs.tomo.size.y * sizeof(float)            // angles
//                                                     );
        
//         /* Set total number of processes to be sent to the GPUs */
//         int total_number_of_processes = getTotalProcesses(  ngpus, 
//                                                             configs.tomo.size.z, 
//                                                             configs.tomo.blocksize,
//                                                             total_required_mem_per_slice_bytes, 
//                                                             true);
//         /* Set processes pipeline for different geometries */
//         Process *process = setProcesses(configs, gpus, ngpus, total_number_of_processes);

//         printf("Total number of process = %d \n", total_number_of_processes);
//         printf("ngpus = %d \n", ngpus);
//         printf("configs.tomo.blocksize = %d \n", configs.tomo.blocksize);
//         fflush(stdout);

//         for(int i = 0; i < ngpus; i++)
//             printf("gpus[%d] = %d \n", i,gpus[i]);

//         for (int i = 0; i < total_number_of_processes; i++)
//             printf("Init - Process number %d in gpu[%d] = %d \n", process[i].process, process[i].gpu_proc_ind, process[i].gpu);
        
//         fflush(stdout);

//         /* Launch processes */
//         std::vector<std::future<void>> threads_pipeline = {};
//         threads_pipeline.reserve(ngpus); 
//         // std::vector<std::thread> threads_pipeline = {};

//         for (int p = 0; p < total_number_of_processes; p++) {

//             // threads_pipeline.emplace_back(  std::thread(
//             threads_pipeline.push_back(  std::async( std::launch::async,
//                                             _ReconstructionPipeline_GPU,
//                                             configs, 
//                                             process[p],
//                                             object, 
//                                             data, 
//                                             flats, 
//                                             darks,
//                                             angles));

//             if (( p + 1 ) % ngpus == 0) {

//                 for (int g = 0; g < ngpus; g++)
//                     threads_pipeline[g].get(); 
//                     // threads_pipeline[g].join();

//                 threads_pipeline.clear();
//                 HANDLE_ERROR(cudaDeviceSynchronize());
//             }
//         }

//         HANDLE_ERROR(cudaGetLastError());

//         /* Free process (array of structs) */
//         free(process);
//     }
// }