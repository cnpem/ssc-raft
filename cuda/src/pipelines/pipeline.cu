#include "../../inc/pipeline.h"
#include "../../inc/processing.h"
#include "../../inc/reconstructions.h"

using std::thread;

extern "C"{
    void ReconstructionPipeline(float *recon, float *data, float *flats, float *darks, float *angles,
	float *parameters_float, int *parameters_int, int *flags,
	int *gpus, int ngpus)
	{
        int i, Maxgpu;
        int total_number_of_processes;

		/* Multiples devices */
		cudaGetDeviceCount(&Maxgpu);

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpu && "Invalid device number.");

        CFG configs; GPU gpu_parameters;

        setReconstructionParameters(&configs, parameters_float, parameters_int, flags);

        setGPUParameters(&gpu_parameters, configs.tomo.npad, ngpus, gpus);

        /* Set total number of processes to be sent to the GPUs */
        total_number_of_processes = getTotalProcesses(configs, gpu_parameters);
                
        /* Set processes pipeline for different geometries */
        Process *process = setProcesses(configs, gpu_parameters, total_number_of_processes);

        // clock_t b_begin = clock();

        _setReconstructionPipeline(configs, process, gpu_parameters, recon, data, flats, darks, angles, total_number_of_processes);

        // clock_t b_end = clock();
        // double time = double(b_end - b_begin) / CLOCKS_PER_SEC;

        // cudaDeviceSynchronize();

        HANDLE_ERROR(cudaGetLastError());

        /* Free process (array of structs) */
        free(process);
    }
}


extern "C"{
    void _setReconstructionPipeline(CFG configs, Process *process, GPU gpus,
    float *recon, float *data, 
    float *flats, float *darks, float *angles,
    int total_number_of_processes)
    {
        int gpu_index, process_index = 0;

        std::vector<thread> threads_pipeline;

        while (process_index < total_number_of_processes){

            threads_pipeline.emplace_back( thread(_ReconstructionProcessPipeline, configs, process[process_index], gpus, recon, data, flats, darks, angles) );		

            process_index = process_index + 1;

            if (process_index % gpus.ngpus == 0){

                for (gpu_index = 0; gpu_index < gpus.ngpus; gpu_index++)
                    threads_pipeline[gpu_index].join();

                threads_pipeline.clear();
                cudaDeviceSynchronize();
            }
        }
        
    }
}


extern "C" {

	void _ReconstructionProcessPipeline(CFG configs, Process process, GPU gpus, 
    float *recon, float *frames, float *flats, float *darks, float *angles)
	{	
		/* Initialize GPU device */
		HANDLE_ERROR(cudaSetDevice(process.index_gpu));
		
        /* Local GPUs Pointers: allocation */
        WKP *workspace = allocateWorkspace(configs, process);

        /* Copy data from host to device */
        HANDLE_ERROR(cudaMemcpy(workspace->angles, angles, configs.tomo.size.y * sizeof(float), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(workspace->tomo, frames + process.ind_tomo, process.n_tomo * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(workspace->flat,  flats + process.ind_tomo, process.n_tomo * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(workspace->dark,  darks + process.ind_tomo, process.n_tomo * sizeof(float), cudaMemcpyHostToDevice));

        /* Enter Reconstruction Pipeline */
        _ReconstructionPipeline(configs, workspace, process, gpus);

        /* Copy Reconstructed data from device to host */
        HANDLE_ERROR(cudaMemcpy(&recon[process.ind_recon], workspace->recon, process.n_recon * sizeof(float), cudaMemcpyDeviceToHost));

        freeWorkspace(workspace, configs);

		// cudaDeviceSynchronize();
	}
}


extern "C"{
    void _ReconstructionPipeline(CFG configs, WKP *workspace, Process process, GPU gpus)
    {
        float lambda_computed = 0;

        dim3 tomo_size  = dim3(configs.tomo.size.x , configs.tomo.size.y , process.batch_size_tomo );
        dim3 recon_size = dim3(configs.recon.size.x, configs.recon.size.y, process.batch_size_recon);

        if( configs.flags.do_flat_dark_correction == 1)
            getFlatDarkCorrection(workspace->tomo, workspace->flat, workspace->dark, tomo_size, configs.numflats, gpus);

        if( configs.flags.do_flat_dark_log == 1) 
            getLog(workspace->tomo, tomo_size, gpus);

        if( configs.flags.do_phase_filter == 1) 
            getPhaseFilter(gpus, configs.geometry, workspace->tomo, configs.phase_filter_type, configs.phase_filter_reg, tomo_size, configs.tomo.npad);

        if( configs.flags.do_rings == 1) 
            lambda_computed = getRings(workspace->tomo, tomo_size, configs.rings_lambda, configs.rings_block, gpus);
        
        // if( configs.flags.do_rotation == 1 && configs.flags.do_rotation_auto_offset == 1) 
        //     int rotation_axis_offset = getRotAxisOfsset();
        // else
        //     rotation_axis_offset = configs.rotation_axis_offset;

        // if( configs.flags.do_alignment == 1) 
        //     getTomogramAlignment();

        // if( configs.flags.do_rotation == 1 && configs.flags.do_rotation_correction == 1 && configs.reconstruction_method != 0) 
        //     setRotAxisCorrection();

        switch (configs.geometry.geometry){
            case 0:
                /* Parallel */
                getReconstructionParallel(configs, process, gpus, workspace);
                break;
            case 1:
                /* Conebeam */
                getReconstructionConebeam(configs, process, gpus, workspace);
                break;
            case 2:
                /* Fanbeam */
                printf("No pipeline for Fanbeam yet. \n");
                // getReconstructionFanbeam(configs, process, gpus, workspace);
                break;
            default:
                printf("Nope. \n");
                break;
        }	
        

    }
}

extern "C"{

    void getReconstructionParallel(CFG configs, Process process, GPU gpus,  WKP *workspace)
    {
        dim3 tomo_size  = dim3(configs.tomo.size.x , configs.tomo.size.y , process.batch_size_tomo );
        dim3 pad_size   = dim3(configs.tomo.npad.x , configs.tomo.size.y , process.batch_size_tomo );
        dim3 recon_size = dim3(configs.recon.size.x, configs.recon.size.y, process.batch_size_recon);

        switch (configs.reconstruction_method){
            case 0:
                /* FBP */
                getFBP( configs, gpus, workspace->recon, workspace->tomo, workspace->angles, 
                        tomo_size, pad_size, recon_size);
                break;
            case 1:
                /* BST */
                
                break;
            case 2:
                /* EM RT eEM */
                get_eEM_RT(configs, gpus, workspace->recon, workspace->tomo, workspace->angles, process.batch_size_tomo);
                break;
            case 3:
                /* EM RT tEM */
                get_tEM_RT(configs, gpus, workspace->recon, workspace->tomo, workspace->flat, workspace->angles, process.batch_size_tomo);
                break;
            case 4:
                /* EM RT eEM TV */
                
                break;
            case 5:
                /* EM RT tEM TV */
                
                break;
            case 6:
                /* EM FST eEM */
                
                break;
            case 7:
                /* EM FST eEM TV*/
                
                break;
            case 8:
                /* EM FST tEM TV*/
                
                break;
            default:
                printf("Nope. \n");
                break;
        }	
    }

    void getReconstructionConebeam(CFG configs, Process process, GPU gpus, WKP *workspace)
    {

        switch (configs.reconstruction_method){
            case 0:
                /* FDK */
                break;
            case 1:
                /* EM Conical eEM*/
                
                break;
            case 2:
                /* EM Conical tEM*/
                
                break;
            case 3:
                /* EM Conical eEM TV*/
                
                break;
            case 4:
                /* EM Conical tEM TV*/
                
                break;
            default:
                printf("Nope. \n");
                break;
        }	
    }

    // void getReconstructionFanbeam(CFG configs, Process process, GPU gpus,  WKP *workspace)
    // {

    //     switch (configs.reconstruction_method){
    //         case 0:
    //             /* FDK */
    //             printf("No filter was selected!");
    //             break;
    //         case 1:
    //             /* EM Conical */
                
    //             break;
    //         case 2:
    //             /* FBP */
                
    //             break;
    //         case 3:
    //             /* BST */
                
    //             break;
    //         case 4:
    //             /* EM RT */
                
    //             break;
    //         case 5:
    //             /* EM FST */
                
    //             break;

    //         default:
    //             printf("Nope. \n");
    //             break;
    //     }	
    // }
}

/* ==================================================================== */
/* ==================================================================== */
/* ==================================================================== */
/* ==================================================================== */
/* ==================================================================== */
/* Archived Functions */
//
// extern "C" {

// 	void _ReconstructionThreads(CFG configs, float *recon, float *frames, float *flats, float *darks, float *angles, int max_block_size, int ngpu)
// 	{	
// 		/* Initialize GPU device */
// 		HANDLE_ERROR(cudaSetDevice(ngpu))
		
// 		int i, sub_block; 

//         int sub_block_size = min(max_block_size,32);

//         int block_index = (int)ceil( (float) max_block_size / sub_block_size );
// 		int ptr = 0; 
// 		size_t ptr_block = 0;

//         // GPUs Pointers: declaration and allocation
//         WKP *workspace = allocateWorkspace(configs, max_block_size);

//         /* Send angles array to the device - needs to do only once */
//         HANDLE_ERROR(cudaMemcpy(workspace->angles, angles, configs.nangles * sizeof(float), cudaMemcpyHostToDevice));


// 		/* Loop for each batch of size 'sub_block' */
// 		for (i = 0; i < block_index; i++){

// 			sub_block = min(max_block_size - ptr, sub_block_size);
// 			ptr_block = (size_t)configs.nrays * configs.angles * ptr;

//             /* Update pointer */
// 			ptr = ptr + sub_block;

//             /* Copy data from host to device */
//             HANDLE_ERROR(cudaMemcpy(workspace->tomo  , frames, ptr_block * sizeof(float), cudaMemcpyHostToDevice));
//             HANDLE_ERROR(cudaMemcpy(workspace->flat  , flats , ptr_block * sizeof(float), cudaMemcpyHostToDevice));
//             HANDLE_ERROR(cudaMemcpy(workspace->dark  , darks , ptr_block * sizeof(float), cudaMemcpyHostToDevice));


//             _Reconstruction(configs, workspace, max_block_size, ngpu);

//             /* Copy answer to CPU */
//             HANDLE_ERROR(cudaMemcpy(recon, workspace->recon, ptr_block * sizeof(float), cudaMemcpyDeviceToHost));

// 		}
        
// 		cudaDeviceSynchronize();
// 	}
// }

// extern "C"{
// 	void PipelineReconstruction(float *recon, float *data, float *flats, float *darks, float *angles,
// 	float *parameters_float, int *parameters_int,
// 	int *gpus, int ngpus)
// 	{	
// 		int i, Maxgpu;

// 		/* Multiples devices */
// 		cudaGetDeviceCount(&Maxgpu);

// 		/* If devices input are larger than actual devices on GPU, exit */
// 		for(i = 0; i < ngpus; i++) 
// 			assert(gpus[i] < Maxgpu && "Invalid device number.");

//         CFG configs;

//         /* Transpose the data from (nangles, nslices, nrays) to (nslices,nangles,nrays) */ 
// 		getTransposedData(data , configs);
//         getTransposedFlat(flats, configs);
//         getTransposedDark(darks, configs);

// 		int max_block_gpu = ( configs.nslices + ngpus - 1 ) / ngpus;
//         int sub_block_gpu, ptr_gpu = 0; 
// 		size_t ptr_block_gpu = 0;


// 		/* Launch async Threads for each device.
// 			Each device solves a block of 'max_volume_gpu' size.
// 		*/
//         // See future c++ async launch
//         std::vector<std::future<void>> threads = {};

//         for (i = 0; i < ngpus; i++){
            
//             sub_block_gpu = min(configs.nslices - ptr_gpu, max_block_gpu);
//             ptr_block_gpu = (size_t)configs.nrays * configs.nangles * ptr_gpu;
            
//             /* Update pointer */
//             ptr_gpu = ptr_gpu + sub_block_gpu;
            
//             threads.push_back( std::async( std::launch::async, _ReconstructionThreads, configs, recon + ptr_block_gpu, data + ptr_block_gpu, flats + ptr_block_gpu, darks + ptr_block_gpu, angles, sub_block_gpu, gpus[i]));		
//         }
    
//         // Log("Synchronizing all threads...\n");
    
//         for (i = 0; i < ngpus; i++)
//             threads[i].get();

// 		cudaDeviceSynchronize();
// 	}
// }