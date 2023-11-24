#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{
    void ReconstructionPipeline(float *recon, float *data, float *flats, float *darks, float *angles,
	float *parameters_float, int *parameters_int,
	int *gpus, int ngpus)
	{

        int i, Maxgpu;
        int process_index, total_number_of_processes;

		/* Multiples devices */
		cudaGetDeviceCount(&Maxgpu);

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpu && "Invalid device number.");

        CFG configs;

        /* Set total number of processes to be sent to the GPUs */
        total_number_of_processes = getTotalProcesses(configs, ngpus);
                
        /* Set processes pipeline for different geometries */
        Process *process = (Process *)malloc(sizeof(Process) * total_number_of_processes);

        switch (configs.geometry){
            case 0:
                /* Parallel */
                for (process_index = 0; process_index < total_number_of_processes; process_index++)
                    setProcessParallel(configs, process, process_index, total_number_of_processes, gpus, ngpus);
                break;
            case 1:
                /* Conebeam */
                for (process_index = 0; process_index < total_number_of_processes; process_index++)
                    setProcessConical(configs, process, process_index, total_number_of_processes, gpus, ngpus);
                break;
            case 2:
                /* Fanbeam */
                printf("No pipeline for Fanbeam yet.");
                break;
            default:
                printf("Nope.");
                break;
        }	

        clock_t b_begin = clock();

        _setReconstructionPipeline(configs, process, data, flats, darks, angles, total_number_of_processes, ngpus);

        clock_t b_end = clock();
        double time = double(b_end - b_begin) / CLOCKS_PER_SEC;

        cudaDeviceSynchronize();
        printf(cudaGetErrorString(cudaGetLastError()));
        printf("\n");

        free(process);
    }
}


extern "C"{
    void _setReconstructionPipeline(CFG configs, Process process,
    float *recon, float *data, 
    float *flats, float *darks, float *angles,
    int total_number_of_processes, int ngpus)
    {
        int process_index, i;

        std::vector<thread> threads_back;
        
        // float *c_proj[ndevs];
        // float *c_recon[ndevs];
        // float *c_beta[ndevs];

        // GPUs Pointers: declaration and allocation
        WKP *workspace = (WKP *)malloc(sizeof(WKP) * total_number_of_processes); 

        while (process_index < total_number_of_processes){

            threads_back.emplace_back( thread(_ReconstructionProcessesPipeline, configs, workspace[process_index], data, flats, darks, angles, process[process_index]) );		

            process_index = process_index + 1;

            for (i = 0; i < total_number_of_processes; i++)
                threads_back[i].join();

            threads_back.clear();
        
        }
        
    }
}


extern "C" {

	void _ReconstructionProcessesPipeline(CFG configs, WKP *workspace, float *recon, float *frames, float *flats, float *darks, float *angles, Process process)
	{	
		/* Initialize GPU device */
		HANDLE_ERROR(cudaSetDevice(process.index_gpu));
		
        /* Local GPUs Pointers: allocation */
        workspace = allocateWorkspace(configs, process);

        /* Copy data from host to device */
        HANDLE_ERROR(cudaMemcpy(workspace->angles, angles, configs.nangles * sizeof(float), cudaMemcpyHostToDevice));

        HANDLE_ERROR(cudaMemcpy(workspace->tomo, &frames[process.ind_tomo], process.n_tomo * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(workspace->flat,  &flats[process.ind_tomo], process.n_tomo * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(workspace->dark,  &darks[process.ind_tomo], process.n_tomo * sizeof(float), cudaMemcpyHostToDevice));

        /* Enter Reconstruction Pipeline */
        _ReconstructionPipeline(configs, workspace, process);

        /* Copy Reconstructed data from device to host */
        HANDLE_ERROR(cudaMemcpy(&recon[process.ind_recon], workspace->recon, process.n_recon * sizeof(float), cudaMemcpyDeviceToHost));

		cudaDeviceSynchronize();
	}
}


extern "C"{
    void _Reconstruction(CFG configs, WKP *workspace, int max_block_size, int ngpu)
    {
		
        if( configs.do_flat_dark_correction == 1)
            getFlatDarkCorrection(configs, workspace->tomo, workspace->flat, workspace->dark, configs.nrays, configs.nangles, max_block_size, configs.numflats);

        // if( configs.do_phase_filter == 1) 
        //     getPaganinFilter();

        if( configs.do_rings == 1) 
            getRings(workspace->tomo, configs.nrays, configs.nangles, max_block_size, configs.rings_lambda, configs.rings_block);
        
        // if( configs.do_rotation_offset == 1) 
        //     getRotAxisOfssetCorrection();

        // if( configs.do_alignment == 1) 
        //     getTomogramAlignment();

        switch (configs.geometry){
            case 0:
                /* Parallel */
                getReconstructionParallel(configs, workspace->tomo);
                break;
            case 1:
                /* Conebeam */
                getReconstructionConebeam(configs, workspace->tomo);
                break;
            case 2:
                /* Fanbeam */
                printf("No pipeline for Fanbeam yet.");
                // getReconstructionFanbeam(configs, workspace->tomo);
                break;
            default:
                printf("Nope.");
                break;
        }	
        

    }
}

extern "C"{

    void getReconstructionParallel(CFG configs, float *tomogram){

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
                printf("Nope.");
                break;
        }	
    }

    void getReconstructionConebeam(CFG configs, float *tomogram){

        switch (configs.reconstruction_method){
            case 0:
                /* FBP */
                
                break;
            case 1:
                /* BST */
                
                break;
            case 2:
                /* EM RT eEM */
                
                break;
            case 3:
                /* EM RT tEM */
                
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
                printf("Nope.");
                break;
        }	
    }

    // void getReconstructionFanbeam(CFG configs, float *tomogram){

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
    //             printf("Nope.");
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