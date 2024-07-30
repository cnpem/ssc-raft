#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <cstdio>
#include "common/configs.hpp"
#include "common/opt.hpp"
#include "pipelines/pipeline.hpp"
#include "geometries/parallel/em.hpp"
#include "geometries/parallel/fbp.hpp"
#include "geometries/parallel/bst.hpp"
#include "processing/processing.hpp"

using std::thread;

extern "C"{
    void ReconstructionPipeline(float *obj, float *data,
            float *flats, float *darks, float *angles,
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

        setGPUParameters(&gpu_parameters, configs.tomo.padsize, ngpus, gpus);

        /* Set total number of processes to be sent to the GPUs */
        total_number_of_processes = getTotalProcesses(configs, BYTES_TO_GB * getTotalDeviceMemory(), configs.tomo.size.z, true);

        /* Set processes pipeline for different geometries */
        Process *process = setProcesses(configs, gpu_parameters, total_number_of_processes);

        // clock_t b_begin = clock();

        _setReconstructionPipeline(&configs, process, gpu_parameters,
                obj, data, flats, darks, angles,
                total_number_of_processes);

        HANDLE_ERROR(cudaGetLastError());

        /* Free process (array of structs) */
        free(process);
    }
}


extern "C"{
    void _setReconstructionPipeline(CFG *configs, Process *process, GPU gpus,
            float *obj, float *data, float *flats, float *darks,
            float *angles, int total_number_of_processes)
    {

        configs->tomo.batchsize    = dim3(   configs->tomo.size.x, configs->tomo.size.y, process->tomobatch_size);
        configs->tomo.padbatchsize = dim3(configs->tomo.padsize.x, configs->tomo.padsize.y, process->tomobatch_size);
        configs->obj.batchsize     = dim3(    configs->obj.size.x,  configs->obj.size.y,  process->objbatch_size);

        std::vector<thread> threads_pipeline;

        for (int p = 0; p < total_number_of_processes; ++p) {

            threads_pipeline.emplace_back(  thread(
                        _ReconstructionProcessPipeline,
                        (*configs), process[p],
                        gpus, obj, data, flats, darks,
                        angles
                        ));

            if (p % gpus.ngpus == gpus.ngpus - 1) {
                for (int g = 0; g < gpus.ngpus; ++g) {
                    threads_pipeline[g].join();
                    cudaSetDevice(g);
                    cudaDeviceSynchronize();
                }
                threads_pipeline.clear();
            }


        }
    }
}


extern "C" {

    void _ReconstructionProcessPipeline(CFG configs, Process process, GPU gpus,
            float *obj, float *data, float *flats, float *darks, float *angles)
    {

        /* Initialize GPU device */
        HANDLE_ERROR(cudaSetDevice(process.index_gpu));

        /* Local GPUs Pointers: allocation */
        WKP *workspace = allocateWorkspace(configs, process.tomobatch_size, process.objbatch_size);

        /* Copy data from host to device */
        HANDLE_ERROR(cudaMemcpy(workspace->angles, angles, configs.tomo.size.y * sizeof(float), cudaMemcpyHostToDevice));


        HANDLE_ERROR(cudaMemcpy(workspace->tomo,  data + process.tomoptr_index,
                    process.tomoptr_size * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(workspace->flat, flats + process.tomo_index_z * configs.tomo.size.x,
                    process.tomobatch_size * configs.tomo.size.x * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(workspace->dark, darks + process.tomo_index_z * configs.tomo.size.x,
                    process.tomobatch_size * configs.tomo.size.x * sizeof(float), cudaMemcpyHostToDevice));

        /* Enter Reconstruction Pipeline */
        _ReconstructionPipeline(configs, workspace, gpus);

        /* Copy Reconstructed data from device to host */
        HANDLE_ERROR(cudaMemcpy(&obj[process.objptr_index], workspace->obj, process.objptr_size * sizeof(float), cudaMemcpyDeviceToHost));

        freeWorkspace(workspace, configs);

        // cudaDeviceSynchronize();
    }
}


extern "C"{
    void _ReconstructionPipeline(CFG configs, WKP *workspace, GPU gpus)
    {
        if( configs.flags.do_flat_dark_correction )
            getBackgroundCorrection(gpus, workspace->tomo, workspace->flat, workspace->dark,
                    configs.tomo.batchsize, configs.numflats);

        if( configs.flags.do_flat_dark_log )
            getLog(workspace->tomo, configs.tomo.batchsize);

        printf("Do rings with: lambda: %f rings_block: %d\n",
                configs.rings_lambda, configs.rings_block);

        if( configs.flags.do_rings )
            getTitarenkoRings(gpus, workspace->tomo,
                    configs.tomo.batchsize, configs.rings_lambda,
                    configs.rings_block);


        printf("Do rotation? %d\n", configs.flags.do_rotation);

        if ( configs.flags.do_rotation) {
            printf("do rotation_auto_offset: %d\n", configs.flags.do_rotation_auto_offset);
            const int rotation_axis_offset = configs.flags.do_rotation_auto_offset ?
                 getCentersino(workspace->tomo, workspace->tomo,
                         workspace->dark, workspace->flat,
                         configs.tomo.size.x, configs.tomo.size.y) :
                 configs.rotation_axis_offset;
            printf("deviation: %d\n", rotation_axis_offset);
            getCorrectRotationAxis(workspace->tomo, workspace->tomo,
                    configs.tomo.batchsize, rotation_axis_offset);
        }

        getReconstructionMethods(configs, gpus, workspace);
    }
}

extern "C"{


    void getReconstructionMethods(CFG configs, GPU gpus, WKP *workspace)
    {
        switch (configs.reconstruction_method){
            case 0:
                /* FBP */
                getFBP( configs, gpus,
                        workspace->obj,
                        workspace->tomo,
                        workspace->angles,
                        configs.tomo.batchsize,
                        configs.tomo.padbatchsize,
                        configs.obj.batchsize);
                break;
            case 1:
                /* BST */
                getBST( configs, gpus,
                        workspace->obj,
                        workspace->tomo,
                        workspace->angles,
                        configs.tomo.batchsize,
                        configs.tomo.padbatchsize,
                        configs.obj.batchsize);
                break;
            case 2:
                /* EM RT eEM */
                get_eEM_RT( configs, gpus,
                        workspace->obj,
                        workspace->tomo,
                        workspace->angles,
                        configs.tomo.batchsize.z);
                break;
            case 3:
                /* EM RT tEM */
                // get_tEM_RT( configs, gpus,
                //             workspace->obj,
                //             workspace->tomo,
                //             workspace->flat,
                //             workspace->angles,
                //             process.tomobatch_size);
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
            case 9:
                /* FDK */
                break;
            case 10:
                /* EM Conical eEM*/
                break;
            case 11:
                /* EM Conical tEM*/
                break;
            case 12:
                /* EM Conical eEM TV*/
                break;
            case 13:
                /* EM Conical tEM TV*/
                break;
            default:
                printf("No reconstruction method selected. Finshing run... \n");
                exit(EXIT_SUCCESS);
                break;
        }
    }

}

