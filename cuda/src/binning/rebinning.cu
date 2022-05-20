/* 
@file rebinning.cu
@Paola Cunha Ferraz (paola.ferraz@lnls.br)
@brief RAFT: Reconstruction Algorithms for Tomography
@version 0.2
@date 2022-04-23

Cone-beam Rebinning to Parallel beam 3D
*/

#include "../inc/include.h"
#include "../inc/common/complex.hpp"
#include "../inc/common/types.hpp"
#include "../inc/common/operations.hpp"
#include "../inc/common/logerror.hpp"

extern "C"{
    void GPUrebinning(float *conetomo, float *tomo, float *parameters, size_t *volumesize, int *gpus, int ngpus)
    {	/* Extern C function to GPU Conebeam Rebinning */
        int i, Maxgpu;
        
        /* Variables for time profile */
        cudaEvent_t start, stop;
        float milliseconds;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        /* Multiples devices */
        cudaGetDeviceCount(&Maxgpu);

        /* If devices input are larger than actual devices on GPU, exit */
        for(i = 0; i < ngpus; i++) 
            assert(gpus[i] < Maxgpu && "Invalid device number.");

        /* Start time */
        cudaEventRecord(start);
        
        /* Struct to parameters (PAR param) and Profiling (PROF prof - not finished) */
        PAR param; PROF prof;

        /* Set all parameters necessary to Rebinning */
        set_rebinning_parameters_gpu(&param, parameters, volumesize, gpus);
        
        /* Call Function to GPU Rebinning Kernel */
        _GPUrebinning(param, &prof, conetomo, tomo, param.gpus[0]);

        /* Record Total time*/
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        cudaDeviceSynchronize();
    }
}

extern "C"{
    void CPUrebinning(float *conetomo, float *tomo, float *parameters, size_t *volumesize)
    {	/* Extern C function to CPU Conebeam Rebinning */
        
        /* Struct to parameters (PAR param) and Profiling (PROF prof - not finished) */
        PAR param; PROF prof;

        /* Set all parameters necessary to Rebinning */
        set_rebinning_parameters_cpu(&param, parameters, volumesize);

        /* Call Function to CPU Rebinning */
        cone_rebinning_cpu(tomo, conetomo, param);

    }
}

void _GPUrebinning(PAR param, PROF *prof, float *conetomo, float *tomo, int ngpus)
{	
    /* Initialize GPU device */
    HANDLE_ERROR(cudaSetDevice(ngpus))
    
    size_t sizex = param.Nx; size_t sizey = param.Ny; size_t sizez = param.Nz;

    /* Alloc Rebinning Struct with gpu pointers*/
    RDAT workspace; RDAT *reb = allocate_gpu_rebinning(&workspace, param);

    /* Copy data from host to device */
	HANDLE_ERROR(cudaMemcpy(reb->dctomo, conetomo, sizex * sizey * sizez * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(reb->dtomo , tomo    , sizex * sizey * sizez * sizeof(float), cudaMemcpyHostToDevice));

    /* Call Conebeam Rebinning */
    cone_rebinning_kernel<<<param.Grd,param.BT>>>(reb->dtomo, reb->dctomo, param);

    /* Copy data from device to host */
    HANDLE_ERROR(cudaMemcpy(tomo, reb->dtomo, sizex * sizey * sizez * sizeof(float), cudaMemcpyDeviceToHost));

    /* Free rebinning struct: free gpu pointers */    
    free_gpu_rebinning(reb);

    cudaDeviceSynchronize();
}
