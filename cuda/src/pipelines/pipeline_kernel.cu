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

extern "C"{

    void getReconstructionMethods(CFG configs, WKP *workspace, dim3 tomo_size, dim3 obj_size)
    {
        switch (configs.ReconParam.method){

            case ReconstructionMethod::fbpRT:
                /* FBP */
                getFBP( configs.ReconParam, 
                        workspace->objPadd, 
                        workspace->tomoPadd, 
                        workspace->angles, 
                        tomo_size, obj_size, 
                        configs.geometry.detector_pixel.x, 
                        configs.geometry.detector_pixel.y
                    );
                break;
            case ReconstructionMethod::fbpBST:
                /* BST */
                break;
            case ReconstructionMethod::eEMRT:
                /* EM RT eEM */
                break;
            case ReconstructionMethod::tEMRT:
                /* EM RT tEM */
                break;
            case ReconstructionMethod::tEMFQ:
                /* EM FQ tEM */
                break;
            case ReconstructionMethod::fdk:
                /* FDK */
                break;
            default:
                printf("No reconstruction method selected. Finishing run... \n");
                exit(EXIT_SUCCESS);
                break;
        }
    }

}

extern "C"{
    void _ReconstructionPipeline(CFG configs, WKP *workspace)
    {
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nslices = configs.tomo.size.z;
        int nraysp  = configs.tomo.size.x * ( 1.0f + configs.tomo.pad.x );

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;
        int nxp     = configs.obj.size.x * ( 1.0f + configs.obj.pad.x );
        int nyp     = configs.obj.size.y * ( 1.0f + configs.obj.pad.y );

        if( configs.flags.do_flat_dark_correction )
        {
            getBackgroundCorrection(workspace->tomo, workspace->flat, workspace->dark, configs.tomo.size.z, configs.nflats);
            
            getLog(workspace->tomo, configs.tomo.size.z);

            free(workspace->flat); /* Dealocate variable we will not use anymore */
            free(workspace->dark); /* Dealocate variable we will not use anymore */
        }

        if( configs.flags.do_rings )
        {
            getTitarenkoRings(workspace->tomo, configs.tomo.size.z, configs.RingsParam.rings_lambda, configs.RingsParam.rings_block);        
        }

        /* Padding */
        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock( (int)ceil(  nraysp / TPBX ) + 1,
                            (int)ceil( nangles / TPBY ) + 1,
                            (int)ceil( nslices / TPBZ ) + 1);
        
        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 ObjthreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 ObjgridBlock(  (int)ceil(     nxp / TPBX ) + 1,
                            (int)ceil(     nyp / TPBY ) + 1,
                            (int)ceil( nslices / TPBZ ) + 1);
        
        opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(workspace->tomo, workspace->tomoPadd, configs.tomo.padding_mode, 
            dim3(nrays, nangles, nslices), configs.tomo.pad);

        free(workspace->tomo); /* Dealocate variable we will not use anymore */
        
        /* Reconstruction */
        getReconstructionMethods(configs, workspace, 
                                dim3(nraysp,nangles,nslices),
                                dim3(nxp,nyp,nslices));

        free(workspace->tomoPadd); /* Dealocate variable we will not use anymore */

        /* Recuperate Padding */
        opt::remove_paddR2R<<<ObjgridBlock,ObjthreadsPerBlock>>>(workspace->objPadd, workspace->obj, 
            dim3(nx, ny, nslices), configs.obj.pad);
        
        free(workspace->objPadd); /* Dealocate variable we will not use anymore */

    }
}

extern "C" {

    void ReconstructionPipeline_GPU(CFG configs,
    float *object, float *data, float *flats, float *darks, float *angles, 
    int gpu_device)
    {
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nslices = configs.tomo.size.z;

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;
        int nz      = configs.obj.size.z;

        size_t tomoptr_size = nslices * nangles * nrays;
        size_t objptr_size  =      nz *      ny *    nx;
        size_t flatptr_size =           nangles * nrays;
        size_t darkptr_size =           nangles * nrays;

        /* Initialize GPU device */
        HANDLE_ERROR(cudaSetDevice(gpu_device));

        /* Local GPUs Pointers: allocation */
        WKP *workspace = Initialize_workspace(configs, nslices, nz);

        /* Copy data from host to device */
        opt::CPUToGPU<float>(angles, workspace->angles,      nangles);
        opt::CPUToGPU<float>(  data, workspace->tomo  , tomoptr_size);
        opt::CPUToGPU<float>( flats, workspace->flat  , flatptr_size);
        opt::CPUToGPU<float>( darks, workspace->dark  , darkptr_size);

        /* Enter Reconstruction Pipeline */
        _ReconstructionPipeline(configs, workspace);

        /* Copy Reconstructed data from device to host */
        opt::GPUToCPU<float>(object, workspace->obj, objptr_size);

        free(workspace->obj); /* Dealocate variable we will not use anymore */
        free(workspace);

        // cudaDeviceSynchronize();
    }
}