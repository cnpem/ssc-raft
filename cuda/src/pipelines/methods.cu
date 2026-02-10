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
#include "processing/rings_all_stripes.cuh"

#include <thread>

extern "C"{

    void getReconstructionMethods(CFG configs, WKP *workspace, int tomoblock, int objblock, cudaStream_t nstream)
    {
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nraysp  = PDIM(configs.tomo.size.x,configs.tomo.pad.x); 

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;
        int nxp     = PDIM(configs.obj.size.x,configs.obj.pad.x); 
        int nyp     = PDIM(configs.obj.size.y,configs.obj.pad.y);

        WBST *bst_workspace; int bst_padd, blocksize_bst;

        /* Padding */
        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock = opt::setGridBlock(dim3(nraysp,nangles,tomoblock), TomothreadsPerBlock);
        
        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 ObjthreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 ObjgridBlock = opt::setGridBlock(dim3(nxp,nyp,objblock), ObjthreadsPerBlock);
        
        opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(workspace->tomo, 
                                                            workspace->tomoPadd, 
                                                            configs.tomo.padding_mode, 
                                                            dim3(nrays, nangles, tomoblock), 
                                                            configs.tomo.pad);
        switch (configs.ReconParam.method)
        {
            case static_cast<int>(ReconstructionMethod::none):
                /* No reconstruction done */
            break;
            case static_cast<int>(ReconstructionMethod::fbp):
                /* FBP */
                getFBP( configs.ReconParam, 
                        configs.tomo, 
                        configs.obj,
                        workspace->objPadd, 
                        workspace->tomoPadd, 
                        workspace->angles, 
                        tomoblock, 
                        configs.geometry.detector_pixel.x
                    );
                // printf("FBP After \n");
                // fflush(stdout);
            break;
            case static_cast<int>(ReconstructionMethod::bst):
                /* BST */
                bst_padd = 2; blocksize_bst = 1;
                
                bst_workspace = InitializeBST_workspace(dim3(nraysp,nangles,tomoblock), 
                                                        dim3(   nxp,    nxp, objblock), 
                                                        bst_padd, blocksize_bst);

                getBST( workspace->objPadd, workspace->tomoPadd, workspace->angles, 
                        nraysp, nangles, tomoblock, nxp, bst_padd, 
                        configs.ReconParam.filter_reg, configs.ReconParam.paganin_slices, 
                        configs.ReconParam.filter, configs.ReconParam.rotation_axis_offset, 
                        configs.geometry.detector_pixel.x, bst_workspace, nstream);

                freeBSTWorkspace(bst_workspace);
            break;
            case static_cast<int>(ReconstructionMethod::eEMRT):
                /* EM RT eEM */
            break;
            case static_cast<int>(ReconstructionMethod::tEMRT):
                /* EM RT tEM */
            break;
            case static_cast<int>(ReconstructionMethod::tEMFQ):
                /* EM FQ tEM */
            break;
            case static_cast<int>(ReconstructionMethod::fdk):
                /* FDK */
            break;
            default:
                printf("No reconstruction method selected. Finishing run... \n");
            break;
        }
        
        /* Recuperate Padding for reconstruction */
        opt::remove_paddR2R<<<ObjgridBlock,ObjthreadsPerBlock>>>(workspace->objPadd, 
                                                                 workspace->obj, 
                                                                 dim3(nx, ny, objblock),
                                                                 configs.obj.pad);

        // printf("rm pad obj After \n");
        // fflush(stdout);   

        /* Recuperate Padding for tomogram */
        opt::remove_paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(workspace->tomoPadd, 
                                                                   workspace->tomo, 
                                                                   dim3(nrays, nangles, tomoblock),
                                                                   configs.tomo.pad);
        // printf("rm pad tomo After \n");
        // fflush(stdout);   
    }

    void getRingsMethods(CFG configs, WKP *workspace, int tomoblock, cudaStream_t nstream)
    {
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nflats  = configs.flat.size.z;

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;

        // printf("Rings method: %d \n", configs.RingsParam.method);
        // printf("Rings method none: %d \n", static_cast<int>(RingsMethod::none));
        // fflush(stdout);

        switch (configs.RingsParam.method)
        {
            case static_cast<int>(RingsMethod::none):
                /* No rings done */
                printf("No rings method selected.\n");
            break;
            case static_cast<int>(RingsMethod::allstripes):
                /* All Stripes */
                get_remove_all_stripes(workspace->tomo, 
                                       configs.RingsParam.all_stripes, 
                                       tomoblock, nangles, nrays);
            break;
            case static_cast<int>(RingsMethod::titarenko):
                /* Titarenko */
                getTitarenkoRings(  workspace->tomo,
                                    dim3(nrays,nangles,tomoblock), 
                                    configs.RingsParam.rings_lambda, 
                                    configs.RingsParam.rings_block,
                                    nstream);
            break;
            default:
                printf("No rings method selected.\n");
            break;
        }
    }

    void getAlignMethods(CFG configs, WKP *workspace, int tomoblock, cudaStream_t nstream)
    {
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nflats  = configs.flat.size.z;

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;

        // printf("Rings method: %d \n", configs.RingsParam.method);
        // printf("Rings method none: %d \n", static_cast<int>(RingsMethod::none));
        // fflush(stdout);

        switch (configs.AlignParam.method)
        {
            case static_cast<int>(AlignMethod::none):
                /* No Alignment done */
                printf("No Alignment method selected.\n");
            break;
            case static_cast<int>(AlignMethod::rotaxisM):
                printf("Rotation axis correction during reconstruction method.\n");
            break;
            case static_cast<int>(AlignMethod::rotaxis):
                /* Centersino Offset */
                getRotAxisCorrection(workspace->tomo, 
                                     configs.AlignParam.rotation_axis_offset, 
                                     configs.tomo.size);
            break;
            case static_cast<int>(AlignMethod::excentric):
                /* Excentric Offset */
                // printf("Excentric Tomo Stitching offset: %d\n",configs.AlignParam.excentric_offset);
                // fflush(stdout);
                getExcentricTomo(workspace->tomo, nrays, nangles, tomoblock, configs.AlignParam.excentric_offset);
    
                /* New shape of tomogram after excentric stitching */
                opt::set_excentric_tomo_dimensions(configs.tomo.size);
            break;
            default:
                printf("No Alignment method selected.\n");
            break;
        }
    }

    void getBackgroundCorrectionMethods(CFG configs, WKP *workspace, int tomoblock, cudaStream_t nstream)
    {
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nflats  = configs.flat.size.z;

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;

        switch (configs.back_corr_method)
        {
            case static_cast<int>(BackCorrectionMethod::none):
                /* No Background Correction done */
                printf("No Background Correction (flat/dark correction) method selected.\n");
            break;
            case static_cast<int>(BackCorrectionMethod::classic):
                /* Classic Flat and Dark correction */
                getBackgroundCorrection_slices( workspace->tomo,
                                                workspace->flat, 
                                                workspace->dark, 
                                                dim3(nrays,nangles,tomoblock), 
                                                nflats, 
                                                1);
            break;
            case static_cast<int>(BackCorrectionMethod::classicNoLog):
                /* Classic Flat and Dark correction */
                getBackgroundCorrection_slices( workspace->tomo,
                                                workspace->flat, 
                                                workspace->dark, 
                                                dim3(nrays,nangles,tomoblock), 
                                                nflats, 
                                                0);
            break;
            default:
                printf("No Background Correction (flat/dark correction) method selected.\n");
            break;
        }
    }

}

