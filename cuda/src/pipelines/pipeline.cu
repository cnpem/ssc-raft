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

    void getReconstructionMethods(CFG configs, WKP *workspace, int nblocks)
    {
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nraysp  = PDIM(configs.tomo.size.x,configs.tomo.pad.x); 

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;
        int nxp     = PDIM(configs.obj.size.x,configs.obj.pad.x); 
        int nyp     = PDIM(configs.obj.size.y,configs.obj.pad.y);

        /* Padding */
        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock( (int)ceil(  nraysp / TPBX ) + 1,
                            (int)ceil( nangles / TPBY ) + 1,
                            (int)ceil( nblocks / TPBZ ) + 1);
        
        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 ObjthreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 ObjgridBlock(  (int)ceil(     nxp / TPBX ) + 1,
                            (int)ceil(     nyp / TPBY ) + 1,
                            (int)ceil( nblocks / TPBZ ) + 1);
        
        opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(workspace->tomo, 
                                                            workspace->tomoPadd, 
                                                            configs.tomo.padding_mode, 
                                                            dim3(nrays, nangles, nblocks), 
                                                            configs.tomo.pad);
        switch (configs.ReconParam.method)
        {
            case ReconstructionMethod::none:
                /* No reconstruction done */
            break;
            case ReconstructionMethod::fbpRT:
                /* FBP */
                getFBP( configs.ReconParam, 
                        workspace->objPadd, 
                        workspace->tomoPadd, 
                        workspace->angles, 
                        dim3(nraysp,nangles,nblocks), 
                        dim3(nxp,nyp,nblocks), 
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
        /* Recuperate Padding for reconstruction */
        opt::remove_paddR2R<<<ObjgridBlock,ObjthreadsPerBlock>>>(workspace->objPadd, 
                                                                 workspace->obj, 
                                                                 dim3(nx, ny, nblocks),
                                                                 configs.obj.pad);
        /* Recuperate Padding for tomogram */
        opt::remove_paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(workspace->tomoPadd, 
                                                                   workspace->tomo, 
                                                                   dim3(nrays, nangles, nblocks),
                                                                   configs.tomo.pad);
    }

}

extern "C"{
    void ReconstructionPipeline(CFG configs, WKP *workspace, int blocksize, size_t ptr)
    {
        cudaStream_t nstream = 0;
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nraysp  = PDIM(configs.tomo.size.x,configs.tomo.pad.x); 

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;
        int nxp     = PDIM(configs.obj.size.x,configs.obj.pad.x); 
        int nyp     = PDIM(configs.obj.size.y,configs.obj.pad.y);

        printf("tomo shape:\n");
        printDim(configs.tomo.size);
        printf("tomo pad shape:\n");
        printDim(configs.tomo.pad);
        printf("obj shape:\n");
        printDim(configs.obj.size);
        printf("obj pad shape:\n");
        printDim(configs.obj.pad);
        printf("configs.nflats: %d\n",configs.nflats);
        printf("nraysp: %d\n",nraysp);
        printf("nxp: %d\n",nxp);
        printf("nyp: %d\n",nyp);

        printf("configs.flags.do_flat_dark_correction: %d\n",configs.flags.do_flat_dark_correction);
        printf("configs.flags.do_flat_dark_log: %d\n",configs.flags.do_flat_dark_log);

        printf("configs.flags.do_rings: %d\n",configs.flags.do_rings);

        printf("configs.flags.do_reconstruction: %d\n",configs.flags.do_reconstruction);
        fflush(stdout);

        if( configs.flags.do_flat_dark_correction == 1 )
        {
            printf("Background Correction\n");
            fflush(stdout);
            getBackgroundCorrection_slices( workspace->tomo,
                                            workspace->flat + (size_t)ptr * nrays * configs.nflats, 
                                            workspace->dark + (size_t)ptr * nrays, 
                                            dim3(nrays,nangles,blocksize), 
                                            configs.nflats, 
                                            configs.flags.do_flat_dark_log);
        }

        if( configs.flags.do_rings == 1 )
        {
            printf("Rings\n");
            fflush(stdout);
            getTitarenkoRings(  workspace->tomo,
                                dim3(nrays,nangles,blocksize), 
                                configs.RingsParam.rings_lambda, 
                                configs.RingsParam.rings_block,
                                nstream);        
        }

        if( configs.flags.do_reconstruction == 1)
        {
            /* Reconstruction */
            printf("Reconstruction\n");
            fflush(stdout);
            getReconstructionMethods(configs, workspace, blocksize);
        }
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

        int i;
        int blocksize = configs.tomo.blocksize;
        int ptr = 0;
        int subblock; 

        /* Compute total memory used on a singles slice */
        size_t total_required_mem_per_slice_bytes = (
            calcSliceMemoryBytes(configs.tomo)           + // Tomo slice
            calcSliceMemoryBytes(configs.obj)            + // Reconstructed object slice
            calcPaddedSliceMemoryBytes(configs.obj)      + // Reconstructed padded object slice
            2 * calcPaddedSliceMemoryBytes(configs.tomo) + // Tomo padded slice
            configs.tomo.size.y * sizeof(float)                 // angles
            );

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, 
                                                       total_required_mem_per_slice_bytes, 
                                                       true, 
                                                       BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(sizez, blocksize_aux);
            blocksize          = min(   32,     blocksize);
        }
        int ind_block = (int)ceil( (float) sizez / blocksize );
        
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nslices = configs.tomo.size.z;

        int nx      = configs.obj.size.x;
        int ny      = configs.obj.size.y;

        /* Local GPUs Pointers: allocation */
        WKP *workspace = Initialize_workspace(  dim3(nrays,nangles,blocksize),
                                                dim3(nx,ny,blocksize),
                                                dim3(nrays,sizez,configs.nflats),
                                                dim3(nrays,sizez,1),
                                                configs.tomo.pad,
                                                configs.obj.pad);

        opt::CPUToGPU<float>( flats,   workspace->flat, (size_t)nrays * sizez * configs.nflats);
        opt::CPUToGPU<float>( darks,   workspace->dark, (size_t)nrays * sizez                 );
        opt::CPUToGPU<float>(angles, workspace->angles,                                nangles);

        printf("blocksize: %d\n",blocksize);
        printf("configs.tomo.blocksize: %d\n",configs.tomo.blocksize);
        printf("ind_block: %d\n",ind_block);
        printf("sizez: %d\n",sizez);
        fflush(stdout);
       
        /* Centersino computation */
        for (i = 0; i < ind_block; i++){

            subblock = min(sizez - ptr, blocksize);

            /* Copy data from host to device */
            
            opt::CPUToGPU<float>( data + (size_t)ptr * nrays * nangles, workspace->tomo, (size_t)nrays * nangles * subblock);

            /* Enter Reconstruction Pipeline */
            ReconstructionPipeline(configs, workspace, subblock, ptr);

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
        printf("configs.nflats: %d\n",configs.nflats);
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
				
				subblock = min(nslices - ptr, subvolume);

				threads.push_back(  std::async( std::launch::async, 
                                    ReconstructionPipeline_GPU, 
                                    configs, 
                                    object + (size_t)ptr *    nx *      ny,
                                    data   + (size_t)ptr * nrays * nangles, 
                                    flats  + (size_t)ptr * nrays * configs.nflats, 
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


