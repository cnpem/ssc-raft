#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{
    void _getReconstructionPipeline()
    {
        size_t i, j, k;

        cudaSetDevice(ngpu);

		// GPUs Pointers: declaration and allocation
        WKP *workspace = allocateWorkspace(configs, slicesize);

        /* Copy data from host to device */
        HANDLE_ERROR(cudaMemcpy(workspace->tomo, frames, configs.nangles * configs.nrays * slicesize * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(workspace->flat, flats , configs.nangles * configs.nrays * slicesize * sizeof(float), cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(workspace->dark, darks , configs.nangles * configs.nrays * slicesize * sizeof(float), cudaMemcpyHostToDevice));

        if( configs.iscorrection == 1){
            getFlatDarkCorrection(CFG configs, workspace->tomo, workspace->flat, workspace->dark, configs.nrays, configs.nangles, slicesize, configs.numflats);
        }

        // if( configs.isphasefilter == 1) 
        //     getPaganinFilter();

        if( configs.isrings == 1) 
            getRings(workspace->tomo, nrays, nangles, slicesize, configs.lambdarings, configs.ringblocks);
        
        if( configs.isrotoffset == 1) 
            getRotAxisOfssetCorrection();


        _getRecon();

        /* Copy answer to CPU */
        HANDLE_ERROR(cudaMemcpy(recon, workspace->recon, configs.nx * configs.ny * slicesize * sizeof(float), cudaMemcpyDeviceToHost));


    }
}
extern "C"{

    void _getRecon(){

        switch ((int)configs.recon_method){
            case 0:
                /* FDK */
                printf("No filter was selected!");
                break;
            case 1:
                /* EM Conical */
                
                break;
            case 2:
                /* FBP */
                
                break;
            case 3:
                /* BST */
                
                break;
            case 4:
                /* EM RT */
                
                break;
            case 5:
                /* EM FST */
                
                break;

            default:
                printf("Nope.");
                break;
        }	
    }
}

extern "C"{
	void getReconstruction(float *projections, float _Complex *recovery, float _Complex *initialGuessN, 
	float *compact, float *mask, float *parameters, size_t *iterations, size_t *volumesize, 
	int *devices, int ndev)
	{	
		int i, Maxgpu;

		/* Multiples devices */
		cudaGetDeviceCount(&Maxgpu);

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpu && "Invalid device number.");

        /* Compute memory for maximum blocksize */ 
        //size_t memframe = 8*reconsize*(reconsize+truenangles);
        size_t memframe = 10*nrays*nangles;
        size_t maxusage = 1ul<<33;
        size_t maxblock = min(max(maxusage/memframe,size_t(1)),32ul);

        /* Transpose the data from (nangles, nslices, nrays) to (nslices,nangles,nrays) */ 
		getTransposedData(data , configs);
        getTransposedFlat(flats, configs);
        getTransposedDark(darks, configs);

		param.subvolume = (param.Nz + ndev - 1) / ndev;

		/* Launch async Threads for each device.
			Each device solves a block of 'param.frame' size.
		*/
        // See future c++ async launch
        std::vector<std::future<void>> threads = {};

        for (i = 0; i < ndev; i++){
            
            if(param.subvolume*(i+1) > param.Nz)
                param.subvolume = param.Nz - param.subvolume*i;

            if(param.subvolume < 1)
                continue;
            
            threads.push_back( std::async( std::launch::async, _getReconstructionThreads, param, &prof, projections + param.frame*param.subvolume*i, recovery + param.frame*param.subvolume*i, initialGuessN + param.frame*param.subvolume*i, compact, mask, devices[i]));		
        }
    
        // Log("Synchronizing all threads...\n");
    
        for (i = 0; i < ndev; i++)
            threads[i].get();

		cudaDeviceSynchronize();
	}
}

extern "C" {

	void _getReconstructionThreads(PAR param, PROF *prof, float *projections, float _Complex *recovery, float _Complex *initialGuessN, float *compact, float *mask, int ndev)
	{	
		/* Initialize GPU device */
		HANDLE_ERROR(cudaSetDevice(ndev))
		
		int bz; 

		/* Plan for Fourier transform - cufft */
		int n[] = {(int)param.Npadx,(int)param.Npady};
		HANDLE_FFTERROR(cufftPlanMany(&param.mplan, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, param.blocksize));

		param.zblock = param.blocksize;

		/* Loop for each batch of size 'batch' in threads */

		for (bz = 0; bz < param.subvolume; bz+=param.blocksize){

			if( (bz + param.blocksize) > param.subvolume){
				
				param.zblock = param.subvolume - bz;
				
				if(param.zblock < 1)
					continue;

				HANDLE_FFTERROR(cufftDestroy(param.mplan));

				HANDLE_FFTERROR(cufftPlanMany(&param.mplan, 2, n, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, param.zblock));

				prainGPU(param, prof, projections + param.frame*bz, recovery + param.frame*bz, initialGuessN + param.frame*bz, compact, mask, ndev);
				
			}else{
				prainGPU(param, prof, projections + param.frame*bz, recovery + param.frame*bz, initialGuessN + param.frame*bz, compact, mask, ndev);
			}
		}

		/* Destroy plan */
		HANDLE_FFTERROR(cufftDestroy(param.mplan));

		cudaDeviceSynchronize();
	}
}