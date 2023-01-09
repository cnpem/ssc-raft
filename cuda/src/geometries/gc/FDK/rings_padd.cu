#include "../../../../inc/include.h"
#include "../../../../inc/gp/rings.h"
#include "../../../../inc/common/kernel_operators.hpp"
#include "../../../../inc/common/complex.hpp"
#include "../../../../inc/common/types.hpp"
#include "../../../../inc/common/operations.hpp"
#include "../../../../inc/common/logerror.hpp"

void ringsgpu_fdk(int gpu, float* data, int nrays, int nangles, int nslices, float lambda_rings, int ringblocks)
{   
    cudaSetDevice(gpu);
    size_t blocksize = min((size_t)nslices,32ul);


    for(size_t bz=0; bz<nslices; bz+=blocksize){
        blocksize = min(blocksize,size_t(nslices)-bz);

        // HANDLE_ERROR( cudaMemcpy(tomogram, data + bz*nrays*nangles, sizeof(float) * nrays * nangles * blocksize, cudaMemcpyHostToDevice) );	
        
        for (int m = 0; m < ringblocks / 2; m++){
            // Rings(tomogram, nrays, nangles, blocksize, lambda_rings, nrays*nangles);
            Rings(data, nrays, nangles, blocksize, lambda_rings, nrays*nangles);
            size_t offset = nrays*nangles;
            size_t step = (nangles / ringblocks) * nrays;
            float* tomptr = data;
            // float* tomptr = tomogram;

            for (int n = 0; n < ringblocks - 1; n++){
                Rings(data, nrays, nangles, blocksize, lambda_rings, nrays*nangles);
                // Rings(tomogram, nrays, nangles, blocksize, lambda_rings, nrays*nangles);

                tomptr += step;
            }
            Rings(tomptr, nrays, nangles%ringblocks + nangles/ringblocks, blocksize, lambda_rings, offset);
        }
    
    }

}



