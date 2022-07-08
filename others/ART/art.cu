// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi

#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C" {
    void MaskedART(int device, float* _recon, float* _sino, int nrays, int nangles, int blocksize, float* _mask, int numiter, float vmin)
    {
        cudaSetDevice(device);

        rImage sintable(nangles,1);
        rImage costable(nangles,1);

        for(int a=0; a<nangles; a++)
        {
            sintable.cpuptr[a] = sinf(float(M_PI)*a/float(nangles));
            costable.cpuptr[a] = cosf(float(M_PI)*a/float(nangles));
        }

        sintable.LoadToGPU();
        costable.LoadToGPU();

        rImage recon(_recon,nrays,nrays,blocksize);
        rImage backward(nrays,nrays,blocksize);

        rImage sino(_sino,nrays,nangles,blocksize);
        rImage forward(nrays,nangles,blocksize);
        rImage mask(_mask,nrays,blocksize,1);
        mask *= -1.0f;

        for(int iter=0; iter<numiter; iter++)
        {
            KRadon_RT<<<dim3(nrays/64,nangles,blocksize),64>>>(forward.gpuptr, recon.gpuptr, nrays, nangles);
            forward -= sino;
            forward *= mask;

            KBackProjection_RT<<<dim3(nrays/64,nrays,blocksize),64>>>(
                (char*)backward.gpuptr, forward.gpuptr, nrays, nrays, nangles, 0, EType::TypeEnum::FLOAT32, 0, sintable.gpuptr, costable.gpuptr);

            recon += backward;
            recon.Clamp(vmin,1E30f);
        }

        recon.CopyTo(_recon);
        cudaDeviceSynchronize();
        std::cout << device << " end." << std::endl;
    }
    void MMaskedART(int* device, int ndevs, float* _recon, float* _sino, int nrays, int nangles, int blocksize, float* _mask, int numiter, float vmin)
    {
        std::vector<std::future<void>> threads;
        for(int t=0; t<ndevs; t++)
        {
            int offset = t*blocksize/ndevs;
            float* recon = _recon + nrays*nrays*offset;
            float* sino = _sino + nrays*nangles*offset;
            float* mask = _mask + nrays*offset;
            threads.push_back(	std::async(std::launch::async,MaskedART, device[t], recon, sino, nrays, nangles, blocksize/ndevs, mask, numiter, vmin) );
        }
        for(int t=0; t<ndevs; t++)
            threads[t].get();
    }

}