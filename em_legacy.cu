// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi

#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"

extern "C" {
void MaskedEM(int device, float* _recon, float* _tomo, float* _flat, int nrays, int nangles, int blocksize, float* _mask, int numiter)
	{
		std::cout << device << " " << _recon << std::endl;
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

		rImage tomo(_tomo,nrays,nangles,blocksize);
		rImage flat(_flat,nrays,nangles,blocksize);
		rImage BI(nrays,nrays,blocksize);
		rImage BP(nrays,nrays,blocksize);
		rImage ffk(nrays,nangles,blocksize);

		rImage mask(_mask,nrays,blocksize,1);

		RingsEM(tomo.gpuptr, flat.gpuptr, nrays, nangles, blocksize);

		tomo *= mask;
		KBackProjection_RT<<<recon.ShapeBlock(),recon.ShapeThread()>>>(
			(char*)BI.gpuptr, tomo.gpuptr, nrays, nrays, nangles, EType::TypeEnum::FLOAT32, 0, sintable.gpuptr, costable.gpuptr);

		for(int iter=0; iter<numiter; iter++)
		{
			KRadon_RT<<<ffk.ShapeBlock(),ffk.ShapeThread()>>>(ffk.gpuptr, recon.gpuptr, nrays, nangles);
			KExponential<<<ffk.ShapeBlock(),ffk.ShapeThread()>>>(ffk, flat);

			ffk *= mask;

			KBackProjection_RT<<<recon.ShapeBlock(),recon.ShapeThread()>>>(
				(char*)BP.gpuptr, ffk.gpuptr, nrays, nrays, nangles, EType::TypeEnum::FLOAT32, 0, sintable.gpuptr, costable.gpuptr);

			KDivide_transmission<<<recon.ShapeBlock(),recon.ShapeThread()>>>(recon, BP, BI);
			recon.Clamp(0,1.0f);
		}

		recon.CopyTo(_recon);
		cudaDeviceSynchronize();
	}
	void MMaskedEM(int* device, int ndevs, float* _recon, float* _tomo, float* _flat, int nrays, int nangles, int blocksize, float* _mask, int numiter)
	{
		std::vector<std::future<void>> threads;
		for(int t=0; t<ndevs; t++)
		{
			int offset = t*blocksize/ndevs;
			float* recon = _recon + nrays*nrays*offset;
			float* tomo = _tomo + nrays*nangles*offset;
			float* flat = _flat + nrays*offset;
			float* mask = _mask + nrays*offset;
			threads.push_back(	std::async(std::launch::async,MaskedEM,device[t], recon, tomo, flat, nrays, nangles, blocksize/ndevs, mask, numiter) );
		}
		for(int t=0; t<ndevs; t++)
			threads[t].get();
	}
}