// Authors: Giovanni Baraldi, Eduardo X. Miqueles

#include "geometries/parallel/radon.hpp"
#include "geometries/parallel/bst.hpp"
#include "processing/filters.hpp"
#include "common/complex.hpp"
#include "common/types.hpp"
#include "common/opt.hpp"
#include "common/logerror.hpp"


__global__ void sino2p(complex* padded, float* in, 
size_t nrays, size_t nangles, int pad0, int csino)
{
	int center = nrays/2 - csino;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(idx < nrays/2)
	{
		size_t fory = blockIdx.y;
		size_t revy = blockIdx.y + nangles;
		
		size_t slicez = blockIdx.z * nrays * nangles;
		
		//float Arg2 = (2.0f*idx - pad0*nrays/2 + 1.0f)/(pad0*nrays/2 - 1.0f);
		//double b1 = cyl_bessel_i0f(sqrtf(fmaxf(1.0f - Arg2 * Arg2,0.0f)));
		//double b2 = cyl_bessel_i0f(1.0f);
		float w_bessel = 1;//fabsf(b1/b2);
		if(idx==0)
			w_bessel *= 0.5f;

		w_bessel *= sq(pad0);
		
		if(center - 1 - idx >= 0)
			padded[pad0*slicez + pad0*fory*nrays/2 + idx] = w_bessel * in[slicez + fory*nrays + center - 1 - idx];
		else
			padded[pad0*slicez + pad0*fory*nrays/2 + idx] = w_bessel * in[slicez + fory*nrays];
			
		if(center + 0 + idx >= 0)
			padded[pad0*slicez + pad0*revy*nrays/2 + idx] = w_bessel * in[slicez + fory*nrays + center + 0 + idx];
		else
			padded[pad0*slicez + pad0*revy*nrays/2 + idx] = w_bessel * in[slicez + fory*nrays];
	}
}

__global__ void convBST(complex* block, size_t nrays, size_t nangles)
{
    /* Convolution BST with kernel = sigma = 2 /( Nx * max( min(i,Nx-i), 0.5) ) ) */
	size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
 	size_t ty = blockIdx.y;
	size_t tz = blockIdx.z;
 	
	float sigma = 2.0f / (nrays * (fmaxf(fminf(tx,nrays-tx),0.5f)));
	size_t offset = tz*nangles*nrays + ty*nrays + tx;
	
	if(tx < nrays)
		block[offset] *= sigma;
}

__global__ void polar2cartesian_fourier(complex* cartesian, complex* polar, float *angles,
size_t nrays, size_t nangles, size_t sizeimage)
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	// int ty = blockIdx.y;
	
	if(tx < sizeimage)
	{
		size_t cartplane = blockIdx.z * sizeimage * sizeimage;
		polar += blockIdx.z * nrays * nangles;
		
		// int posx = tx - sizeimage/2;
		// int posy = ty - sizeimage/2;
		
		// float rho = nrays * hypotf(posx,posy) / sizeimage;
		// float angle = (nangles)*(0.5f*atan2f(posy, posx)/float(M_PI)+0.5f);

        float angle = angles[tx / nrays];
        float rho = 0.5 + (tx % nrays) - nrays/2.0; // rho_idx - rho_max.
        float xpos = cosf(angle)*rho;
        float ypos = sinf(angle)*rho;
        int x_id = __float2int_rn(xpos) + (int)nrays/2;
        int y_id = __float2int_rn(ypos) + (int)nrays/2;
		
		size_t irho = size_t(rho);
		int iarc    = int(angle);
		complex interped = 0;
		
		if(irho < nrays/2-1)
		{
			float pfrac = rho-irho;
			float tfrac = iarc-angle;
			
			iarc = iarc%(nangles);
			
			int uarc = (iarc+1)%(nangles);
			
			complex interp0 = polar[iarc*nrays + irho]*(1.0f-pfrac) + polar[iarc*nrays + irho+1]*pfrac;
			complex interp1 = polar[uarc*nrays + irho]*(1.0f-pfrac) + polar[uarc*nrays + irho+1]*pfrac;
			
			interped = interp0*tfrac + interp1*(1.0f-tfrac);
		}
		
		// cartesian[cartplane + sizeimage*((ty+sizeimage/2)%sizeimage) + (tx+sizeimage/2)%sizeimage] = interped*(4*(tx%2-0.5f)*(ty%2-0.5f));
        cartesian[cartplane + sizeimage*y_id + x_id] = interped*(4*x_id*y_id);

    }
}

void BST(float* blockRecon, float *wholesinoblock, float *angles,
int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0)
{
	int blocksize = 1;

	cImage cartesianblock(sizeimage, sizeimage*blocksize);
	cImage polarblock(Nrays * pad0,Nangles*blocksize);
	cImage realpolar(Nrays * pad0,Nangles*blocksize);

    float *dangles = opt::allocGPU<float>( Nangles );
    opt::CPUToGPU<float>(angles, dangles, Nangles);

	cufftHandle plan1d;
	cufftHandle plan2d;
  
  	int dimms1d[] = {(int)Nrays*pad0/2};
  	int dimms2d[] = {(int)sizeimage,(int)sizeimage};
  	int beds[] = {Nrays*pad0/2};
  
	HANDLE_FFTERROR( cufftPlanMany(&plan1d, 1, dimms1d, beds, 1, Nrays*pad0/2, beds, 1, Nrays*pad0/2, CUFFT_C2C, Nangles*blocksize*2) );
	HANDLE_FFTERROR( cufftPlanMany(&plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize) );
	
	size_t insize = Nrays*Nangles;
	size_t outsize = sizeimage*sizeimage;
	
	for(size_t zoff = 0; zoff < (size_t)trueblocksize; zoff+=blocksize)
	{
		float* sinoblock = wholesinoblock + insize*zoff;
		
		dim3 blocks((Nrays+255)/256,Nangles,blocksize);
		dim3 threads(128,1,1); 
		
		sino2p<<<blocks,threads>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, pad0, 0);
		
		Nangles *= 2;
		Nrays *= pad0;
		Nrays /= 2;

		blocks.y *= 2;
		blocks.x *= pad0;
		blocks.x /= 2;

		HANDLE_FFTERROR(cufftExecC2C(plan1d, realpolar.gpuptr, polarblock.gpuptr, CUFFT_FORWARD));
	 	convBST<<<blocks,threads>>>(polarblock.gpuptr, Nrays, Nangles);
	 	
		blocks = dim3((sizeimage+255)/256,sizeimage,blocksize);
		threads = dim3(256,1,1);

		polar2cartesian_fourier<<<blocks,threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, dangles, Nrays, Nangles, sizeimage);
	  
		HANDLE_FFTERROR( cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE) );
	  
		cudaDeviceSynchronize();

		GetX<<<dim3((sizeimage+127)/128,sizeimage),128>>>(blockRecon + outsize*zoff, cartesianblock.gpuptr, sizeimage);

	 	HANDLE_ERROR( cudaPeekAtLastError() );
	 	
		Nangles /= 2;
		Nrays *= 2;
		Nrays /= pad0;
	}
	cufftDestroy(plan1d);
	cufftDestroy(plan2d);
    cudaFree(dangles);
}


void getBST(
	float* blockRecon, float *wholesinoblock, float *angles,
	cImage& cartesianblock, cImage& polarblock, cImage& realpolar,
	cufftHandle plan1d, cufftHandle plan2d,
	int Nrays, int Nangles, int trueblocksize, int blocksize, int sizeimage, int pad0)
{
	size_t insize = Nrays*Nangles;
	size_t outsize = sizeimage*sizeimage;
	
	for(size_t zoff = 0; zoff < (size_t)trueblocksize; zoff+=blocksize)
	{
		float* sinoblock = wholesinoblock + insize*zoff;
		
		dim3 blocks((Nrays+255)/256,Nangles,blocksize);
		dim3 threads(128,1,1); 
		
		sino2p<<<blocks,threads>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, pad0, 0);
		
		Nangles *= 2;
		Nrays *= pad0;
		Nrays /= 2;

		blocks.y *= 2;
		blocks.x *= pad0;
		blocks.x /= 2;

		HANDLE_FFTERROR(cufftExecC2C(plan1d, realpolar.gpuptr, polarblock.gpuptr, CUFFT_FORWARD));
	 	convBST<<<blocks,threads>>>(polarblock.gpuptr, Nrays, Nangles);
	 	
		blocks = dim3((sizeimage+255)/256,sizeimage,blocksize);
		threads = dim3(256,1,1);

		polar2cartesian_fourier<<<blocks,threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, angles, Nrays, Nangles, sizeimage);
	  
		HANDLE_FFTERROR( cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE) );
	  
		cudaDeviceSynchronize();

		GetX<<<dim3((sizeimage+127)/128,sizeimage),128>>>(blockRecon + outsize*zoff, cartesianblock.gpuptr, sizeimage);

	 	HANDLE_ERROR( cudaPeekAtLastError() );
	 	
		Nangles /= 2;
		Nrays *= 2;
		Nrays /= pad0;
	}
}

extern "C"{

    void getBSTGPU(	float* obj, float *tomo, float *angles,
	int nrays, int nangles, int blockgpu, int sizeimage, int pad0,
    int gpu)
    {
        HANDLE_ERROR( cudaSetDevice(gpu) );

        size_t blocksize_aux = calc_blocksize(blockgpu, nrays, nangles, pad0, true);

        size_t blocksize = min((size_t)blockgpu, blocksize_aux);

        // BST initialization starts here:
        int blocksize_bst = 1;

        float *dtomo   = opt::allocGPU<float>((size_t)    nrays *   nangles * blocksize);
        float *dobj    = opt::allocGPU<float>((size_t)sizeimage * sizeimage * blocksize);
        float *dangles = opt::allocGPU<float>( nangles );

        opt::CPUToGPU<float>(angles, dangles, nangles);	

        cImage cartesianblock_bst(sizeimage, sizeimage*blocksize_bst);
        cImage polarblock_bst(nrays * pad0, nangles*blocksize_bst);
        cImage realpolar_bst(nrays * pad0, nangles*blocksize_bst);
            
        cufftHandle plan1d_bst;
        cufftHandle plan2d_bst;

        int dimms1d[] = {(int)nrays*pad0/2};
        int dimms2d[] = {(int)sizeimage,(int)sizeimage};
        int beds[] = {nrays*pad0/2};

        HANDLE_FFTERROR( cufftPlanMany(&plan1d_bst, 1, dimms1d, beds, 1, nrays*pad0/2, beds, 1, nrays*pad0/2, CUFFT_C2C, nangles*blocksize_bst*2) );
        HANDLE_FFTERROR( cufftPlanMany(&plan2d_bst, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize_bst) );
    
        // BST initialization finishes here. 

        for (size_t b = 0; b < blockgpu; b += blocksize) {

            blocksize = min(size_t(blockgpu) - b, blocksize);

            opt::CPUToGPU<float>(tomo, dtomo, (size_t)nrays * nangles * blocksize);

            getBST( dobj, dtomo, dangles, 
                    cartesianblock_bst, polarblock_bst, realpolar_bst,
                    plan1d_bst, plan2d_bst,
                    nrays, nangles, blocksize, blocksize_bst, sizeimage, pad0);

            opt::GPUToCPU<float>(   obj + (size_t)sizeimage * sizeimage * blocksize, 
                                    dobj, 
                                    (size_t)sizeimage * sizeimage * blocksize);
        }
        cudaDeviceSynchronize();

        cudaFree(dobj);
        cudaFree(dtomo);
        cudaFree(dangles);
    }

    void getBSTMultiGPU(int* gpus, int ngpus, 
    float* obj, float *tomo, float *angles,
	int nrays, int nangles, int nslices, int sizeimage, int pad0)
    {
        int t;
        int blockgpu = (nslices + ngpus - 1) / ngpus;
        
        std::vector<std::future<void>> threads;

        for(t = 0; t < ngpus; t++){ 
            
            blockgpu = min(nslices - blockgpu * t, blockgpu);

            threads.push_back(std::async( std::launch::async, getBSTGPU, 
                                            obj  + (size_t)t * blockgpu * sizeimage * sizeimage, 
                                            tomo + (size_t)t * blockgpu * nrays * nrays, 
                                            angles,  
                                            nrays, nangles, blockgpu, 
                                            sizeimage, pad0, gpus[t]));
        }

        for(auto& t : threads)
            t.get();
    }

}


