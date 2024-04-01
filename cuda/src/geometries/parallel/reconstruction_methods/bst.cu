// Authors: Giovanni Baraldi, Eduardo X. Miqueles

#include "geometries/parallel/radon.hpp"
#include "geometries/parallel/bst.hpp"
#include "processing/filters.hpp"
#include "common/complex.hpp"
#include "common/types.hpp"
#include "common/opt.hpp"
#include "common/logerror.hpp"
#include <logger.hpp>

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
	int ty = blockIdx.y;
	
	if(tx < sizeimage)
	{
		size_t cartplane = blockIdx.z * sizeimage * sizeimage;
		polar += blockIdx.z * nrays * nangles;
		
		int posx = tx - sizeimage/2;
		int posy = ty - sizeimage/2;
		
		float rho = nrays * hypotf(posx,posy) / sizeimage;
		float angle = (nangles)*(0.5f*atan2f(posy, posx)/float(M_PI)+0.5f);
		
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
		
		cartesian[cartplane + sizeimage*((ty+sizeimage/2)%sizeimage) + (tx+sizeimage/2)%sizeimage] = interped*(4*(tx%2-0.5f)*(ty%2-0.5f));

    }
}


void EMFQ_BST(float* blockRecon, float *wholesinoblock, float *angles,
int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0)
{
	int blocksize = 1;

    Filter filter(Filter::EType::none, 1, 0.0, 0);

    int blocksize_bst = 1;
    cImage filtersino(Nrays, Nangles*blocksize_bst);

    cufftHandle filterplan;
    int dimmsfilter[] = {Nrays};
    HANDLE_FFTERROR( cufftPlanMany(&filterplan, 1, dimmsfilter, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, Nangles*blocksize_bst) );

	cImage cartesianblock(sizeimage, sizeimage*blocksize);
	cImage polarblock(Nrays * pad0,Nangles*blocksize);
	cImage realpolar(Nrays * pad0,Nangles*blocksize);

	cufftHandle plan1d;
	cufftHandle plan2d;
  
  	int dimms1d[] = {(int)Nrays*pad0/2};
  	int dimms2d[] = {(int)sizeimage,(int)sizeimage};
  	int beds[] = {(int)Nrays*pad0/2};
  
	HANDLE_FFTERROR( cufftPlanMany(&plan1d, 1, dimms1d, beds, 1, Nrays*pad0/2, beds, 1, Nrays*pad0/2, CUFFT_C2C, Nangles*blocksize*2) );
	HANDLE_FFTERROR( cufftPlanMany(&plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize) );
	
	size_t insize = Nrays*Nangles;
	size_t outsize = sizeimage*sizeimage;
	
	for(size_t zoff = 0; zoff < (size_t)trueblocksize; zoff+=blocksize)
	{
		float* sinoblock = wholesinoblock + insize*zoff;

        if (filter.type != Filter::EType::none)
            BSTFilter(filterplan, filtersino.gpuptr, sinoblock, Nrays, Nangles, 0.0, filter);
		
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

        HANDLE_ERROR( cudaPeekAtLastError() );
		polar2cartesian_fourier<<<blocks,threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, angles, Nrays, Nangles, sizeimage);
	  
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
}

void EMFQ_BST_ITER(
	float* blockRecon, float *wholesinoblock, float *angles,
	cImage& cartesianblock, cImage& polarblock, cImage& realpolar,
	cufftHandle plan1d, cufftHandle plan2d,
	int Nrays, int Nangles, int trueblocksize, int blocksize, int sizeimage, 
    int pad0)
    {
	size_t insize = Nrays*Nangles;
	size_t outsize = sizeimage*sizeimage;

    Filter filter(Filter::EType::none, 1, 0.0, 0);

    int blocksize_bst = 1;
    cImage filtersino(Nrays, Nangles*blocksize_bst);

    cufftHandle filterplan;
    int dimmsfilter[] = {Nrays};
    HANDLE_FFTERROR( cufftPlanMany(&filterplan, 1, dimmsfilter, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, Nangles*blocksize_bst) );

	for(size_t zoff = 0; zoff < (size_t)trueblocksize; zoff+=blocksize)
	{
		float* sinoblock = wholesinoblock + insize*zoff;
		
        if (filter.type != Filter::EType::none)
            BSTFilter(filterplan, filtersino.gpuptr, sinoblock, Nrays, Nangles, 0.0, filter);

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

void getBST(
float* blockRecon, float *wholesinoblock, float *angles,
int Nrays, int Nangles, int trueblocksize, int sizeimage, 
int pad0, float reg, float paganin, int filter_type, int offset, int gpu)
{
    HANDLE_ERROR( cudaSetDevice(gpu) );

    ssc_event_start("getBST()", {
        ssc_param_int("GPU device number", gpu),
        ssc_param_int("Sub blocksize", trueblocksize)
    });

    int blocksize_bst = 1;

	size_t insize = Nrays*Nangles;
	size_t outsize = sizeimage*sizeimage;

    Filter filter(filter_type, reg, paganin, offset);

    cImage filtersino(Nrays, Nangles*blocksize_bst);

    cufftHandle filterplan;
    int dimmsfilter[] = {Nrays};
    HANDLE_FFTERROR( cufftPlanMany(&filterplan, 1, dimmsfilter, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, Nangles*blocksize_bst) );

    cImage cartesianblock(sizeimage, sizeimage*blocksize_bst);
    cImage polarblock(Nrays * pad0, Nangles*blocksize_bst);
    cImage realpolar(Nrays * pad0, Nangles*blocksize_bst);
        
    cufftHandle plan1d;
    cufftHandle plan2d;

    int dimms1d[] = {(int)Nrays*pad0/2};
    int dimms2d[] = {(int)sizeimage,(int)sizeimage};
    int beds[] = {Nrays*pad0/2};

    HANDLE_FFTERROR( cufftPlanMany(&plan1d, 1, dimms1d, beds, 1, Nrays*pad0/2, beds, 1, Nrays*pad0/2, CUFFT_C2C, Nangles*blocksize_bst*2) );
    HANDLE_FFTERROR( cufftPlanMany(&plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize_bst) );

    // BST initialization finishes here. 

	for(size_t zoff = 0; zoff < (size_t)trueblocksize; zoff+=blocksize_bst)
	{
		float* sinoblock = wholesinoblock + insize*zoff;
        
        ssc_trace_start("BST_Filter");

        if (filter.type != Filter::EType::none)
            BSTFilter(filterplan, filtersino.gpuptr, sinoblock, Nrays, Nangles, 0.0, filter);

        ssc_trace_stop(); /* BST_Filter */
		
		dim3 blocks((Nrays+255)/256,Nangles,blocksize_bst);
		dim3 threads(128,1,1); 
		
        ssc_trace_start("BST_Core");

		sino2p<<<blocks,threads>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, pad0, 0);
		
		Nangles *= 2;
		Nrays *= pad0;
		Nrays /= 2;

		blocks.y *= 2;
		blocks.x *= pad0;
		blocks.x /= 2;

		HANDLE_FFTERROR(cufftExecC2C(plan1d, realpolar.gpuptr, polarblock.gpuptr, CUFFT_FORWARD));
	 	convBST<<<blocks,threads>>>(polarblock.gpuptr, Nrays, Nangles);
	 	
		blocks = dim3((sizeimage+255)/256,sizeimage,blocksize_bst);
		threads = dim3(256,1,1);

		polar2cartesian_fourier<<<blocks,threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, angles, Nrays, Nangles, sizeimage);
	  
		HANDLE_FFTERROR( cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE) );
	  
		cudaDeviceSynchronize();

		GetX<<<dim3((sizeimage+127)/128,sizeimage),128>>>(blockRecon + outsize*zoff, cartesianblock.gpuptr, sizeimage);

        ssc_trace_stop(); /* BST_Core */

	 	HANDLE_ERROR( cudaPeekAtLastError() );
	 	
		Nangles /= 2;
		Nrays *= 2;
		Nrays /= pad0;
	}
    cufftDestroy(filterplan);
    cufftDestroy(plan1d);
    cufftDestroy(plan2d);

    ssc_event_stop(); /* getBST */

}

extern "C"{

    void getBSTGPU(	CFG configs, 
    float* obj, float *tomo, float *angles,
	int blockgpu, int gpu)
    {
        HANDLE_ERROR( cudaSetDevice(gpu) );

        /* Projection data sizes */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;

        /* Reconstruction sizes */
        int sizeImagex = configs.obj.size.x;

        int padding          = configs.tomo.pad.x;
        int filter_type      = configs.reconstruction_filter_type;
        float paganin_reg    = configs.reconstruction_paganin_reg;
        float regularization = configs.reconstruction_reg;
        int axis_offset      = configs.rotation_axis_offset;

        int blocksize        = configs.blocksize;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(blockgpu, configs.total_required_mem_per_slice_bytes, true, A100_MEM);
            blocksize          = min(blockgpu, blocksize_aux);
        }
        int ind_block = (int)ceil( (float) blockgpu / blocksize );
        int ptr = 0, subblock;

        ssc_event_start("getBSTGPU()", {
            ssc_param_int("GPU device number", gpu),
            ssc_param_int("nrays", nrays),
            ssc_param_int("nangles", nangles),
            ssc_param_int("sizeImagex", sizeImagex),
            ssc_param_int("Axis offset", axis_offset),
            ssc_param_int("Filter type", filter_type),
            ssc_param_int("GPU blocksize", blockgpu),
            ssc_param_int("Computed sub blocks", blocksize),
            ssc_param_int("Number of sub blocks", ind_block)
        });
        
        float *dtomo   = opt::allocGPU<float>((size_t)     nrays *    nangles * blocksize);
        float *dobj    = opt::allocGPU<float>((size_t)sizeImagex * sizeImagex * blocksize);
        float *dangles = opt::allocGPU<float>( nangles );

        opt::CPUToGPU<float>(angles, dangles, nangles);	

        for (int i = 0; i < ind_block; i++){

            subblock = min(blockgpu - ptr, (int)blocksize);

            opt::CPUToGPU<float>(tomo + (size_t)ptr * nrays * nangles, 
                                 dtomo, (size_t)nrays * nangles * subblock);

            getBST( dobj, dtomo, dangles, 
                    nrays, nangles, subblock, sizeImagex, padding,
                    regularization, paganin_reg, filter_type, axis_offset, gpu);

            opt::GPUToCPU<float>(obj + (size_t)ptr * sizeImagex * sizeImagex, 
                                 dobj, (size_t)sizeImagex * sizeImagex * subblock);

            /* Update pointer */
			ptr = ptr + subblock;
        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaFree(dobj));
        HANDLE_ERROR(cudaFree(dtomo));
        HANDLE_ERROR(cudaFree(dangles));

        ssc_event_stop(); /* getBSTGPU() */
    }

    void getBSTMultiGPU(int* gpus, int ngpus, 
    float* obj, float* tomogram, float* angles, 
    float *paramf, int *parami)
    {
        ssc_event_start("getBSTMultiGPU()", {ssc_param_int("ngpus", ngpus)});

        int i, Maxgpudev;

		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");
        CFG configs; GPU gpu_parameters;

        setBSTParameters(&configs, paramf, parami);
        // printBSTParameters(&configs);

        /* Projection data sizes */
        int nrays    = configs.tomo.size.x;
        int nangles  = configs.tomo.size.y;
        int nslices  = configs.tomo.size.z;

        /* Reconstruction sizes */
        int sizeImagex = configs.obj.size.x;

        int blockgpu = (nslices + ngpus - 1) / ngpus;
        int subblock, ptr = 0; 
        
        std::vector<std::future<void>> threads;
        threads.reserve(ngpus);

        if (ngpus == 1){ /* 1 device */

            getBSTGPU(configs, obj, tomogram, angles, nslices, gpus[0]);

        }else{        

            for(i = 0; i < ngpus; i++){ 
                
                subblock   = min(nslices - ptr, blockgpu);

                threads.push_back(std::async( std::launch::async, 
                    getBSTGPU, configs,
                    obj      + (size_t)ptr * sizeImagex * sizeImagex, 
                    tomogram + (size_t)ptr *     nrays *   nangles, 
                    angles,  subblock, gpus[i]));

                /* Update pointer */
				ptr = ptr + subblock;        
            }
            for (i = 0; i < ngpus; i++)
				threads[i].get();
        }
        ssc_event_stop(); /* getBSTMultiGPU */
    }
}


