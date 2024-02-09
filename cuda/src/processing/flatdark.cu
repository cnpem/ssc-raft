#include "common/logerror.hpp"
#include "common/operations.hpp"
#include "processing/processing.hpp"


static __global__ void KFlatDark(float* data, 
float* dark, float* flat, 
dim3 size, int numflats)
{  
    // Supports 2 flats only
    long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
    long long int idy = threadIdx.y + blockIdx.y*blockDim.y;

    long long int line;
    float ft, flat_before, flat_after, dk, T, Q, interp;

    if(idx < size.x && idy < size.y && blockIdx.z < size.z){
        
        dk          = dark[size.x * blockIdx.z            + idx];
        flat_before = flat[size.x * numflats * blockIdx.z + idx];

        line        = size.x * size.y * blockIdx.z + size.x * idy + idx;

        if(numflats > 1){
            interp     = float( idy ) / float( size.y ); 

            flat_after = flat[size.x * numflats * blockIdx.z + size.x + idx];

            ft         = flat_before * ( 1.0f - interp ) + interp * flat_after;
        }else{
            ft         = flat_before;
        }

        T          = data[line] - dk;
        Q          = ft         - dk;

        data[line] = fmaxf(T, 0.5f) / fmaxf(Q,0.5f); 
    }
}

extern "C"{
	void getFlatDarkCorrection(float* frames, 
    float* flat, float* dark, 
    dim3 size, int numflats, GPU gpus)
	{
        dim3 threadsPerBlock(gpus.BT.x,   gpus.BT.y, size.z);
        dim3       gridBlock(gpus.Grd.x, gpus.Grd.y,      1);

		/* Do the dark subtraction and division by flat (without log) */
		KFlatDark<<<gridBlock,threadsPerBlock>>>(frames, dark, flat, size, numflats);

        HANDLE_ERROR(cudaGetLastError());
	}

	void getFlatDarkGPU(GPU gpus, int gpu, 
    float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log)
	{	
		// Supports 2 flats max
		cudaSetDevice(gpu);

		int i;
		int blocksize = min(size.z,32); // Herança do Giovanni -> Mudar

		int nblock = (int)ceil( (float) size.z / blocksize );
		int ptr = 0, subblock;
        
        size_t n = size.x * size.y;
		// GPUs Pointers: declaration and allocation
        float *d_frames = opt::allocGPU<float>((size_t) n                 * blocksize);
        float *d_flat   = opt::allocGPU<float>((size_t) size.x * numflats * blocksize);
        float *d_dark   = opt::allocGPU<float>((size_t) size.x            * blocksize);

		for(i = 0; i < nblock; i++){
			
			subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * n                 , d_frames, (size_t)subblock *                 n);
            opt::CPUToGPU<float>(flat   + (size_t)ptr * size.x * numflats , d_flat  , (size_t)subblock * size.x * numflats);
            opt::CPUToGPU<float>(dark   + (size_t)ptr * size.x            , d_dark  , (size_t)subblock * size.x           );
        
			getFlatDarkCorrection(d_frames, d_flat, d_dark, dim3(size.x, size.y, subblock), numflats, gpus);

			if (is_log == 1)
				getLog(d_frames, dim3(size.x, size.y, subblock));

            opt::GPUToCPU<float>(frames + (size_t)ptr * n , d_frames, (size_t)subblock * n);

			/* Update pointer */
			ptr = ptr + subblock;
		}
        cudaFree(d_frames);
        cudaFree(d_flat);
        cudaFree(d_dark);

        HANDLE_ERROR(cudaGetLastError());

		cudaDeviceSynchronize();    
	}

	void getFlatDarkMultiGPU(int* gpus, int ngpus, 
    float* frames, float* flat, float* dark, 
    int nrays, int nangles, int nslices, int numflats, 
    int is_log)
	{
		int i;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		int ptr = 0, subblock;
		
		GPU gpu_parameters;

		setGPUParameters(&gpu_parameters, dim3(nrays,nangles,nslices), ngpus, gpus);

		std::vector<std::future<void>> threads;
		
		if ( ngpus == 1 ){
			getFlatDarkGPU( gpu_parameters, 
                            gpus[0], 
                            frames, 
                            flat, 
                            dark, 
                            dim3(nrays, nangles, nslices), 
                            numflats, is_log);
		}else{
			for(i = 0; i < ngpus; i++){ 
				
				subblock = min(nslices - ptr, blockgpu);

				threads.push_back(std::async( std::launch::async, 
							getFlatDarkGPU, 
							gpu_parameters,
							gpus[i], 
							frames + (size_t)ptr * nrays * nangles, 
							flat   + (size_t)ptr * nrays * numflats, 
							dark   + (size_t)ptr * nrays, 
							dim3(nrays, nangles, subblock), 
							numflats, is_log
							));

				/* Update pointer */
				ptr = ptr + subblock;
			}

			for(auto& t : threads)
				t.get();
		}
	}
}

