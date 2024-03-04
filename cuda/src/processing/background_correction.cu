#include "common/logerror.hpp"
#include "common/opt.hpp"
#include "processing/processing.hpp"


static __global__ void BackgroundCorrection(float* data, 
float* dark, float* flat, 
dim3 size, int numflats)
{  
    // Supports 2 flats only
    long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
    long long int idy = threadIdx.y + blockIdx.y*blockDim.y;
    long long int idz = threadIdx.z + blockIdx.z*blockDim.z;

    long long int line;
    float ft, flat_before, flat_after, dk, T, Q, interp;

    if(idx < size.x && idy < size.y && idz < size.z){
        
        dk          = dark[size.x * idz            + idx];
        flat_before = flat[size.x * numflats * idz + idx];

        line        = size.x * size.y * idz + size.x * idy + idx;

        if(numflats > 1){
            interp  = float( idy ) / float( size.y ); 

            flat_after = flat[size.x * numflats * idz + size.x + idx];

            ft      = flat_before * ( 1.0f - interp ) + interp * flat_after;
        }else{
            ft      = flat_before;
        }

        T          = data[line] - dk;
        Q          = ft         - dk;

        data[line] = fmaxf(T, 0.5f) / fmaxf(Q,0.5f); 
    }
}

extern "C"{
	void getBackgroundCorrection(GPU gpus, 
    float* frames, float* flat, float* dark, 
    dim3 size, int numflats)
	{
        dim3 threadsPerBlock( gpus.BT.x,  gpus.BT.y,           1);
        dim3       gridBlock(gpus.Grd.x, gpus.Grd.y,      size.z);
        
		/* Do the dark subtraction and division by flat (without log) */
		BackgroundCorrection<<<gridBlock,threadsPerBlock>>>(frames, dark, flat, size, numflats);

        HANDLE_ERROR(cudaGetLastError());
	}

	void getBackgroundCorrectionGPU(GPU gpus, int gpu, 
    float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log)
	{	
		// Supports 2 flats max
		HANDLE_ERROR(cudaSetDevice(gpu));

		int i;
		int blocksize = min(size.z,32); // Herança do Giovanni -> Mudar

		int nblock = (int)ceil( (float) size.z / blocksize );
		int ptr = 0, subblock;
        
		// GPUs Pointers: declaration and allocation
        float *d_frames = opt::allocGPU<float>((size_t) size.x *   size.y * blocksize);
        float *d_flat   = opt::allocGPU<float>((size_t) size.x * numflats * blocksize);
        float *d_dark   = opt::allocGPU<float>((size_t) size.x            * blocksize);

		for(i = 0; i < nblock; i++){
			
			subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * size.x *   size.y, d_frames, (size_t)subblock * size.x *   size.y);
            opt::CPUToGPU<float>(flat   + (size_t)ptr * size.x * numflats, d_flat  , (size_t)subblock * size.x * numflats);
            opt::CPUToGPU<float>(dark   + (size_t)ptr * size.x           , d_dark  , (size_t)subblock * size.x           );
        
			getBackgroundCorrection(gpus, d_frames, d_flat, d_dark, dim3(size.x, size.y, subblock), numflats);

			if (is_log == 1)
				getLog(d_frames, dim3(size.x, size.y, subblock));
            
            opt::GPUToCPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames, (size_t)subblock * size.x * size.y);

			/* Update pointer */
			ptr = ptr + subblock;
		}
        HANDLE_ERROR(cudaDeviceSynchronize());    

        HANDLE_ERROR(cudaFree(d_frames));
        HANDLE_ERROR(cudaFree(d_flat));
        HANDLE_ERROR(cudaFree(d_dark));
	}

	void getBackgroundCorrectionMultiGPU(int* gpus, int ngpus, 
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
        threads.reserve(ngpus); 
		
		if ( ngpus == 1 ){
			getBackgroundCorrectionGPU( gpu_parameters, 
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
                    getBackgroundCorrectionGPU, 
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

