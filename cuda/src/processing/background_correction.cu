#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <unistd.h>
#include <cstddef>
#include <cstdlib>
#include <vector>
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
    dim3 size, int numflats, cudaStream_t stream)
	{
        dim3 threadsPerBlock( gpus.BT.x,  gpus.BT.y,           1);
        dim3       gridBlock(gpus.Grd.x, gpus.Grd.y,      size.z);
        
		/* Do the dark subtraction and division by flat (without log) */
        BackgroundCorrection<<<gridBlock,threadsPerBlock, 0, stream>>>(frames, dark, flat, size, numflats);

        HANDLE_ERROR(cudaGetLastError());
	}

	void getBackgroundCorrectionGPU(GPU gpus, int gpu, 
    float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log, int blocksize)
	{
		// Supports 2 flats max
		HANDLE_ERROR(cudaSetDevice(gpu));

        const size_t nstreams = 2;

		int i;
        size_t total_required_mem_per_slice_bytes = static_cast<float>(sizeof(float)) * ( size.x * size.y + 6 * size.x ) * nstreams;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(size.z, total_required_mem_per_slice_bytes, 
                                                        true, BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(size.z, blocksize_aux);
        }
        int nblock = (int)ceil( (float) size.z / blocksize );
		int ptr = 0, subblock;

        float *d_flat = opt::allocGPU<float>((size_t) size.z * size.x * numflats * blocksize);
        float *d_dark = opt::allocGPU<float>((size_t) size.z * size.x * blocksize);

        opt::CPUToGPU<float>(flat, d_flat, (size_t)size.z * size.x * numflats);
        opt::CPUToGPU<float>(dark, d_dark, (size_t)size.z * size.x);

        float *d_frames[nstreams];
        cudaStream_t streams[nstreams];

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamCreate(&streams[st]);

            d_frames[st] = opt::allocGPU<float>((size_t) size.x * size.y * blocksize, streams[st]);
        }

		for(i = 0; i < nblock; i++) {
            int st = i % nstreams;
            cudaStream_t stream = streams[i % nstreams];
            
			subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames[st], (size_t)subblock * size.x * size.y, stream);

			getBackgroundCorrection(gpus, d_frames[st],
                    d_flat + (size_t)ptr * size.x * numflats,
                    d_dark + (size_t)ptr * size.x,
                    dim3(size.x, size.y, subblock), numflats, stream);

            if (is_log == 1)
                getLog(d_frames[st], dim3(size.x, size.y, subblock), stream);

            opt::GPUToCPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames[st], (size_t)subblock * size.x * size.y, stream);

			/* Update pointer */
			ptr = ptr + subblock;
        }

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamSynchronize(streams[st]);

            HANDLE_ERROR(cudaFreeAsync(d_frames[st], streams[st]));

            cudaStreamDestroy(streams[st]);
        }

        HANDLE_ERROR(cudaFree(d_flat));
        HANDLE_ERROR(cudaFree(d_dark));

        HANDLE_ERROR(cudaDeviceSynchronize());

	}

	void getBackgroundCorrectionMultiGPU(int* gpus, int ngpus,
    float* frames, float* flat, float* dark,
    int nrays, int nangles, int nslices, int numflats,
    int is_log, int blocksize)
	{
		int i;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		int ptr = 0, subblock;

        GPU gpu_parameters;
        setGPUParameters(&gpu_parameters, dim3(nrays,nangles,nslices), ngpus, gpus);

		std::vector<std::future<void>> threads;
        threads.reserve(ngpus);

        for (i = 0; i < ngpus; i++) {
            subblock = min(nslices - ptr, blockgpu);

            threads.push_back(std::async( std::launch::async,
                getBackgroundCorrectionGPU,
                gpu_parameters,
                gpus[i],
                frames + (size_t)ptr * nrays * nangles,
                flat   + (size_t)ptr * nrays * numflats,
                dark   + (size_t)ptr * nrays,
                dim3(nrays, nangles, subblock),
                numflats, is_log, blocksize
                ));

            /* Update pointer */
            ptr = ptr + subblock;
        }

        for(auto& t : threads)
            t.get();
	}
}

