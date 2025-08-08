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


static __global__ void BackgroundCorrection_slices(float* data, 
float* dark, float* flat, 
dim3 size, int numflats, int is_log)
{  
    // Supports 2 flats only
    long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
    long long int idy = threadIdx.y + blockIdx.y*blockDim.y;
    long long int idz = threadIdx.z + blockIdx.z*blockDim.z;

    long long int line;
    float ft, flat_before, flat_after, dk, T, Q, interp;

    if(idx < size.x && idy < size.y && idz < size.z){
        
        dk          = dark[size.x * idz + idx];
        flat_before = flat[size.x * idz + idx];

        line        = size.x * size.y * idz + size.x * idy + idx;

        if(numflats > 1){
            interp  = float( idy ) / float( size.y ); 

            flat_after = flat[size.x * size.y + size.x * idz + idx];

            ft      = flat_before * ( 1.0f - interp ) + interp * flat_after;
        }else{
            ft      = flat_before;
        }

        T          = data[line] - dk;
        Q          = ft         - dk;

        data[line] = fmaxf(T, 0.5f) / fmaxf(Q,0.5f); 

        if ( is_log == 1 ) data[line] = - logf(data[line]);
    }
}

static __global__ void BackgroundCorrection_frames(float* data, 
    float* dark, float* flat, 
    dim3 size, int numflats, int is_log)
    {  
        // Supports 2 flats only
        long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
        long long int idy = threadIdx.y + blockIdx.y*blockDim.y;
        long long int idz = threadIdx.z + blockIdx.z*blockDim.z;
    
        long long int line;
        float ft, flat_before, flat_after, dk, T, Q, interp;
    
        if(idx < size.x && idy < size.y && idz < size.z){
            
            dk          = dark[size.x * idy + idx];
            flat_before = flat[size.x * idy + idx];
    
            line        = size.x * size.y * idz + size.x * idy + idx;
    
            if(numflats > 1){
                interp  = float( idz ) / float( size.z ); 
    
                flat_after = flat[size.x * size.y + size.x * idy + idx];
    
                ft      = flat_before * ( 1.0f - interp ) + interp * flat_after;
            }else{
                ft      = flat_before;
            }
    
            T          = data[line] - dk;
            Q          = ft         - dk;
    
            data[line] = fmaxf(T, 0.5f) / fmaxf(Q,0.5f); 
    
            if ( is_log == 1 ) data[line] = - logf(data[line]);
        }
    }


extern "C"{
	void getBackgroundCorrection_slices(float* frames, float* flat, float* dark, 
        dim3 size, int numflats, int is_log, cudaStream_t stream)
	{
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);
        
        /* Do the dark subtraction and division by flat (without log) */
        BackgroundCorrection_slices<<<gridBlock,threadsPerBlock, 0, stream>>>(frames, dark, flat, 
                                                                                  size, numflats, is_log);
        HANDLE_ERROR(cudaGetLastError());
	}

    void getBackgroundCorrection_frames(float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log, cudaStream_t stream)
    {
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);
        
        /* Do the dark subtraction and division by flat (without log) */
        BackgroundCorrection_frames<<<gridBlock,threadsPerBlock, 0, stream>>>(frames, dark, flat, 
                                                                                    size, numflats, is_log);

        HANDLE_ERROR(cudaGetLastError());
    }

	void getBackgroundCorrectionGPU(int gpu, 
    float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log, int input_slices, int blocksize)
	{
		// Supports 2 flats max
		HANDLE_ERROR(cudaSetDevice(gpu));

        int nslices;
        if ( input_slices == 1 ){ nslices = size.z; }else{  nslices = size.y; }

        const size_t nstreams = 2;
		int i;
        size_t total_required_mem_per_slice_bytes = nstreams * (static_cast<float>(sizeof(float)) * ( size.x * size.y             ) + // Raw data sinogram
                                                                static_cast<float>(sizeof(float)) * ( size.x * nslices * numflats ) + // Flat line
                                                                static_cast<float>(sizeof(float)) * ( size.x * nslices            )   // Dark line
                                                                );
        if ( blocksize == 0 )
        {
            int blocksize_aux  = compute_GPU_blocksize(size.z, total_required_mem_per_slice_bytes, 
                                                        true, BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(size.z, blocksize_aux);
            blocksize          = min(32, blocksize);
        }
        int nblock = (int)ceil( (float) size.z / blocksize );
		int ptr = 0, subblock;

        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);

        float *d_frames[nstreams];
        float *d_flat[nstreams];
        float *d_dark[nstreams];
        cudaStream_t streams[nstreams];

        for (int st = 0; st < nstreams; ++st) 
        {
            cudaStreamCreate(&streams[st]);

            d_frames[st] = opt::allocGPU<float>((size_t) size.x * size.y * blocksize , streams[st]);
            d_flat[st]   = opt::allocGPU<float>((size_t) size.x * nslices * numflats, streams[st]);
            d_dark[st]   = opt::allocGPU<float>((size_t) size.x * nslices           , streams[st]);

            opt::CPUToGPU<float>(flat, d_flat[st], (size_t)size.x * nslices * numflats, streams[st]);
            opt::CPUToGPU<float>(dark, d_dark[st], (size_t)size.x * nslices           , streams[st]);
        }

		for(i = 0; i < nblock; i++) {
            int st = i % nstreams;
            cudaStream_t stream = streams[i % nstreams];
            
			subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames[st], (size_t)subblock * size.x * size.y, stream);

            if ( input_slices == 1 ){
                BackgroundCorrection_slices<<<gridBlock,threadsPerBlock, 0, stream>>>(d_frames[st],
                                                                                      d_dark[st] + (size_t)ptr * size.x,
                                                                                      d_flat[st] + (size_t)ptr * size.x * numflats, 
                                                                                      dim3(size.x,size.y,subblock),
                                                                                      numflats,
                                                                                      is_log);
            }else{
                BackgroundCorrection_frames<<<gridBlock,threadsPerBlock, 0, stream>>>(d_frames[st],
                                                                                      d_dark[st],
                                                                                      d_flat[st], 
                                                                                      dim3(size.x,size.y,subblock),
                                                                                      numflats,
                                                                                      is_log);
            }
            opt::GPUToCPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames[st], (size_t)subblock * size.x * size.y, stream);

			/* Update pointer */
			ptr = ptr + subblock;
        }

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamSynchronize(streams[st]);

            HANDLE_ERROR(cudaFreeAsync(d_frames[st], streams[st]));
            HANDLE_ERROR(cudaFreeAsync(d_flat[st]  , streams[st]));
            HANDLE_ERROR(cudaFreeAsync(d_dark[st]  , streams[st]));

            cudaStreamDestroy(streams[st]);
        }

        HANDLE_ERROR(cudaDeviceSynchronize());

	}

	void getBackgroundCorrectionMultiGPU(int* gpus, int ngpus,
    float* frames, float* flat, float* dark,
    int sizex, int sizey, int sizez, int numflats,
    int is_log, int input_slices, int blocksize)
	{
		int i;
		int blockgpu = (sizez + ngpus - 1) / ngpus;
		int ptr = 0, subblock;

		std::vector<std::future<void>> threads;
        threads.reserve(ngpus);

        for (i = 0; i < ngpus; i++) {
            subblock = min(sizez - ptr, blockgpu);

            threads.push_back(std::async( std::launch::async,
                getBackgroundCorrectionGPU,
                gpus[i],
                frames + (size_t)ptr * sizex * sizey,
                flat   + (size_t)ptr * sizex * numflats * input_slices,
                dark   + (size_t)ptr * sizex            * input_slices,
                dim3(sizex, sizey, subblock),
                numflats, is_log, input_slices, blocksize
                ));

            /* Update pointer */
            ptr = ptr + subblock;
        }

        for(auto& t : threads)
            t.get();
	}
}

