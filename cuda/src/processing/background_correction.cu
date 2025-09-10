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
float* dark, float* flat, dim3 size, int numflats, int is_log)
{  
    // Supports 2 flats only
    long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
    long long int idy = threadIdx.y + blockIdx.y*blockDim.y;
    long long int idz = threadIdx.z + blockIdx.z*blockDim.z;

    long long int line;
    float ft, flat_before, flat_after, dk, T, Q, interp;

    if(idx < size.x && idy < size.y && idz < size.z){
        
        dk          = dark[size.x * idz + idx]; 
        flat_before = flat[size.x * numflats * 0 + size.x * idz + idx]; /* size.x * numflats * 0 + size.x * idz + idx */

        line        = size.x * size.y * idz + size.x * idy + idx;

        if(numflats > 1){
            interp  = float( idy ) / float( size.y ); 

            flat_after = flat[size.x * numflats + size.x * idz + idx]; /* size.x * numflats * 1 + size.x * idz + idx */

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
float* dark, float* flat, dim3 size, int numflats, int is_log)
{  
    // Supports 2 flats only
    long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
    long long int idy = threadIdx.y + blockIdx.y*blockDim.y;
    long long int idz = threadIdx.z + blockIdx.z*blockDim.z;

    long long int line;
    float ft, flat_before, flat_after, dk, T, Q, interp;

    if(idx < size.x && idy < size.y && idz < size.z){
        
        dk          = dark[size.x * idy + idx];
        flat_before = flat[size.x * idy + idx]; /* size.x * size.y * 0 + size.x * idy + idx */

        line        = size.x * size.y * idz + size.x * idy + idx;

        if(numflats > 1){
            interp  = float( idz ) / float( size.z ); 

            flat_after = flat[size.x * size.y + size.x * idy + idx]; /* size.x * size.y * 1 + size.x * idy + idx */

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

static __global__ void BackgroundCorrection_framesTranspose(float* data, float *out,
float* dark, float* flat, dim3 size, int numflats, int is_log)
{  
    // Supports 2 flats only
    long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
    long long int idy = threadIdx.y + blockIdx.y*blockDim.y;
    long long int idz = threadIdx.z + blockIdx.z*blockDim.z;

    long long int line, coll;
    float ft, flat_before, flat_after, dk, T, Q, interp;

    if(idx < size.x && idy < size.y && idz < size.z){
        
        dk          = dark[size.x * idy + idx];
        flat_before = flat[size.x * idy + idx]; /* size.x * size.y * 0 + size.x * idy + idx */

        line        = size.x * size.y * idz + size.x * idy + idx;
        coll        = size.x * size.z * idy + size.x * idz + idx;

        if(numflats > 1){
            interp  = float( idz ) / float( size.z ); 

            flat_after = flat[size.x * size.y + size.x * idy + idx]; /* size.x * size.y * 1 + size.x * idy + idx */

            ft      = flat_before * ( 1.0f - interp ) + interp * flat_after;
        }else{
            ft      = flat_before;
        }

        T          = data[line] - dk;
        Q          = ft         - dk;

        out[coll] = fmaxf(T, 0.5f) / fmaxf(Q,0.5f); 

        if ( is_log == 1 ) out[coll] = - logf(out[coll]);
    }
}
        
extern "C"{
	void getBackgroundCorrection_slices(float* frames, float* flat, float* dark, 
        dim3 size, int numflats, int is_log)
	{
        /* 
        frames: tomogram volume with axis (size.x,size.y,size.z) = (nrays, nangles, nslices):
        flat: Axis ALWAYS (size.x,nslices,numflats) = (nrays, nslices, numflats)
        dark: Axis ALWAYS (size.x,nslices,       1) = (nrays, nslices,        1)
        */
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);
        
        /* Do the dark subtraction and division by flat */
        BackgroundCorrection_slices<<<gridBlock,threadsPerBlock>>>(frames, dark, flat, size, numflats, is_log);
        HANDLE_ERROR(cudaGetLastError());
	}

    void getBackgroundCorrection_frames(float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log)
    {
        /* 
        frames: tomogram volume with axis (size.x,size.y,size.z) = (nrays, nslices, nangles):
        flat: Axis ALWAYS (size.x,size.y,numflats) = (nrays, nslices, numflats)
        dark: Axis ALWAYS (size.x,size.y,       1) = (nrays, nslices,        1)
        */
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);
        
        /* Do the dark subtraction and division by flat */
        BackgroundCorrection_frames<<<gridBlock,threadsPerBlock>>>(frames, dark, flat, size, numflats, is_log);

        HANDLE_ERROR(cudaGetLastError());
    }

    void getBackgroundCorrection_framesTranspose(float* frames, float* flat, float* dark, 
        dim3 size, int numflats, int is_log)
        {
            /* 
            frames: tomogram volume with axis (size.x,size.y,size.z) = (nrays, nslices, nangles):
            flat: Axis ALWAYS (size.x,size.y,numflats) = (nrays, nslices, numflats)
            dark: Axis ALWAYS (size.x,size.y,       1) = (nrays, nslices,        1)
            */
            dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
            dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);

            size_t nsize = size.x * size.y * size.z;

            float *out = opt::allocGPU<float>((nsize));
            
            /* Do the dark subtraction and division by flat */
            BackgroundCorrection_framesTranspose<<<gridBlock,threadsPerBlock>>>(frames, out, dark, flat, size, numflats, is_log);

            opt::GPUToGPU<float>(out, frames, nsize);

            HANDLE_ERROR(cudaFree(out));
    
            HANDLE_ERROR(cudaGetLastError());
        }

    void getBackgroundCorrectionGPU_slicesStreams(int gpu, float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log, int blocksize, const int nstreams)
    {
        // Supports 2 flats max
        HANDLE_ERROR(cudaSetDevice(gpu));

        int i;
        size_t total_required_mem_per_slice_bytes = (static_cast<float>(sizeof(float)) * ( size.x * size.y            ) + // Raw data sinogram
                                                     static_cast<float>(sizeof(float)) * ( size.x * size.z * numflats ) + // Flat line
                                                     static_cast<float>(sizeof(float)) * ( size.x * size.z            )   // Dark line
                                                    );
                                                    
        total_required_mem_per_slice_bytes *= nstreams;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(size.z, 
                                                       total_required_mem_per_slice_bytes, 
                                                       true, 
                                                       BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(size.z, blocksize_aux);
            blocksize          = min(32, blocksize); /* Set up a maximum blocksize for now */
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

            d_frames[st] = opt::allocGPU<float>((size_t) size.x * size.y * blocksize, streams[st]);
            d_flat[st]   = opt::allocGPU<float>((size_t) size.x * size.z *  numflats, streams[st]);
            d_dark[st]   = opt::allocGPU<float>((size_t) size.x * size.z            , streams[st]);

            opt::CPUToGPU<float>(flat, d_flat[st], (size_t)size.x * size.z * numflats, streams[st]);
            opt::CPUToGPU<float>(dark, d_dark[st], (size_t)size.x * size.z           , streams[st]);
        }

        for(i = 0; i < nblock; i++) {
            int st = i % nstreams;
            cudaStream_t stream = streams[i % nstreams];
            
            subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames[st], (size_t)subblock * size.x * size.y, stream);

            BackgroundCorrection_slices<<<gridBlock,threadsPerBlock, 0, stream>>>(d_frames[st],
                                                                                  d_dark[st] + (size_t)ptr * size.x,
                                                                                  d_flat[st] + (size_t)ptr * size.x * numflats, 
                                                                                  dim3(size.x,size.y,subblock),
                                                                                  numflats,
                                                                                  is_log);
            
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

	void getBackgroundCorrectionGPU_slices(int gpu, float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log, int blocksize)
	{
		// Supports 2 flats max
		HANDLE_ERROR(cudaSetDevice(gpu));

		int i;

        size_t total_required_mem_per_slice_bytes = (static_cast<float>(sizeof(float)) * ( size.x * size.y            ) + // Raw data sinogram
                                                     static_cast<float>(sizeof(float)) * ( size.x * size.z * numflats ) + // Flat line
                                                     static_cast<float>(sizeof(float)) * ( size.x * size.z            )   // Dark line
                                                    );
        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(size.z, 
                                                       total_required_mem_per_slice_bytes, 
                                                       true, 
                                                       BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(size.z, blocksize_aux);
            blocksize          = min(32, blocksize); /* Set up a maximum blocksize for now */
        }
        int nblock = (int)ceil( (float) size.z / blocksize );
		int ptr = 0, subblock;

        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);

        float *d_frames = opt::allocGPU<float>((size_t) size.x * size.y * blocksize);
        float *d_flat   = opt::allocGPU<float>((size_t) size.x * size.z *  numflats);
        float *d_dark   = opt::allocGPU<float>((size_t) size.x * size.z            );

        opt::CPUToGPU<float>(flat, d_flat, (size_t)size.x * size.z * numflats);
        opt::CPUToGPU<float>(dark, d_dark, (size_t)size.x * size.z           );

		for(i = 0; i < nblock; i++) {

			subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames, (size_t)subblock * size.x * size.y);

            BackgroundCorrection_slices<<<gridBlock,threadsPerBlock>>>(d_frames,
                                                                       d_dark + (size_t)ptr * size.x,
                                                                       d_flat + (size_t)ptr * size.x * numflats, 
                                                                       dim3(size.x,size.y,subblock),
                                                                       numflats,
                                                                       is_log);

            opt::GPUToCPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames, (size_t)subblock * size.x * size.y);

			/* Update pointer */
			ptr = ptr + subblock;
        }
        HANDLE_ERROR(cudaFree(d_frames));
        HANDLE_ERROR(cudaFree(d_flat  ));
        HANDLE_ERROR(cudaFree(d_dark  ));
        HANDLE_ERROR(cudaDeviceSynchronize());
	}

    void getBackgroundCorrectionGPU_framesStreams(int gpu, float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log, int blocksize, const int nstreams)
    {
        // Supports 2 flats max
        HANDLE_ERROR(cudaSetDevice(gpu));

        int i;
        size_t total_required_mem_per_slice_bytes = (static_cast<float>(sizeof(float)) * ( size.x * size.y            ) + // Raw data sinogram
                                                     static_cast<float>(sizeof(float)) * ( size.x * size.y * numflats ) + // Flat line
                                                     static_cast<float>(sizeof(float)) * ( size.x * size.y            )   // Dark line
                                                    );
        total_required_mem_per_slice_bytes *= nstreams;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(size.z,
                                                       total_required_mem_per_slice_bytes, 
                                                       true, 
                                                       BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(size.z, blocksize_aux);
            blocksize          = min(32, blocksize); /* Set up a maximum blocksize for now */
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

            d_frames[st] = opt::allocGPU<float>((size_t) size.x * size.y * blocksize, streams[st]);
            d_flat[st]   = opt::allocGPU<float>((size_t) size.x * size.y *  numflats, streams[st]);
            d_dark[st]   = opt::allocGPU<float>((size_t) size.x * size.y            , streams[st]);

            opt::CPUToGPU<float>(flat, d_flat[st], (size_t)size.x * size.y * numflats, streams[st]);
            opt::CPUToGPU<float>(dark, d_dark[st], (size_t)size.x * size.y           , streams[st]);
        }

        for(i = 0; i < nblock; i++) {
            int st = i % nstreams;
            cudaStream_t stream = streams[i % nstreams];
            
            subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames[st], (size_t)subblock * size.x * size.y, stream);

            BackgroundCorrection_frames<<<gridBlock,threadsPerBlock, 0, stream>>>(d_frames[st],
                                                                                  d_dark[st],
                                                                                  d_flat[st], 
                                                                                  dim3(size.x,size.y,subblock),
                                                                                  numflats,
                                                                                  is_log);

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

    void getBackgroundCorrectionGPU_frames(int gpu, float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log, int blocksize)
    {
        // Supports 2 flats max
        HANDLE_ERROR(cudaSetDevice(gpu));

        int i;
        size_t total_required_mem_per_slice_bytes = (static_cast<float>(sizeof(float)) * ( size.x * size.y            ) + // Raw data sinogram
                                                     static_cast<float>(sizeof(float)) * ( size.x * size.y * numflats ) + // Flat line
                                                     static_cast<float>(sizeof(float)) * ( size.x * size.y            )   // Dark line
                                                    );

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(size.z,
                                                        total_required_mem_per_slice_bytes, 
                                                        true, 
                                                        BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(size.z, blocksize_aux);
            blocksize          = min(32, blocksize); /* Set up a maximum blocksize for now */
        }
        int nblock = (int)ceil( (float) size.z / blocksize );
        int ptr = 0, subblock;

        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock = opt::setGridBlock(size, threadsPerBlock);


        float *d_frames = opt::allocGPU<float>((size_t) size.x * size.y * blocksize);
        float *d_flat   = opt::allocGPU<float>((size_t) size.x * size.y *  numflats);
        float *d_dark   = opt::allocGPU<float>((size_t) size.x * size.y            );

        opt::CPUToGPU<float>(flat, d_flat, (size_t)size.x * size.y * numflats);
        opt::CPUToGPU<float>(dark, d_dark, (size_t)size.x * size.y           );

        for(i = 0; i < nblock; i++) {

            subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames, (size_t)subblock * size.x * size.y);

            BackgroundCorrection_frames<<<gridBlock,threadsPerBlock>>>(d_frames,
                                                                       d_dark,
                                                                       d_flat, 
                                                                       dim3(size.x,size.y,subblock),
                                                                       numflats,
                                                                       is_log);

            opt::GPUToCPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames, (size_t)subblock * size.x * size.y);

            /* Update pointer */
            ptr = ptr + subblock;
        }
        HANDLE_ERROR(cudaFree(d_frames));
        HANDLE_ERROR(cudaFree(d_flat  ));
        HANDLE_ERROR(cudaFree(d_dark  ));
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

	void getBackgroundCorrectionMultiGPU(int* gpus, int ngpus,
    float* frames, float* flat, float* dark,
    int sizex, int sizey, int sizez, int numflats,
    int is_log, int order, int blocksize, const int nstreams)
	{
        /* 
        frames: tomogram volume with axis (sizex, sizey, sizez):

            1. If order = 1 (True), then the last axis represent the slices
                (sizex, sizey, sizez) = (nrays, nangles, nslices)

            2. If order = 0 (False), then the last axis represent the angles
                (sizex, sizey, sizez) = (nrays, nslices, nangles)
        
        flat: Axis ALWAYS (nrays, nslices, numflats)
        dark: Axis ALWAYS (nrays, nslices, 1)
        */
		int i;
		int blockgpu = (sizez + ngpus - 1) / ngpus;
		int ptr = 0, subblock;

		std::vector<std::future<void>> threads;
        threads.reserve(ngpus);

        if ( ( order == SLICES_ANGLES_RAYS ) && ( nstreams == 0 ) ){

            for (i = 0; i < ngpus; i++) {
                subblock = min(sizez - ptr, blockgpu);

                threads.push_back(std::async( std::launch::async,
                    getBackgroundCorrectionGPU_slices,
                    gpus[i],
                    frames + (size_t)ptr * sizex * sizey,
                    flat   + (size_t)ptr * sizex * numflats,
                    dark   + (size_t)ptr * sizex,
                    dim3(sizex, sizey, subblock),
                    numflats, is_log, blocksize
                    ));

                /* Update pointer */
                ptr = ptr + subblock;
            }

            for(auto& t : threads) t.get();

        }else if ( ( order == SLICES_ANGLES_RAYS ) && ( nstreams > 0 ) ){

            for (i = 0; i < ngpus; i++) {
                subblock = min(sizez - ptr, blockgpu);

                threads.push_back(std::async( std::launch::async,
                    getBackgroundCorrectionGPU_slicesStreams,
                    gpus[i],
                    frames + (size_t)ptr * sizex * sizey,
                    flat   + (size_t)ptr * sizex * numflats,
                    dark   + (size_t)ptr * sizex,
                    dim3(sizex, sizey, subblock),
                    numflats, is_log, blocksize, nstreams
                    ));

                /* Update pointer */
                ptr = ptr + subblock;
            }

            for(auto& t : threads) t.get();

        }else if ( ( order == ANGLES_SLICES_RAYS ) && ( nstreams == 0 ) ){

            for (i = 0; i < ngpus; i++) {
                subblock = min(sizez - ptr, blockgpu);

                threads.push_back(std::async( std::launch::async,
                    getBackgroundCorrectionGPU_frames,
                    gpus[i],
                    frames + (size_t)ptr * sizex * sizey,
                    flat,
                    dark,
                    dim3(sizex, sizey, subblock),
                    numflats, is_log, blocksize
                    ));

                /* Update pointer */
                ptr = ptr + subblock;
            }

            for(auto& t : threads) t.get();

        }else if ( ( order == ANGLES_SLICES_RAYS ) && ( nstreams > 0 ) ){

            for (i = 0; i < ngpus; i++) {
                subblock = min(sizez - ptr, blockgpu);

                threads.push_back(std::async( std::launch::async,
                    getBackgroundCorrectionGPU_framesStreams,
                    gpus[i],
                    frames + (size_t)ptr * sizex * sizey,
                    flat,
                    dark,
                    dim3(sizex, sizey, subblock),
                    numflats, is_log, blocksize, nstreams
                    ));

                /* Update pointer */
                ptr = ptr + subblock;
            }

            for(auto& t : threads) t.get();
        
        }else{
            printf("Input data axis order is not set. Finishin run...\n");
        }
	}
}

