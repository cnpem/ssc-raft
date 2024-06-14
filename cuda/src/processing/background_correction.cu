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
    dim3 size, int numflats, int is_log)
	{
		// Supports 2 flats max
		HANDLE_ERROR(cudaSetDevice(gpu));

		int i;
		int blocksize = min(size.z, 32); // Herança do Giovanni -> Mudar

		int nblock = (int)ceil( (float) size.z / blocksize );
		int ptr = 0, subblock;

        float *d_flat   = opt::allocGPU<float>((size_t) size.z * size.x * numflats * blocksize);
        float *d_dark   = opt::allocGPU<float>((size_t) size.z * size.x * blocksize);

        opt::CPUToGPU<float>(flat, d_flat, (size_t)size.z * size.x * numflats);
        opt::CPUToGPU<float>(dark, d_dark, (size_t)size.z * size.x);

        const size_t nstreams = 2;
        cudaStream_t streams[nstreams];

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamCreate(&streams[st]);
        }

		for(i = 0; i < nblock; i++) {
            cudaStream_t stream = streams[i % nstreams];

            // GPUs Pointers: declaration and allocation
            float *d_frames = opt::allocGPU<float>((size_t) size.x *   size.y * blocksize, stream);

			subblock = min(size.z - ptr, blocksize);

            opt::CPUToGPU<float>(frames + (size_t)ptr * size.x *   size.y, d_frames, (size_t)subblock * size.x *   size.y, stream);

			getBackgroundCorrection(gpus, d_frames,
                    d_flat + (size_t)ptr * size.x * numflats,
                    d_dark + (size_t)ptr * size.x,
                    dim3(size.x, size.y, subblock), numflats, stream);

            if (is_log == 1)
                getLog(d_frames, dim3(size.x, size.y, subblock), stream);
            opt::GPUToCPU<float>(frames + (size_t)ptr * size.x * size.y, d_frames, (size_t)subblock * size.x * size.y, stream);

			/* Update pointer */
			ptr = ptr + subblock;

            HANDLE_ERROR(cudaFreeAsync(d_frames, stream));
        }

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamSynchronize(streams[st]);
            cudaStreamDestroy(streams[st]);
        }

        HANDLE_ERROR(cudaFree(d_flat));
        HANDLE_ERROR(cudaFree(d_dark));


        HANDLE_ERROR(cudaDeviceSynchronize());

	}

    void __getBackgroundCorrectionMultiGPU(int* gpus, int ngpus,
            float* frames, float* flat, float* dark,
            int nrays, int nangles, int nslices, int numflats,
            int is_log) {
        const size_t sizez = nslices;
        const size_t sizey = nangles;
        const size_t sizex = nrays;

        #pragma omp parallel for simd collapse(3)
        for (size_t idz = 0; idz < sizez; ++idz) {
            for (size_t idy = 0; idy < sizey; ++idy) {
                for (size_t idx = 0; idx < sizex; ++idx) {

                    const float flat1 = flat[sizex * numflats * idz + idx];
                    const float dk = dark[sizex * idz + idx];

                    float ft;
                    if(numflats > 1) {
                        const float interp = float(idy) / float(sizey);
                        const float flat2 = flat[sizex * numflats * idz + sizex + idx];
                        ft = flat1 * ( 1.0f - interp ) + interp * flat2;
                    }else{
                        ft = flat1;
                    }

                    const size_t line = sizex * sizey * idz + sizex * idy + idx;
                    const float T = frames[line] - dk;
                    const float Q = ft - dk;

                    frames[line] = fmaxf(T, 0.5f) / fmaxf(Q,0.5f);
                }
            }
        }
    }

	void getBackgroundCorrectionMultiGPU(int* gpus, int ngpus,
    float* frames, float* flat, float* dark,
    int nrays, int nangles, int nslices, int numflats,
    int is_log)
	{
		int i;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		int ptr = 0, subblock;

        float *h_frames;
        float *h_flat;
        float *h_dark;


        //HANDLE_ERROR(cudaHostRegister(h_frame_align, sizeof(float) * size_t(nrays * nangles * nslices), cudaHostRegisterDefault));
        //HANDLE_ERROR(cudaHostRegister(h_flat_align, sizeof(float) * size_t(nrays * numflats * nslices), cudaHostRegisterDefault));
        //HANDLE_ERROR(cudaHostRegister(h_dark_align, sizeof(float) * size_t(nrays * 1 * nslices), cudaHostRegisterDefault));

        //HANDLE_ERROR(cudaMallocHost(&h_frames, sizeof(float) * size_t(nrays) * size_t(nangles) * size_t(nslices)));
        //HANDLE_ERROR(cudaMallocHost(&h_flat, sizeof(float) * nslices * numflats * nslices));
        //HANDLE_ERROR(cudaMallocHost(&h_dark, sizeof(float) * nslices * 1 * nslices));

        //HANDLE_ERROR(cudaMemcpy(h_frames, frames, sizeof(float) * size_t(nrays) * size_t(nangles) * size_t(nslices), cudaMemcpyHostToHost));
        //HANDLE_ERROR(cudaMemcpy(h_flat, flat, sizeof(float) * nslices * numflats * nslices, cudaMemcpyHostToHost));
        //HANDLE_ERROR(cudaMemcpy(h_dark, dark, sizeof(float) * nslices * 1 * nslices, cudaMemcpyHostToHost));


        //printf(">>> pinned allocs: %p %p %p\n", h_frames, h_flat, h_dark);

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
                numflats, is_log
                ));

            /* Update pointer */
            ptr = ptr + subblock;
        }

        for(auto& t : threads)
            t.get();

        //HANDLE_ERROR(cudaMemcpy(frames, h_frames, sizeof(float) * nrays * nangles * nslices, cudaMemcpyHostToHost));
        //HANDLE_ERROR(cudaMemcpy(flat, h_flat, sizeof(float) * nslices * numflats * nslices, cudaMemcpyHostToHost));
        //HANDLE_ERROR(cudaMemcpy(dark, h_dark, sizeof(float) * nslices * 1 * nslices, cudaMemcpyHostToHost));

        //HANDLE_ERROR(cudaFreeHost(h_frames));
        //HANDLE_ERROR(cudaFreeHost(h_flat));
        //HANDLE_ERROR(cudaFreeHost(h_dark));
	}
}

