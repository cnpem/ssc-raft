#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cstddef>
#include "common/logerror.hpp"
#include "common/operations.hpp"
#include "common/opt.hpp"
#include "common/types.hpp"
#include "processing/processing.hpp"

template <bool bShared>
__global__ void KConvolve0(float *restrict image0, const float *kernel, size_t sizex, int hkernelsize, float *globalmem)
{
    extern __shared__ float sharedmem[];
    float *restrict intermediate = bShared ? sharedmem : (globalmem + (sizex + 2 * PADDING) * blockIdx.x);

    if (threadIdx.x < PADDING)
    {
        intermediate[threadIdx.x] = image0[blockIdx.x * sizex];
        intermediate[sizex + 2 * PADDING - 1 - threadIdx.x] = image0[blockIdx.x * sizex + sizex - 1];
    }
    for (int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
        intermediate[PADDING + idx] = image0[blockIdx.x * sizex + idx];

    __syncthreads();

    for (int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
    {
        float accum = 0;

        for (int t = -hkernelsize; t <= hkernelsize; t++)
            accum += kernel[t + hkernelsize] * intermediate[PADDING + idx + t];

        image0[blockIdx.x * sizex + idx] = -accum;
    }
}

extern "C"
{
    void Convolve0(GArray<float> img, GArray<float> kernel, GArray<float> globalmem, cudaStream_t stream = 0)
    {
        int khsize = (kernel.shape.x * kernel.shape.y * kernel.shape.z) / 2;
        size_t sizex = img.shape.x;

        dim3 threads = dim3(img.shape.x > 1024 ? 1024 : img.shape.x, 1, 1);
        dim3 blocks = dim3(img.shape.y, 1, 1);

        size_t shbytes = (img.shape.x + 2 * PADDING) * sizeof(float);

        if (shbytes > 18000)
            KConvolve0<false><<<blocks, threads, 0, stream>>>(img.ptr, kernel.ptr, sizex, khsize, globalmem.ptr);
        else
            KConvolve0<true><<<blocks, threads, shbytes, stream>>>(img.ptr, kernel.ptr, sizex, khsize, nullptr);

        HANDLE_ERROR(cudaGetLastError());
    }
}

template <bool bShared>
__global__ void KConvolve(float *restrict x, float *restrict p, const float *kernel, size_t sizex, int hkernelsize, float lambda, float alpha, int numiterations, float *restrict residuum2, float *restrict momentum, float beta, float *globalmem)
{
    extern __shared__ float sharedmem[];
    float *restrict padded_intermediate = bShared ? sharedmem : (globalmem + (sizex + 128) * blockIdx.x);

    float *restrict intermediate = padded_intermediate + PADDING;
    float *restrict vel = momentum + 2 * blockIdx.x * sizex;

    __syncthreads();

    for (int iter = 0; iter < numiterations; iter++)
    {
        if (threadIdx.x < PADDING)
        {
            padded_intermediate[threadIdx.x] = p[blockIdx.x * sizex];
            padded_intermediate[sizex + 2 * PADDING - 1 - threadIdx.x] = p[blockIdx.x * sizex + sizex - 1];
        }
        for (int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
            intermediate[idx] = p[blockIdx.x * sizex + idx];

        __syncthreads();

        for (int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
        {
            float accum = lambda * intermediate[idx];

            for (int t = -hkernelsize; t <= hkernelsize; t++)
                accum += kernel[t + hkernelsize] * intermediate[idx + t];

            float velocity = beta * vel[idx] + alpha * intermediate[idx];
            vel[idx] = velocity;

            float pvelocity = beta * vel[idx + sizex] - alpha * accum;
            vel[idx + sizex] = pvelocity;

            x[blockIdx.x * sizex + idx] += velocity;
            p[blockIdx.x * sizex + idx] += pvelocity;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
        intermediate[idx] *= intermediate[idx];
    __syncthreads();

    for (int group = sizex / 2; group > 32; group /= 2)
    {
        for (int idx = threadIdx.x; idx < group; idx += blockDim.x)
            intermediate[idx] += intermediate[idx + group];
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        for (int group = 32; group > 0; group /= 2)
        {
            if (threadIdx.x < group)
                intermediate[threadIdx.x] += intermediate[threadIdx.x + group];
            __syncwarp();
        }
    }

    if (threadIdx.x == 0)
        residuum2[blockIdx.x] = intermediate[0];
}

extern "C"
{
    void Convolve(GArray<float> x, GArray<float> p, GArray<float> kernel, float lambda, float alpha, int iter, GArray<float> res2, GArray<float> momentum, float beta, GArray<float> globalmem, cudaStream_t stream = 0)
    {
        int khsize = (kernel.shape.x * kernel.shape.y * kernel.shape.z) / 2;

        dim3 threads = dim3(x.shape.x > 1024 ? 1024 : x.shape.x, 1, 1);
        dim3 blocks = dim3(x.shape.y, 1, 1);

        float *kern = kernel.ptr;
        size_t shbytes = (x.shape.x + 2 * PADDING) * sizeof(float);

        if (shbytes > 18000)
            KConvolve<false><<<blocks, threads, 0, stream>>>(x.ptr, p.ptr, kern, x.shape.x, khsize, lambda, alpha, iter, res2.ptr, momentum.ptr, beta, globalmem.ptr);
        else
            KConvolve<true><<<blocks, threads, shbytes, stream>>>(x.ptr, p.ptr, kern, x.shape.x, khsize, lambda, alpha, iter, res2.ptr, momentum.ptr, beta, nullptr);

        HANDLE_ERROR(cudaGetLastError());
    }

    __global__ void KRedVolume(float *out, const float *volume, size_t volsizex, size_t volsizey, size_t slicesize)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t plane = blockIdx.y * slicesize;

        if (idx < volsizex)
        {
            float accum = 0;
            for (size_t idy = 0; idy < volsizey; idy++)
                accum += volume[plane + idy * volsizex + idx];

            out[blockIdx.y * volsizex + idx] = accum / volsizey;
        }
    }

    __device__ int float_to_int(float f)
    {
        int i = __float_as_int(f);
        i ^= (i < 0) ? 0x7FFFFFFF : 0x0;
        return i;
    }

    __device__ float int_to_float(int i)
    {
        i ^= (i < 0) ? 0x7FFFFFFF : 0x0;
        float f = __int_as_float(i);
        return i;
    }

    __global__ void KRedVolumeMM(float *out, complex *fminmax, const float *volume, size_t volsizex, size_t volsizey, size_t slicesize)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t plane = blockIdx.y * slicesize;
        __shared__ int smax[32];
        __shared__ int smin[32];

        if (threadIdx.x < 32)
        {
            smin[threadIdx.x] = 0x7FFFFFFF;
            smax[threadIdx.x] = 0x80000000;
        }

        __syncthreads();

        if (idx < volsizex)
        {
            float accum = 0;
            for (size_t idy = 0; idy < volsizey; idy++)
                accum += volume[plane + idy * volsizex + idx];

            out[blockIdx.y * volsizex + idx] = accum / volsizey;

            int intaccum = float_to_int(accum);

            atomicMin(smin + threadIdx.x % 32, intaccum);
            atomicMax(smax + threadIdx.x % 32, intaccum);
        }

        __syncthreads();

        int tmax = 16;
        while (threadIdx.x < tmax)
        {
            smin[threadIdx.x] = min(smin[threadIdx.x], smin[threadIdx.x + tmax]);
            smax[threadIdx.x] = max(smax[threadIdx.x], smax[threadIdx.x + tmax]);
            __syncwarp((1 << tmax) - 1);
            tmax /= 2;
        }
        if (threadIdx.x == 0)
            fminmax[blockIdx.x + gridDim.x * blockIdx.y] = complex(int_to_float(smin[0]), int_to_float(smax[0]));
    }

    float VolumeAverage(float *mbar, float *volume,
            size_t sizex, size_t sizey, size_t sizez,
            float lambda, size_t slicesize, cudaStream_t stream = 0)
    {
        dim3 threads = dim3(min((int)sizex, 256), 1, 1);
        dim3 blocks = dim3((sizex + 255) / 256, sizez, 1);

        if (lambda < 0)
        {
            cImage minmax(blocks.x * blocks.y, 1, 1, EAllocCPUGPU, stream);

            KRedVolumeMM<<<blocks, threads, 0, stream>>>(mbar, minmax.gpuptr, volume, sizex, sizey, slicesize);

            minmax.LoadFromGPU(stream);

            //cudaDeviceSynchronize();

            cudaStreamSynchronize(stream);

            float vmin = +1E-10;
            float vmax = -1E-10;

            for (size_t i = 0; i < minmax.GetSize(); i++)
            {
                vmin = fminf(vmin, minmax.cpuptr[i].x);
                vmax = fmaxf(vmax, minmax.cpuptr[i].y);
            }

            minmax.DeallocGPU(stream);

            HANDLE_ERROR(cudaGetLastError());
            return 1.0f / (vmax - vmin);
        }
        else
        {
            KRedVolume<<<blocks, threads, 0, stream>>>(mbar, volume, sizex, sizey, slicesize);

            HANDLE_ERROR(cudaGetLastError());
            return lambda;
        }
    }

    __global__ void KApplyTitarenkoRings(float *volume, const float *nstaravg,
            size_t volsizex, size_t volsizey, size_t slicesize)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t plane = blockIdx.y * slicesize;

        if (idx < volsizex){
            const float nstar = nstaravg[blockIdx.y * volsizex + idx];

            for (size_t idy = 0; idy < volsizey; idy++){
                // volume[plane + idy*volsizex + idx] = -sqrtf(fmaxf(est1 * est2, 0.0f));
                volume[plane + idy * volsizex + idx] += nstar;
            }
        }
    }

    void ApplyTitarenkoRings(float *volume, float *nstaravg,
            int sizex, int sizey, int sizez, size_t slicesize, cudaStream_t stream = 0)
    {
        dim3 threads = dim3(min((int)sizex, 128), 1, 1);
        dim3 blocks = dim3((sizex + 127) / 128, sizez, 1);
        KApplyTitarenkoRings<<<blocks, threads, 0, stream>>>(volume, nstaravg, sizex, sizey, slicesize);

        HANDLE_ERROR(cudaGetLastError());
    }

    float CalcAlpha(float *kern, size_t kernsize, float lambda)
    {
        float alpha = lambda + 1E-10f;
        for (size_t i = 0; i < kernsize; i++)
            alpha += fabsf(kern[i]);

        alpha = 1.75f / (sqrt(alpha) + sqrtf(lambda));
        alpha *= alpha;

        return alpha;
    }

    float CalcBeta(float *kern, size_t kernsize, float lambda)
    {
        float beta = lambda + 1E-10f;
        for (size_t i = 0; i < kernsize; i++)
            beta += fabsf(kern[i]);

        beta = (sqrtf(beta) - sqrtf(lambda)) / (sqrtf(beta) + sqrtf(lambda));
        beta = beta * beta;

        return 0.95f * beta;
    }

    void CalcNorm2(float *out, const float *in, size_t msizex, size_t msizey)
    {
        for (size_t j = 0; j < msizey; j++)
        {
            out[j] = 1E-5f;
            for (size_t i = 0; i < msizex; i++)
                out[j] += in[j * msizex + i] * in[j * msizex + i];
            out[j] = 1.0f / out[j];
        }
    }

    __global__ void KAverageTitarenkoRings(float *out, const float *nvec1, const float *nvec2, size_t size)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < size)
            out[idx] = nvec1[idx] * 0.5f + nvec2[idx] * 0.5f;
    }

    //problem is here, just dont know exactly what it is
    void TitarenkoRingsFilter(GArray<float> sinobar, float lambda, const float *norm2vec,
            cudaStream_t stream = 0)
    {
        size_t msizex = sinobar.shape.x;
        size_t msizey = sinobar.shape.y;
        // const float fk1[] = {1.0f, -4.0f  ,  6.0f , -4.0f  ,  1.0f};
        float fk2[] = {-2.0f, 13.0f, -34.0f, 46.0f, -34.0f, 13.0f, -2.0f};

        float fk1[] = {-1.0f, 0.0f, 9.0f, 20.0f, 9.0f, 0.0f, -1.0f};
        /*const float fk2[] = {-1.55794444e+01f,  2.06396597e+02f, -1.27719611e+03f,  4.89847819e+03f,
            -1.30230239e+04f,  2.54103134e+04f, -3.75466717e+04f,  4.26945658e+04f,
            -3.75466717e+04f,  2.54103134e+04f, -1.30230239e+04f,  4.89847819e+03f,
            -1.27719611e+03f,  2.06396597e+02f, -1.55794444e+01f}; */

        rImage kernel1(fk1, sizeof(fk1) / sizeof(fk1[0]), 1, 1, MemoryType::EAllocCPUGPU, stream);
        rImage kernel2(fk2, sizeof(fk2) / sizeof(fk2[0]), 1, 1, MemoryType::EAllocCPUGPU, stream);

        rImage xvec1(msizex, msizey, 1, MemoryType::EAllocGPU, stream);
        rImage xvec2(msizex, msizey, 1, MemoryType::EAllocGPU, stream);
        rImage mvec1(msizex, msizey, 1, MemoryType::EAllocCPUGPU, stream);
        rImage mvec2(msizex, msizey, 1, MemoryType::EAllocGPU, stream);
        rImage momentum(msizex, 2 * msizey, 1, MemoryType::EAllocGPU, stream);
        rImage intermediate(msizex + 2 * PADDING, msizey, 1, MemoryType::EAllocGPU, stream);
        rImage residuum2(msizey, 1, 1, MemoryType::EAllocCPUGPU, stream);

        momentum.SetGPUToZero(stream);
        xvec1.SetGPUToZero(stream);
        xvec2.SetGPUToZero(stream);
        mvec1.CopyFrom(sinobar.ptr, stream);
        mvec2.CopyFrom(sinobar.ptr, stream);
        mvec1.LoadFromGPU(stream);

        cudaStreamSynchronize(stream);


        const float alpha1 = CalcAlpha(kernel1.cpuptr, kernel1.GetSize(), lambda);
        const float alpha2 = CalcAlpha(kernel2.cpuptr, kernel2.GetSize(), lambda);

        const float beta1 = CalcBeta(kernel1.cpuptr, kernel1.GetSize(), lambda);
        const float beta2 = CalcBeta(kernel2.cpuptr, kernel2.GetSize(), lambda);

        Convolve0(mvec1, kernel1, intermediate, stream); // mvec = FtF(mbar)
        Convolve0(mvec2, kernel2, intermediate, stream); // mvec = FtF(mbar)

        float maxerr = 1;
        while (maxerr > 1E-6f)
        {
            Convolve(xvec1, mvec1, kernel1, lambda, alpha1, 100, residuum2, momentum, beta1, intermediate, stream);

            residuum2.LoadFromGPU(stream);
            //HANDLE_ERROR(cudaDeviceSynchronize());
            cudaStreamSynchronize(stream);
            maxerr = 0;
            for (size_t j = 0; j < msizey; j++)
                maxerr = fmaxf(maxerr, residuum2.cpuptr[j] * norm2vec[j]);
        }

        momentum.SetGPUToZero(stream);
        maxerr = 1;

        while (maxerr > 1E-6f)
        {
            Convolve(xvec2, mvec2, kernel2, lambda, alpha2, 100, residuum2, momentum, beta2, intermediate, stream);

            residuum2.LoadFromGPU(stream);
            cudaStreamSynchronize(stream);
            maxerr = 0;
            for (size_t j = 0; j < msizey; j++)
                maxerr = fmaxf(maxerr, residuum2.cpuptr[j] * norm2vec[j]);
        }

        KAverageTitarenkoRings<<<(xvec1.GetSize() + 31) / 32, 32, 0, stream>>>(sinobar.ptr, xvec1.gpuptr, xvec2.gpuptr, xvec1.GetSize());

        kernel1.DeallocGPU(stream);
        kernel2.DeallocGPU(stream);
        xvec1.DeallocGPU(stream);
        xvec2.DeallocGPU(stream);
        mvec1.DeallocGPU(stream);
        mvec2.DeallocGPU(stream);
        momentum.DeallocGPU(stream);
        intermediate.DeallocGPU(stream);
        residuum2.DeallocGPU(stream);

        HANDLE_ERROR(cudaGetLastError());
    }

    // static __global__ void KDIV(float* divv, float* flat, dim3 shp)
    // {
    //     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    //     size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    //     size_t idz = blockIdx.z;

    //     if(idx < shp.x && idy < shp.y && idz < shp.z)
    //         divv[idx + idy*shp.x + idz*shp.x*shp.y] /= flat[idx + shp.x*idz];

    // }
    // static __global__ void KMUL(float* divv, float* flat, dim3 shp)
    // {
    //     size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    //     size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    //     size_t idz = blockIdx.z;

    //     if(idx < shp.x && idy < shp.y && idz < shp.z)
    //         divv[idx + idy*shp.x + idz*shp.x*shp.y] *= flat[idx + shp.x*idz];

    // }

    float TitarenkoRings(float *volume, int vsizex, int vsizey, int vsizez,
            float lambda, size_t slicesize, cudaStream_t stream = 0)
    {
        size_t msizex = vsizex;
        size_t msizey = vsizez;

        rImage smooth(msizex, msizey, 1, EAllocGPU, stream);
        rImage sinobar(msizex, msizey, 1, EAllocCPUGPU, stream);

        lambda = VolumeAverage(sinobar.gpuptr, volume,
                vsizex, vsizey, vsizez,
                lambda, slicesize, stream);

        sinobar.LoadFromGPU(stream);
        //HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaStreamSynchronize(stream));

        float norm2vec[msizey];

        CalcNorm2(norm2vec, sinobar.cpuptr, msizex, msizey);

        TitarenkoRingsFilter(sinobar, lambda, norm2vec, stream);

        ApplyTitarenkoRings(volume, sinobar.gpuptr,
                vsizex, vsizey, vsizez, slicesize, stream);

        smooth.DeallocGPU(stream);
        sinobar.DeallocGPU(stream);

        HANDLE_ERROR(cudaGetLastError());

        return lambda;
    }
}

extern "C"{

    void getTitarenkoRings(GPU gpus, float *tomogram, dim3 size,
            float lambda_rings, int ring_blocks, cudaStream_t stream)
    {

        /* Projection data sizes */
        int nrays        = size.x;
        int nangles      = size.y;
        int blockslices  = size.z;

        size_t step, offset = nrays * nangles;

        for (int m = 0; m < ring_blocks / 2; m++){

            TitarenkoRings(tomogram,
                                nrays, nangles, blockslices,
                                lambda_rings, offset, stream);
            step = (nangles / ring_blocks) * nrays;
            float *tomptr = tomogram;

            for (int n = 0; n < ring_blocks - 1; n++) {
                TitarenkoRings(tomogram,
                                    nrays, nangles, blockslices,
                                    lambda_rings, offset, stream);
                tomptr += step;
            }
            TitarenkoRings(tomptr,
                                nrays, nangles % ring_blocks + nangles / ring_blocks, blockslices,
                                lambda_rings, offset, stream);
        }

        HANDLE_ERROR(cudaGetLastError());
    }

    void getTitarenkoRingsGPU(GPU gpus, int gpu,
        float *data, dim3 size, 
        float lambda_rings, int ring_blocks,
        int blocksize)
    {
        HANDLE_ERROR(cudaSetDevice(gpu));

        const int nstreams = 3;

        /* Projection data sizes */
        int nrays    = size.x;
        int nangles  = size.y;
        int nslices  = size.z;

		int i;
        size_t total_required_mem_per_slice_bytes = static_cast<float>(sizeof(float)) * ( nrays * nangles ) * nstreams;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(nslices, total_required_mem_per_slice_bytes, 
                                                        true, BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(nslices, blocksize_aux);
            blocksize          = min(32, blocksize);
        }

        int nblock = (int)ceil( (float) nslices / blocksize );
        int ptr = 0, subblock;

        float *tomogram[nstreams];
        cudaStream_t streams[nstreams];

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamCreate(&streams[st]);

            tomogram[st] = opt::allocGPU<float>((size_t) nrays * nangles * blocksize, streams[st]);
        }

        for (i = 0; i < nblock; i++){
            int st = i % nstreams;
            cudaStream_t stream = streams[i % nstreams];

            subblock = min(nslices - ptr, blocksize);

            opt::CPUToGPU<float>(data + (size_t)ptr * nrays * nangles, tomogram[st], (size_t)subblock * nrays * nangles, stream);

            getTitarenkoRings(gpus, tomogram[st],
                                    dim3(nrays, nangles, subblock),
                                    lambda_rings, ring_blocks, stream);

            opt::GPUToCPU<float>(data + (size_t)ptr * nrays * nangles, tomogram[st], (size_t)subblock * nrays * nangles, stream);

            /* Update pointer */
            ptr = ptr + subblock;
        }

        for(int st = 0; st < nstreams; ++st) {
            cudaStreamSynchronize(streams[st]);

            HANDLE_ERROR(cudaFreeAsync(tomogram[st], streams[st]));

            cudaStreamDestroy(streams[st]);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());
    }
}

extern "C"{

    void getTitarenkoRingsMultiGPU(int *gpus, int ngpus,
    float *data, int nrays, int nangles, int nslices,
    float lambda_rings, int ring_blocks,
    int blocksize)
    {
        int i;
        int blockgpu = (nslices + ngpus - 1) / ngpus;
        int ptr = 0, subblock;

        GPU gpu_parameters;

        setGPUParameters(&gpu_parameters, dim3(nrays, nangles, nslices), ngpus, gpus);

		std::vector<std::future<void>> threads;
        threads.reserve(ngpus);

        for (i = 0; i < ngpus; i++){

            subblock = min(nslices - ptr, blockgpu);

            threads.push_back(std::async(std::launch::async,
                getTitarenkoRingsGPU,
                gpu_parameters,
                gpus[i],
                data + (size_t)ptr * nrays * nangles,
                dim3(nrays, nangles, subblock),
                lambda_rings, ring_blocks,
                blocksize));

            /* Update pointer */
            ptr = ptr + subblock;
        }

        for (auto &t : threads)
            t.get();

    }
}
