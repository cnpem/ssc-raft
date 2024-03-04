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
    void Convolve0(rImage &img, rImage &kernel, rImage &globalmem)
    {
        int khsize = kernel.GetSize() / 2;
        size_t sizex = img.sizex;

        dim3 threads = dim3(img.sizex > 1024 ? 1024 : img.sizex, 1, 1);
        dim3 blocks = dim3(img.sizey, 1, 1);

        size_t shbytes = (img.sizex + 2 * PADDING) * sizeof(float);

        if (shbytes > 18000)
            KConvolve0<false><<<blocks, threads, 0>>>(img.gpuptr, kernel.gpuptr, sizex, khsize, globalmem.gpuptr);
        else
            KConvolve0<true><<<blocks, threads, shbytes>>>(img.gpuptr, kernel.gpuptr, sizex, khsize, nullptr);

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
    void Convolve(rImage &x, rImage &p, rImage &kernel, float lambda, float alpha, int iter, rImage &res2, rImage &momentum, float beta, rImage &globalmem)
    {
        int khsize = kernel.GetSize() / 2;

        dim3 threads = dim3(x.sizex > 1024 ? 1024 : x.sizex, 1, 1);
        dim3 blocks = dim3(x.sizey, 1, 1);

        float *kern = kernel.gpuptr;
        size_t shbytes = (x.sizex + 2 * PADDING) * sizeof(float);

        if (shbytes > 18000)
            KConvolve<false><<<blocks, threads, 0>>>(x.gpuptr, p.gpuptr, kern, x.sizex, khsize, lambda, alpha, iter, res2.gpuptr, momentum.gpuptr, beta, globalmem.gpuptr);
        else
            KConvolve<true><<<blocks, threads, shbytes>>>(x.gpuptr, p.gpuptr, kern, x.sizex, khsize, lambda, alpha, iter, res2.gpuptr, momentum.gpuptr, beta, nullptr);

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

    float VolumeAverage(float *mbar, float *volume, size_t sizex, size_t sizey, size_t sizez, float lambda, size_t slicesize)
    {
        dim3 threads = dim3(min((int)sizex, 256), 1, 1);
        dim3 blocks = dim3((sizex + 255) / 256, sizez, 1);

        if (lambda < 0)
        {
            cImage minmax(blocks.x * blocks.y, 1);

            KRedVolumeMM<<<blocks, threads>>>(mbar, minmax.gpuptr, volume, sizex, sizey, slicesize);

            minmax.LoadFromGPU();

            cudaDeviceSynchronize();

            float vmin = +1E-10;
            float vmax = -1E-10;

            for (size_t i = 0; i < minmax.GetSize(); i++)
            {
                vmin = fminf(vmin, minmax.cpuptr[i].x);
                vmax = fmaxf(vmax, minmax.cpuptr[i].y);
            }

            HANDLE_ERROR(cudaGetLastError());
            return 1.0f / (vmax - vmin);
        }
        else
        {
            KRedVolume<<<blocks, threads>>>(mbar, volume, sizex, sizey, slicesize);

            HANDLE_ERROR(cudaGetLastError());
            return lambda;
        }
    }

    __global__ void KApplyTitarenkoRings(float *volume, const float *nstaravg, size_t volsizex, size_t volsizey, size_t slicesize)
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

    void ApplyTitarenkoRings(float *volume, float *nstaravg, int sizex, int sizey, int sizez, size_t slicesize)
    {
        dim3 threads = dim3(min((int)sizex, 128), 1, 1);
        dim3 blocks = dim3((sizex + 127) / 128, sizez, 1);
        KApplyTitarenkoRings<<<blocks, threads>>>(volume, nstaravg, sizex, sizey, slicesize);

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

    void TitarenkoRingsFilter(rImage &sinobar, float lambda, const float *norm2vec)
    {
        size_t msizex = sinobar.sizex;
        size_t msizey = sinobar.sizey;
        // const float fk1[] = {1.0f, -4.0f  ,  6.0f , -4.0f  ,  1.0f};
        float fk2[] = {-2.0f, 13.0f, -34.0f, 46.0f, -34.0f, 13.0f, -2.0f};

        float fk1[] = {-1.0f, 0.0f, 9.0f, 20.0f, 9.0f, 0.0f, -1.0f};
        /*const float fk2[] = {-1.55794444e+01f,  2.06396597e+02f, -1.27719611e+03f,  4.89847819e+03f,
            -1.30230239e+04f,  2.54103134e+04f, -3.75466717e+04f,  4.26945658e+04f,
            -3.75466717e+04f,  2.54103134e+04f, -1.30230239e+04f,  4.89847819e+03f,
            -1.27719611e+03f,  2.06396597e+02f, -1.55794444e+01f}; */

        rImage kernel1(fk1, sizeof(fk1) / sizeof(fk1[0]), 1);
        rImage kernel2(fk2, sizeof(fk2) / sizeof(fk2[0]), 1);

        rImage xvec1(msizex, msizey);
        rImage xvec2(msizex, msizey);
        rImage mvec1(msizex, msizey);
        rImage mvec2(msizex, msizey);
        rImage momentum(msizex, 2 * msizey);
        rImage intermediate(msizex + 2 * PADDING, msizey);

        momentum.SetGPUToZero();
        xvec1.SetGPUToZero();
        xvec2.SetGPUToZero();
        mvec1.CopyFrom(sinobar);
        mvec2.CopyFrom(sinobar);
        mvec1.LoadFromGPU();
        HANDLE_ERROR(cudaDeviceSynchronize());

        rImage residuum2(msizey, 1);

        const float alpha1 = CalcAlpha(kernel1.cpuptr, kernel1.GetSize(), lambda);
        const float alpha2 = CalcAlpha(kernel2.cpuptr, kernel2.GetSize(), lambda);

        const float beta1 = CalcBeta(kernel1.cpuptr, kernel1.GetSize(), lambda);
        const float beta2 = CalcBeta(kernel2.cpuptr, kernel2.GetSize(), lambda);

        Convolve0(mvec1, kernel1, intermediate); // mvec = FtF(mbar)
        Convolve0(mvec2, kernel2, intermediate); // mvec = FtF(mbar)

        float maxerr = 1;
        while (maxerr > 1E-6f)
        {
            Convolve(xvec1, mvec1, kernel1, lambda, alpha1, 100, residuum2, momentum, beta1, intermediate);

            residuum2.LoadFromGPU();
            HANDLE_ERROR(cudaDeviceSynchronize());
            maxerr = 0;
            for (size_t j = 0; j < msizey; j++)
                maxerr = fmaxf(maxerr, residuum2.cpuptr[j] * norm2vec[j]);
        }

        momentum.SetGPUToZero();
        maxerr = 1;

        while (maxerr > 1E-6f)
        {
            Convolve(xvec2, mvec2, kernel2, lambda, alpha2, 100, residuum2, momentum, beta2, intermediate);

            residuum2.LoadFromGPU();
            HANDLE_ERROR(cudaDeviceSynchronize());
            maxerr = 0;
            for (size_t j = 0; j < msizey; j++)
                maxerr = fmaxf(maxerr, residuum2.cpuptr[j] * norm2vec[j]);
        }

        KAverageTitarenkoRings<<<(xvec1.GetSize() + 31) / 32, 32>>>(sinobar.gpuptr, xvec1.gpuptr, xvec2.gpuptr, xvec1.GetSize());
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

    float TitarenkoRings(float *volume, int vsizex, int vsizey, int vsizez, float lambda, size_t slicesize)
    {
        size_t msizex = vsizex;
        size_t msizey = vsizez;

        rImage smooth(msizex, msizey);
        rImage sinobar(msizex, msizey);

        lambda = VolumeAverage(sinobar.gpuptr, volume, vsizex, vsizey, vsizez, lambda, slicesize);

        sinobar.LoadFromGPU();
        HANDLE_ERROR(cudaDeviceSynchronize());

        float norm2vec[msizey];

        CalcNorm2(norm2vec, sinobar.cpuptr, msizex, msizey);

        TitarenkoRingsFilter(sinobar, lambda, norm2vec);

        ApplyTitarenkoRings(volume, sinobar.gpuptr, vsizex, vsizey, vsizez, slicesize);
        HANDLE_ERROR(cudaGetLastError());

        return lambda;
    }
}

extern "C"{

    void getTitarenkoRings(GPU gpus, float *tomogram, dim3 size, float lambda_rings, int ring_blocks)
    {
        float lambda_computed;

        /* Projection data sizes */
        int nrays        = size.x;
        int nangles      = size.y;
        int blockslices  = size.z;

        size_t step, offset = nrays * nangles;

        for (int m = 0; m < ring_blocks / 2; m++){

            lambda_computed = TitarenkoRings(tomogram, 
                                nrays, nangles, blockslices, 
                                lambda_rings, offset);
            
            step = (nangles / ring_blocks) * nrays;
            float *tomptr = tomogram;

            for (int n = 0; n < ring_blocks - 1; n++){
                lambda_computed = TitarenkoRings(tomogram, 
                                    nrays, nangles, blockslices, 
                                    lambda_rings, offset);
                tomptr += step;
            }
            lambda_computed = TitarenkoRings(tomptr, 
                                nrays, nangles % ring_blocks + nangles / ring_blocks, blockslices, 
                                lambda_rings, offset);
        }

        HANDLE_ERROR(cudaGetLastError());
    }

    void getTitarenkoRingsGPU(GPU gpus, int gpu, 
    float *data, dim3 size, float lambda_rings, int ring_blocks)
    {
        HANDLE_ERROR(cudaSetDevice(gpu));

        /* Projection data sizes */
        int nrays    = size.x;
        int nangles  = size.y;
        int nslices  = size.z;

        int i;
        int blocksize = min(nslices, 32); // Herança do Giovanni -> Mudar

        int nblock = (int)ceil((float)nslices / blocksize);
        int ptr = 0, subblock;

        float *tomogram = opt::allocGPU<float>((size_t) nrays * nangles * blocksize);

        for (i = 0; i < nblock; i++){
            
            subblock = min(nslices - ptr, blocksize);

            opt::CPUToGPU<float>(data + (size_t)ptr * nrays * nangles, tomogram, (size_t)subblock * nrays * nangles);

            getTitarenkoRings(gpus, tomogram, 
                                    dim3(nrays, nangles, subblock), 
                                    lambda_rings, ring_blocks);
            
            opt::GPUToCPU<float>(data + (size_t)ptr * nrays * nangles, tomogram, (size_t)subblock * nrays * nangles);

            /* Update pointer */
            ptr = ptr + subblock;
        }
        HANDLE_ERROR(cudaFree(tomogram));
        HANDLE_ERROR(cudaDeviceSynchronize());    
    }
}

extern "C"{

    void getTitarenkoRingsMultiGPU(int *gpus, int ngpus, 
    float *data, int nrays, int nangles, int nslices, 
    float lambda_rings, int ring_blocks)
    {
        int i;
        int blockgpu = (nslices + ngpus - 1) / ngpus;
        int ptr = 0, subblock;

        GPU gpu_parameters;

        setGPUParameters(&gpu_parameters, dim3(nrays, nangles, nslices), ngpus, gpus);

		std::vector<std::future<void>> threads;
        threads.reserve(ngpus);

        if (ngpus == 1){
            getTitarenkoRingsGPU(gpu_parameters, 
                gpus[0], 
                data, 
                dim3(nrays, nangles, nslices), 
                lambda_rings, ring_blocks);
        }else{
            for (i = 0; i < ngpus; i++){

                subblock = min(nslices - ptr, blockgpu);

                threads.push_back(std::async(std::launch::async,
                    getTitarenkoRingsGPU,
                    gpu_parameters,
                    gpus[i],
                    data + (size_t)ptr * nrays * nangles,
                    dim3(nrays, nangles, subblock),
                    lambda_rings, ring_blocks));

                /* Update pointer */
                ptr = ptr + subblock;
            }
        }

        for (auto &t : threads)
            t.get();
    }
}
