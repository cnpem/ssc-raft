#include "../../../../inc/include.h"
#include "../../../../inc/common/kernel_operators.hpp"
#include "../../../../inc/common/complex.hpp"
#include "../../../../inc/common/types.hpp"
#include "../../../../inc/common/operations.hpp"
#include "../../../../inc/common/logerror.hpp"


template<bool bShared>
__global__ void KConvolve0(float* restrict image0, const float* kernel, size_t sizex, int hkernelsize, float* globalmem)
{
    extern __shared__ float sharedmem[];
    float* restrict intermediate = bShared ? sharedmem : (globalmem + (sizex+2*PADDING)*blockIdx.x);

    if(threadIdx.x < PADDING)
    {
        intermediate[threadIdx.x] = image0[blockIdx.x * sizex];
        intermediate[sizex+2*PADDING-1-threadIdx.x] = image0[blockIdx.x * sizex + sizex-1];
    }
    for(int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
        intermediate[PADDING+idx] = image0[blockIdx.x * sizex + idx];
        
    __syncthreads();
    
    for(int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
    {
        float accum = 0;
        
        for(int t=-hkernelsize; t<=hkernelsize; t++)
            accum += kernel[t+hkernelsize] * intermediate[PADDING+idx+t];
            
        image0[blockIdx.x * sizex + idx] = -accum;
    }
}

extern "C"{
    void Convolve0(rImage& img, rImage& kernel, rImage& globalmem)
    {
        int khsize = kernel.GetSize()/2;
        size_t sizex = img.sizex;

        dim3 threads = dim3(img.sizex > 1024 ? 1024 : img.sizex, 1,1);
        dim3 blocks = dim3(img.sizey,1,1);

        size_t shbytes = (img.sizex + 2*PADDING)*sizeof(float);
        
        if(shbytes > 18000)
            KConvolve0<false><<<blocks,threads,0>>>(img.gpuptr, kernel.gpuptr, sizex, khsize, globalmem.gpuptr);
        else
            KConvolve0<true><<<blocks,threads,shbytes>>>(img.gpuptr, kernel.gpuptr, sizex, khsize, nullptr);
        
        HANDLE_ERROR( cudaGetLastError() );
    }
}

template<bool bShared>
__global__ void KConvolve(float* restrict x, float* restrict p, const float* kernel, size_t sizex, int hkernelsize, float lambda, float alpha, int numiterations, float* restrict residuum2, float* restrict momentum, float beta, float* globalmem)
{
    extern __shared__ float sharedmem[];
    float* restrict padded_intermediate = bShared ? sharedmem : (globalmem + (sizex+128)*blockIdx.x);
        
    float* restrict intermediate = padded_intermediate + PADDING;
    float* restrict vel = momentum + 2*blockIdx.x*sizex;
    
    __syncthreads();
        
    for(int iter=0; iter<numiterations; iter++)
    {
        if(threadIdx.x < PADDING)
        {
            padded_intermediate[threadIdx.x] = p[blockIdx.x * sizex];
            padded_intermediate[sizex+2*PADDING-1-threadIdx.x] = p[blockIdx.x * sizex + sizex-1];
        }
        for(int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
            intermediate[idx] = p[blockIdx.x * sizex + idx];
            
        __syncthreads();
        
        for(int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
        {
            float accum = lambda * intermediate[idx];
            
            for(int t=-hkernelsize; t<=hkernelsize; t++)
                accum += kernel[t+hkernelsize] * intermediate[idx+t];
                
            float velocity = beta*vel[idx] + alpha*intermediate[idx];
            vel[idx] = velocity;
            
            float pvelocity = beta*vel[idx+sizex] - alpha*accum;
            vel[idx+sizex] = pvelocity;
            
            x[blockIdx.x * sizex + idx] += velocity;
            p[blockIdx.x * sizex + idx] += pvelocity;
        }
        __syncthreads();
    }
    
    for(int idx = threadIdx.x; idx < sizex; idx += blockDim.x)
        intermediate[idx] *= intermediate[idx];
    __syncthreads();
    
    for(int group = sizex/2; group > 32; group /= 2)
    {
        for(int idx = threadIdx.x; idx < group; idx += blockDim.x)
        intermediate[idx] += intermediate[idx+group];
        __syncthreads();
    }
    
    if(threadIdx.x < 32)
    {
        for(int group = 32; group > 0; group /= 2)
        {
            if(threadIdx.x < group)
            intermediate[threadIdx.x] += intermediate[threadIdx.x+group];
            __syncwarp();
        }
    }
    
    if(threadIdx.x == 0)
        residuum2[blockIdx.x] = intermediate[0];
}

extern "C"{
    void Convolve(rImage& x, rImage& p, rImage& kernel, float lambda, float alpha, int iter, rImage& res2, rImage& momentum, float beta, rImage& globalmem)
    {
        int khsize = kernel.GetSize()/2;	
        
        dim3 threads = dim3(x.sizex > 1024 ? 1024 : x.sizex, 1,1);
        dim3 blocks = dim3(x.sizey,1,1);
        
        float* kern = kernel.gpuptr;
        size_t shbytes = (x.sizex + 2*PADDING)*sizeof(float);

        if(shbytes > 18000)
            KConvolve<false><<<blocks,threads,0>>>(x.gpuptr, p.gpuptr, kern, x.sizex, khsize, lambda, alpha, iter, res2.gpuptr, momentum.gpuptr, beta, globalmem.gpuptr);
        else
            KConvolve<true><<<blocks,threads,shbytes>>>(x.gpuptr, p.gpuptr, kern, x.sizex, khsize, lambda, alpha, iter, res2.gpuptr, momentum.gpuptr, beta, nullptr);
        
        HANDLE_ERROR( cudaGetLastError() );
    }

    __global__ void KRedVolume(float* out, const float* volume, size_t volsizex, size_t volsizey, size_t slicesize)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t plane = blockIdx.y * slicesize;
        
        if(idx < volsizex)
        {
            float accum = 0;
            for(size_t idy = 0; idy < volsizey; idy++)
                accum += volume[plane + idy * volsizex + idx];

            out[blockIdx.y * volsizex + idx] = accum/volsizey;
        }
    }

    __device__ int float_to_int(float f)
    {
        int i = __float_as_int(f);
        i ^= (i<0) ? 0x7FFFFFFF : 0x0;
        return i;
    }

    __device__ float int_to_float(int i)
    {
        i ^= (i<0) ? 0x7FFFFFFF : 0x0;
        float f = __int_as_float(i);
        return i;
    }

    __global__ void KRedVolumeMM(float* out, complex* fminmax, const float* volume, size_t volsizex, size_t volsizey, size_t slicesize)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t plane = blockIdx.y * slicesize;
        __shared__ int smax[32];
        __shared__ int smin[32];
        
        if(threadIdx.x < 32)
        {
            smin[threadIdx.x] = 0x7FFFFFFF;
            smax[threadIdx.x] = 0x80000000;
        }
        
        __syncthreads();
        
        if(idx < volsizex)
        {		
            float accum = 0;
            for(size_t idy = 0; idy < volsizey; idy++)
                accum += volume[plane + idy * volsizex + idx];

            out[blockIdx.y * volsizex + idx] = accum/volsizey;
            
            int intaccum = float_to_int(accum);
            
            atomicMin(smin + threadIdx.x%32, intaccum);
            atomicMax(smax + threadIdx.x%32, intaccum);
        }
        
        __syncthreads();
        
        int tmax = 16;
        while(threadIdx.x < tmax)
        {
            smin[threadIdx.x] = min(smin[threadIdx.x],smin[threadIdx.x+tmax]);
            smax[threadIdx.x] = max(smax[threadIdx.x],smax[threadIdx.x+tmax]);
            __syncwarp( (1<<tmax)-1 );
            tmax /= 2;
        }
        if(threadIdx.x==0)
            fminmax[blockIdx.x + gridDim.x*blockIdx.y] = complex(int_to_float(smin[0]),int_to_float(smax[0]));
    }

    float VolumeAverage(float* mbar, float* volume, size_t sizex, size_t sizey, size_t sizez, float lambda, size_t slicesize)
    {
        dim3 threads = dim3(min((int)sizex,256),1,1);
        dim3 blocks = dim3((sizex+255)/256,sizez,1);
        
        if(lambda < 0)
        {
            cImage minmax(blocks.x*blocks.y,1);

            KRedVolumeMM<<<blocks,threads>>>(mbar, minmax.gpuptr, volume, sizex, sizey, slicesize);

            minmax.LoadFromGPU();

            cudaDeviceSynchronize();
            
            float vmin = +1E-10;
            float vmax = -1E-10;
            // printf("Here1: %e %e \n", vmax, vmin);

            for(size_t i=0; i<minmax.GetSize(); i++)
            {
                // printf("Here: %e %e %e %e %e %e\n",vmax, vmin, minmax.cpuptr[i].x,minmax.cpuptr[i].y, mbar[i], volume[i]);
                vmin = fminf(vmin,minmax.cpuptr[i].x);
                vmax = fmaxf(vmax,minmax.cpuptr[i].y); 
            }

            // printf("Here: %e %e %e %e \n",1.0f/(vmax-vmin), vmax,vmin, lambda);
            HANDLE_ERROR( cudaGetLastError() );
            return 1.0f/(vmax-vmin);
        }
        else
        {
            KRedVolume<<<blocks,threads>>>(mbar, volume, sizex, sizey, slicesize);

            HANDLE_ERROR( cudaGetLastError() );
            return lambda;
        }
    }

    __global__ void KApplyRing(float* volume, const float* nstaravg, size_t volsizex, size_t volsizey, size_t slicesize)
    {	
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t plane = blockIdx.y * slicesize;
        
        if(idx < volsizex)
        {
            const float nstar = nstaravg[blockIdx.y * volsizex + idx];
            
            for(size_t idy = 0; idy < volsizey; idy++)
            {
                //volume[plane + idy*volsizex + idx] = -sqrtf(fmaxf(est1 * est2, 0.0f));
                volume[plane + idy*volsizex + idx] += nstar;
            }
        }
    }

    void ApplyRing(float* volume, float* nstaravg, int sizex, int sizey, int sizez, size_t slicesize)
    {
        dim3 threads = dim3(min((int)sizex,128),1,1);
        dim3 blocks = dim3((sizex+127)/128,sizez,1);
        KApplyRing<<<blocks,threads>>>(volume, nstaravg, sizex, sizey, slicesize);

        HANDLE_ERROR( cudaGetLastError() );
    }

    float CalcAlpha(float* kern, size_t kernsize, float lambda)
    {
        float alpha = lambda+1E-10f;
        for(size_t i=0; i<kernsize; i++)
            alpha += fabsf(kern[i]);
            
        alpha = 1.75f/(sqrt(alpha)+sqrtf(lambda));
        alpha *= alpha;
        
        return alpha;
    }

    float CalcBeta(float* kern, size_t kernsize, float lambda)
    {
        float beta = lambda+1E-10f;
        for(size_t i=0; i<kernsize; i++)
            beta += fabsf(kern[i]);
            
        beta = (sqrtf(beta)-sqrtf(lambda))/(sqrtf(beta)+sqrtf(lambda));
        beta = beta*beta;
        
        return 0.95f*beta;
    }

    void CalcNorm2(float* out, const float* in, size_t msizex, size_t msizey)
    {
        for(size_t j=0; j<msizey; j++)
        {
            out[j] = 1E-5f;
            for(size_t i=0; i<msizex; i++)
                out[j] += in[j*msizex+i]*in[j*msizex+i];
            out[j] = 1.0f/out[j];
        }
    }


    __global__ void KAverageRing(float* out, const float* nvec1, const float* nvec2, size_t size)
    {	
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        if(idx < size)
            out[idx] = nvec1[idx]*0.5f + nvec2[idx]*0.5f;
    }

    void RingFilter(rImage& sinobar, float lambda, const float* norm2vec)
    {
        size_t msizex = sinobar.sizex;
        size_t msizey = sinobar.sizey;
        //const float fk1[] = {1.0f, -4.0f  ,  6.0f , -4.0f  ,  1.0f};
        float fk2[] = {-2.0f,  13.0f, -34.0f,  46.0f, -34.0f,  13.0f,  -2.0f};

        float fk1[] = {-1.0f, 0.0f, 9.0f, 20.0f, 9.0f, 0.0f, -1.0f};
        /*const float fk2[] = {-1.55794444e+01f,  2.06396597e+02f, -1.27719611e+03f,  4.89847819e+03f,
            -1.30230239e+04f,  2.54103134e+04f, -3.75466717e+04f,  4.26945658e+04f,
            -3.75466717e+04f,  2.54103134e+04f, -1.30230239e+04f,  4.89847819e+03f,
            -1.27719611e+03f,  2.06396597e+02f, -1.55794444e+01f}; */

        rImage kernel1(fk1, sizeof(fk1)/sizeof(fk1[0]), 1);
        rImage kernel2(fk2, sizeof(fk2)/sizeof(fk2[0]), 1);
            
        rImage xvec1(msizex,msizey);
        rImage xvec2(msizex,msizey);
        rImage mvec1(msizex,msizey);
        rImage mvec2(msizex,msizey);
        rImage momentum(msizex,2*msizey);
        rImage intermediate(msizex + 2*PADDING, msizey);

        momentum.SetGPUToZero();
        xvec1.SetGPUToZero();
        xvec2.SetGPUToZero();
        mvec1.CopyFrom(sinobar);
        mvec2.CopyFrom(sinobar);
        mvec1.LoadFromGPU();
        cudaDeviceSynchronize();
        
        rImage residuum2(msizey,1);
        
        const float alpha1 = CalcAlpha(kernel1.cpuptr, kernel1.GetSize(), lambda);
        const float alpha2 = CalcAlpha(kernel2.cpuptr, kernel2.GetSize(), lambda);
        
        const float beta1 = CalcBeta(kernel1.cpuptr, kernel1.GetSize(), lambda);
        const float beta2 = CalcBeta(kernel2.cpuptr, kernel2.GetSize(), lambda);
        
        Convolve0(mvec1, kernel1, intermediate); // mvec = FtF(mbar)
        Convolve0(mvec2, kernel2, intermediate); // mvec = FtF(mbar)

        float maxerr = 1;
        while(maxerr > 1E-6f)
        {
            Convolve(xvec1, mvec1, kernel1, lambda, alpha1, 100, residuum2, momentum, beta1, intermediate);
            
            residuum2.LoadFromGPU();
            cudaDeviceSynchronize();
            maxerr = 0;
            for(size_t j=0; j<msizey; j++)
                maxerr = fmaxf(maxerr, residuum2.cpuptr[j]*norm2vec[j]);
        }
        
        momentum.SetGPUToZero();
        maxerr = 1;
        
        while(maxerr > 1E-6f)
        {
            Convolve(xvec2, mvec2, kernel2, lambda, alpha2, 100, residuum2, momentum, beta2, intermediate);
            
            residuum2.LoadFromGPU();
            cudaDeviceSynchronize();
            maxerr = 0;
            for(size_t j=0; j<msizey; j++)
                maxerr = fmaxf(maxerr, residuum2.cpuptr[j]*norm2vec[j]);
        }

        KAverageRing<<<(xvec1.GetSize()+31)/32,32>>>(sinobar.gpuptr, xvec1.gpuptr, xvec2.gpuptr, xvec1.GetSize());
        HANDLE_ERROR( cudaGetLastError() );
    }

    static __global__ void KDIV(float* divv, float* flat, dim3 shp)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        size_t idz = blockIdx.z;

        if(idx < shp.x && idy < shp.y && idz < shp.z)
            divv[idx + idy*shp.x + idz*shp.x*shp.y] /= flat[idx + shp.x*idz];
        
    }
    static __global__ void KMUL(float* divv, float* flat, dim3 shp)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
        size_t idz = blockIdx.z;

        if(idx < shp.x && idy < shp.y && idz < shp.z)
            divv[idx + idy*shp.x + idz*shp.x*shp.y] *= flat[idx + shp.x*idz];
        
    }
    
    void Rings(float* volume, int vsizex, int vsizey, int vsizez, float lambda, size_t slicesize)
    {
        size_t msizex = vsizex;
        size_t msizey = vsizez;

        // for (int i = 0; i < vsizey; i++)
        //     printf("Vol = %e \n",volume[512*1024*1024 + i*1024]);
   
        rImage smooth(msizex,msizey);
        rImage sinobar(msizex,msizey);

        lambda = VolumeAverage(sinobar.gpuptr, volume, vsizex, vsizey, vsizez, lambda, slicesize);

        sinobar.LoadFromGPU();
        cudaDeviceSynchronize();
        float norm2vec[msizey];

        CalcNorm2(norm2vec, sinobar.cpuptr, msizex, msizey);

        RingFilter(sinobar, lambda, norm2vec);
        
        ApplyRing(volume, sinobar.gpuptr, vsizex, vsizey, vsizez, slicesize);
        HANDLE_ERROR( cudaGetLastError() );
    }

    void RingsEM(float* sinogram, float* ptrflat, size_t sizex, size_t sizey, size_t sizez)
    {
        rImage sino(sinogram, sizex, sizey, sizez, MemoryType::ENoAlloc);
        rImage FLAT(ptrflat, sizex, 1, sizez, MemoryType::ENoAlloc);
        rImage flat(FLAT);

        rImage div(sizex, sizey, sizez);
        div.CopyFrom(sino);

        //div /= flat;
        KDIV<<<div.ShapeBlock(),div.ShapeThread()>>>(div.gpuptr, flat.gpuptr, div.Shape());

        Rings(div.gpuptr + 0*sizex*sizey/4, sizex, sizey/4, sizez, -1, sizex*sizey);
        Rings(div.gpuptr + 1*sizex*sizey/4, sizex, sizey/4, sizez, -1, sizex*sizey);
        Rings(div.gpuptr + 2*sizex*sizey/4, sizex, sizey/4, sizez, -1, sizex*sizey);
        Rings(div.gpuptr + 3*sizex*sizey/4, sizex, sizey/4, sizez, -1, sizex*sizey);

        //div *= flat;
        KMUL<<<div.ShapeBlock(),div.ShapeThread()>>>(div.gpuptr, flat.gpuptr, div.Shape());
        div -= sino;
        
        Highpass(div, sqrtf(sizex)/1.1f);
        
        sino += div;

        rImage sinobar(sizex,sizez);
        VolumeAverage(sinobar.gpuptr, sino.gpuptr, sizex, sizey, sizez, 1.0, sizex*sizey);

        flat /= sinobar;
        flat.ln();

        float norm2vec[1] = {1E-3f};
        //flat.LoadFromGPU();
        //cudaDeviceSynchronize();
        //CalcNorm2(norm2vec, flat.cpuptr, sizex, 1);

        RingFilter(flat, 1E-5f, norm2vec);
        Highpass(flat, sqrtf(sizex)/1.1f);
        
        flat.exp();
        //sino /= flat;
        FLAT *= flat;
    }
}


extern "C"{

	void applyringsEM(int gpu, float* sinogram, float* ptrflat, int sizex, int sizey, int sizez)
	{
		cudaSetDevice(gpu);
		size_t blocksize = min(sizez,32);

		rImage gpuvol(sizex,sizey*blocksize);
		rImage gpuflat(sizex, blocksize);
		
		for(size_t bz=0; bz<sizez; bz+=blocksize)
		{
			blocksize = min(blocksize,size_t(sizez)-bz);

			gpuvol.CopyFrom(sinogram + bz*sizex*sizey, 0, sizex*sizey*blocksize);
			gpuflat.CopyFrom(ptrflat + bz*sizex, 0, sizex*blocksize);
			
			RingsEM(gpuvol.gpuptr, gpuflat.gpuptr, sizex, sizey, blocksize);

			gpuvol.CopyTo(sinogram + bz*sizex*sizey, 0, sizex*sizey*blocksize);
			gpuflat.CopyTo(ptrflat + bz*sizex, 0, sizex*blocksize);
		}
		HANDLE_ERROR( cudaGetLastError() );
	}

    void ringsgpu(int gpu, float* data, int nrays, int nangles, int nslices, float lambda_rings, int ringblocks)
	{   
		cudaSetDevice(gpu);
		size_t blocksize = min((size_t)nslices,32ul);

        // printf("GPU: %d, %ld %d, %d, %d, %f, %d\n", gpu,blocksize, nrays,nslices,nangles,lambda_rings,ringblocks);

		rImage tomogram(nrays,nangles,blocksize);
        // float* tomogram;
        // HANDLE_ERROR( cudaMalloc(&tomogram, sizeof(float) * nrays * nangles * (int)blocksize) );


		for(size_t bz=0; bz<nslices; bz+=blocksize){
			blocksize = min(blocksize,size_t(nslices)-bz);

            // HANDLE_ERROR( cudaMemcpy(tomogram, data + bz*nrays*nangles, sizeof(float) * nrays * nangles * blocksize, cudaMemcpyHostToDevice) );	

			tomogram.CopyFrom(data + bz*nrays*nangles, 0, nrays*nangles*blocksize);
            
            for (int m = 0; m < ringblocks / 2; m++){
                // Rings(tomogram, nrays, nangles, blocksize, lambda_rings, nrays*nangles);
                Rings(tomogram.gpuptr, nrays, nangles, blocksize, lambda_rings, nrays*nangles);
                size_t offset = nrays*nangles;
                size_t step = (nangles / ringblocks) * nrays;
                float* tomptr = tomogram.gpuptr;
                // float* tomptr = tomogram;

                for (int n = 0; n < ringblocks - 1; n++){
			        Rings(tomogram.gpuptr, nrays, nangles, blocksize, lambda_rings, nrays*nangles);
                    // Rings(tomogram, nrays, nangles, blocksize, lambda_rings, nrays*nangles);

                    tomptr += step;
                }
                Rings(tomptr, nrays, nangles%ringblocks + nangles/ringblocks, blocksize, lambda_rings, offset);

                // HANDLE_ERROR( cudaMemcpy(data + bz*nrays*nangles, tomogram ,  nrays * nangles * blocksize * sizeof(float) , cudaMemcpyDeviceToHost) );

			    tomogram.CopyTo(data + bz*nrays*nangles, 0, nrays*nangles*blocksize);
		    }
		
            HANDLE_ERROR( cudaGetLastError() );

	    }
        // cudaFree(tomogram);
        cudaDeviceSynchronize();
    }
}

extern "C"{ 
    
    void ringsblock(int* gpus, int ngpus, float* data, int nrays, int nangles, int nslices, float lambda_rings, int ringblocks)
    {
        int t;
        int blockgpu = (nslices + ngpus - 1) / ngpus;
        
        std::vector<std::future<void>> threads;

        for(t = 0; t < ngpus; t++){ 
            
            blockgpu = min(nslices - blockgpu * t, blockgpu);

            threads.push_back(std::async( std::launch::async, ringsgpu, gpus[t], data + (size_t)t * blockgpu * nrays*nangles, nrays, nangles, blockgpu, lambda_rings, ringblocks));
        }
    
        for(auto& t : threads)
            t.get();
    }

}

