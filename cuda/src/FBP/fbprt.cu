// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"


extern "C"{
    void GPUFBP(char* blockRecon, float* sinoblock, int nrays, int nangles, int isizez, int sizeimage, 
                    int csino, CFilter reg, EType datatype, float threshold, float* angs, bool bShiftCenter)
    {
        size_t sizez = size_t(isizez);

        // printf("GPUFBP: %ld %d %d %d\n",sizez,nrays,nangles,sizeimage);

        cudaDeviceSynchronize();
        auto time0 = std::chrono::system_clock::now();

        rImage sintable(1,nangles);
        rImage costable(1,nangles);

        if(angs != nullptr) for(int i=0; i<nangles; i++)
        {
            sintable.cpuptr[i] = sin(angs[i]);
            costable.cpuptr[i] = cos(angs[i]);
            // printf("hare\n");
        }
        else for(int i=0; i<nangles; i++)
        {
            sintable.cpuptr[i] = sin(float(i)*float(M_PI)/float(nangles));
            costable.cpuptr[i] = cos(float(i)*float(M_PI)/float(nangles));
            // printf("no angles hare\n");
        }

        sintable.LoadToGPU();
        costable.LoadToGPU();
        
        SinoFilter(sinoblock, nrays, nangles, sizez, csino, true, reg, bShiftCenter, sintable.gpuptr);

        dim3 threads(16,16,1); 
        dim3 blocks((sizeimage+15)/16,(sizeimage+15)/16,sizez);
        KBackProjection_RT<<<blocks, threads>>>(blockRecon, sinoblock, sizeimage, nrays, nangles, datatype.type, threshold, sintable.gpuptr, costable.gpuptr);

        cudaDeviceSynchronize();
        auto time3 = std::chrono::system_clock::now();

    }
}

extern "C"{
    void fbpsingleGPU(int gpu, float* blockRecon, float* sinoblock, int nrays, 
            int nangles, int isizez, int sizeimage, int csino, float reg_val, int FilterType, float* angs, int bShiftCenter)
    {
        CFilter reg(FilterType,reg_val);

        cudaSetDevice(gpu);
        size_t sizez = size_t(isizez);
        
        cudaDeviceSynchronize();
        auto time0 = std::chrono::system_clock::now();

        dim3 threads(16,16,1); 
        dim3 blocks((sizeimage+15)/16,(sizeimage+15)/16,1);

        rImage sintable(1,nangles);
        rImage costable(1,nangles);

        if(angs != nullptr) for(int i=0; i<nangles; i++)
        {
            sintable.cpuptr[i] = sin(angs[i]);
            costable.cpuptr[i] = cos(angs[i]);
        }
        else for(int i=0; i<nangles; i++)
        {
            sintable.cpuptr[i] = sin(float(i)*float(M_PI)/float(nangles));
            costable.cpuptr[i] = cos(float(i)*float(M_PI)/float(nangles));
        }

        sintable.LoadToGPU();
        costable.LoadToGPU();

        rImage sino(nrays, nangles, min(sizez,32ul), MemoryType::EAllocGPU);
        rImage recon(sizeimage,sizeimage, min(sizez,32ul), MemoryType::EAllocGPU);

        for(size_t b=0; b < sizez; b += 32)
        {
            blocks.z = min(sizez-b,32ul);
            size_t reconoffset = b*sizeimage*sizeimage;

            sino.CopyFrom(sinoblock + b*nrays*nangles, 0, blocks.z*nrays*nangles);
            SinoFilter(sino.gpuptr, nrays, nangles, blocks.z, csino, true, reg, bShiftCenter, sintable.gpuptr);
            KBackProjection_RT<<<blocks, threads>>>((char*)recon.gpuptr, sino.gpuptr, sizeimage, nrays, nangles, EType::TypeEnum::FLOAT32, 0, sintable.gpuptr, costable.gpuptr);
            recon.CopyTo(reconoffset + blockRecon, 0, blocks.z*sizeimage*sizeimage);
        }

        cudaDeviceSynchronize();
        auto time3 = std::chrono::system_clock::now();

        //std::cout << "Total: " << (time3-time0).count()/1000000 << " ms." << std::endl;
    }
    
}

extern "C"{   

    void fbpgpu(int gpu, float* recon, float* tomogram, int nrays, int nangles, int nslices, int reconsize, int centersino,
        float reg_val, float* angles, float threshold, int reconPrecision, int FilterType, int bShiftCenter)
    {
        CFilter reg(FilterType,reg_val);

        EType datatype = EType((EType::TypeEnum)reconPrecision);

        size_t blocksize = min((size_t)nslices,32ul);
        // blocksize = size_t(nslices);

        cudaSetDevice(gpu);

        rImage tomo(nrays, nangles, blocksize, MemoryType::EAllocGPU);
        rImage blockRecon(reconsize, reconsize, blocksize, MemoryType::EAllocGPU);

        // printf("data fbp: %d %d %d %d %d\n",gpu,nrays,nangles,nslices,reconsize);
        // size_t b = 0;
        for(size_t b = 0; b < nslices; b += blocksize){
            
            blocksize = min(size_t(nslices) - b, blocksize);
            // printf("block: %ld %ld\n",blocksize,b);
			
            tomo.CopyFrom(tomogram + (size_t)b*nrays*nangles, 0, (size_t)nrays*nangles*blocksize);
            
            GPUFBP((char*)blockRecon.gpuptr, tomo.gpuptr, nrays, nangles, blocksize, reconsize, centersino, reg, datatype, threshold, angles, bShiftCenter);

            blockRecon.CopyTo(recon + (size_t)b*reconsize*reconsize, 0, (size_t)reconsize*reconsize*blocksize);
        }
        cudaDeviceSynchronize();
    }

    void fbpblock(int* gpus, int ngpus, float* recon, float* tomogram, int nrays, int nangles, int nslices, int reconsize, int centersino,
        float reg_val, float* angles, float threshold, int reconPrecision, int FilterType, int bShiftCenter)
    {
        int t;
        int blockgpu = (nslices + ngpus - 1) / ngpus;
        
        std::vector<std::future<void>> threads;

        for(t = 0; t < ngpus; t++){ 
            
            blockgpu = min(nslices - blockgpu * t, blockgpu);

            threads.push_back(std::async( std::launch::async, fbpgpu, gpus[t], recon + (size_t)t * blockgpu * reconsize*reconsize, 
                tomogram + (size_t)t * blockgpu * nrays*nangles, nrays, nangles, blockgpu, reconsize, centersino,
                reg_val, angles, threshold, reconPrecision, FilterType, bShiftCenter
            ));
        }
    
        for(auto& t : threads)
            t.get();
    }

    void _fbp(int gpu, char* recon, float* tomogram, float* angles, int nrays, int nangles, int nslices, int reconsize, int centersino, CFilter reg, 
        EType datatype, float threshold, int bShiftCenter)
    {
        size_t memframe = 10*nrays*nangles;
        size_t maxusage = 1ul<<33;
        size_t maxblock = min(max(maxusage/memframe,size_t(1)),32ul);

        cudaSetDevice(gpu);

        rImage tomo(nrays, nangles, maxblock);
        Image2D<char> blockRecon(nrays, nrays, maxblock * datatype.Size());

        for(size_t b = 0; b < nslices; b += maxblock){
            
            size_t blocksize = min(nslices - b, maxblock);

            HANDLE_ERROR(cudaMemcpy2D(tomo.gpuptr, 2 * nrays * maxblock, tomogram + b * nrays, 2 * nrays, nrays * blocksize*2, nangles, cudaMemcpyDefault));
            
            GPUFBP(blockRecon.gpuptr, tomo.gpuptr, nrays, nangles, blocksize, reconsize, centersino, reg, datatype, threshold, angles, bShiftCenter);

            HANDLE_ERROR(cudaMemcpy(recon + datatype.Size() * b * reconsize * reconsize, blockRecon.gpuptr, reconsize * reconsize * blocksize * datatype.Size(), cudaMemcpyDefault));
        }
    }

    void _fbpblock(int* gpus, int ngpus, char* recon, float* tomogram, int nrays, int nangles, int nslices, int reconsize, int centersino,
        float reg_val, float* angles, float threshold, int reconPrecision, int FilterType, int bShiftCenter)
    {
        CFilter reg(FilterType,reg_val);

        EType datatype = EType((EType::TypeEnum)reconPrecision);
        
        size_t memframe = 8*nrays*nangles;
        size_t maxusage = 1ul<<33;
        size_t maxblock = min(max(maxusage/memframe,size_t(1)),32ul);

        size_t blockgpu = (nslices + ngpus - 1) / ngpus;
        
        std::vector<std::future<void>> threads;

        for(size_t t = 0; t < ngpus; t++) 
            if(blockgpu * t < nslices){
                size_t lastblock = min(nslices - blockgpu * t, blockgpu);
                char* offsetptr = recon + t * blockgpu * size_t(reconsize*reconsize);


                threads.push_back(std::async( std::launch::async, _fbp, 
                    gpus[t], offsetptr, tomogram, angles, nrays, nangles, lastblock, reconsize, 
                    centersino, reg, datatype, threshold, bShiftCenter
                ));
            }
    
        for(auto& t : threads)
        t.get();
    }
}

extern "C"{
    __global__ void KBackProjection_RT(char* recon, const float *sino, int wdI, int nrays, int nangles,
        EType::TypeEnum datatype, float threshold, const float* sintable, const float* costable)
    {  
        int i = (blockDim.x * blockIdx.x + threadIdx.x);
        int j = (blockDim.y * blockIdx.y + threadIdx.y);

        float sum = 0, frac, sink, cosk, x, y, t, norm;
        int T, k;
        
        if ( (i>=wdI) || (j >= wdI) ) return;

        norm  = 0.5f*float(M_PI)/float(nangles)/float(nrays); // This is still weird

        x     = -wdI/2 + i;
        y     = -wdI/2 + j;

        for(k = 0; k < (nangles); k++){
            sink = sintable[k];
            cosk = costable[k];
        
            t = (x * cosk + y * sink + nrays/2);
            T = int(t);
        
            if (T >= 0 && T < nrays-1){
                frac = t-T;
                sum += sino[nangles * nrays * blockIdx.z + k * nrays + T] * (1.0f - frac) + sino[k * nrays + T + 1] * frac;
            }
        }        

        BasicOps::set_pixel(recon, sum*norm, i, j + wdI*blockIdx.z, wdI, threshold, datatype);
    } 

    __global__ void KRadon_RT(float* restrict frames, const float* image, int nrays, int nangles)
    {
        int ray = (blockDim.x * blockIdx.x + threadIdx.x);
        int ang = (blockDim.y * blockIdx.y + threadIdx.y);
        const size_t sizef = nrays*nrays;
        
        if ( ray>=nrays || ang >= nangles )
            return;
        
        float btheta = float(M_PI)*float(ang + nangles/2)/nangles;
        
        float ct = nrays/2;
        float cos_t = cosf(btheta + 1E-5f);
        float sin_t = sinf(btheta + 1E-5f);

        float x = ct - 2.0f*ct*cos_t + (ray-nrays/2)*sin_t;
        float y = ct - 2.0f*ct*sin_t - (ray-nrays/2)*cos_t;

        float tx0 = -x/cos_t;
        float tx1 = (nrays-x)/cos_t;

        float ty0 = -y/sin_t;
        float ty1 = (nrays-y)/sin_t;

        float d1 = fmaxf(fminf(tx0,tx1), fminf(ty0,ty1));
        int d2 = int(ceil(fminf(fmaxf(tx0,tx1), fmaxf(ty0,ty1)) - d1)+0.1f);

        x += d1*cos_t;
        y += d1*sin_t;
        
        float radon = 0;
        for(int s=0; s<d2; s++)
        {
            radon += image[(int(y+0.5f)*nrays + int(x+0.5f))%sizef + blockIdx.z*sizef];

            x += cos_t;
            y += sin_t;
        }

        frames[blockIdx.z*size_t(nrays*nangles) + nrays*ang + ray] = radon;
    }

}

extern "C"
{
	void GRadon(int device, float* _frames, float* _image, int nrays, int nangles, int blocksize)
	{
		cudaSetDevice(device);

		rImage frames(nrays,nangles,blocksize);
		rImage image(_image,nrays,nrays,blocksize);

		KRadon_RT<<<dim3(nrays/64,nangles,blocksize),64>>>(frames.gpuptr, image.gpuptr, nrays, nangles);

		frames.CopyTo(_frames);
		cudaDeviceSynchronize();
	}

	void GBackprojection(int device, float* _recon, float* _sino, int nrays, int nangles, int blocksize)
	{
		cudaSetDevice(device);

		rImage sintable(nangles,1);
		rImage costable(nangles,1);

		for(int a=0; a<nangles; a++)
		{
			sintable.cpuptr[a] = sinf(float(M_PI)*a/float(nangles));
			costable.cpuptr[a] = cosf(float(M_PI)*a/float(nangles));
		}

		sintable.LoadToGPU();
		costable.LoadToGPU();

		rImage recon(nrays,nrays,blocksize);
		rImage sino(_sino,nrays,nangles,blocksize);

		KBackProjection_RT<<<dim3(nrays/64,nrays,blocksize),64>>>(
			(char*)recon.gpuptr, sino.gpuptr, nrays, nrays, nangles, EType::TypeEnum::FLOAT32, 0, sintable.gpuptr, costable.gpuptr);

		recon.CopyTo(_recon);
		cudaDeviceSynchronize();

	}
}