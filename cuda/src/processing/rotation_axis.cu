// Authors: Giovanni Baraldi, Gilberto Martinez, Eduardo Miqueles
// Sinogram centering

#include <vector_types.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "common/logerror.hpp"
#include "common/operations.hpp"
#include "common/types.hpp"
#include "processing/processing.hpp"

extern "C"{
    __global__ void KCrossFrame16(complex* f, complex* g, const uint16_t* frame0, const uint16_t* frame1, const uint16_t* dark, const uint16_t* flat, size_t sizex)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t idy = blockIdx.y;
        const size_t index = idy*sizex + idx;

        const size_t forw = sizex*idy+idx;
        const size_t invs = sizex*idy+sizex-1-idx;
        
        if(idx < sizex){
            float dk = (float) dark[index];
            float ft = (float) flat[index];
            
            f[forw] = complex(-logf(fmaxf(frame0[index]-dk,0.5f)/fmaxf(ft-dk,0.5f)));
            g[invs] = complex(-logf(fmaxf(frame1[index]-dk,0.5f)/fmaxf(ft-dk,0.5f)));
        }
    }

    __global__ void KCrossCorrelation(complex* F, const complex* G, size_t sizex)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t index = blockIdx.y * sizex + idx;
        
        if(idx < sizex){
            float xx = fminf(sizex-idx,idx)/sizex;
            float yy = fminf(gridDim.y-blockIdx.y,blockIdx.y)/gridDim.y;
            float gauss = expf(-20.0f*(xx*xx + yy*yy));

            complex fg = F[index]*G[index].conj();
            F[index] = fg * gauss / fg.abs();
        }
    }

    int getCentersino16(uint16_t* frame0, uint16_t* frame180, 
    uint16_t* dark, uint16_t* flat, 
    size_t sizex, size_t sizey)
    {

        dim3 blocks((sizex+127)/128,sizey,1);
        dim3 threads(fminf(sizex,128),1,1);

        cImage f(sizex,sizey),g(sizex,sizey);

        int n[] = {(int)sizey};
        int inembed[] = {(int)sizey};

        cufftHandle plan, plan2;
        cufftPlan1d(&plan, sizex, CUFFT_C2C, sizey);
        cufftPlanMany(&plan2, 1, n, inembed, sizex, 1, inembed, sizex, 1, CUFFT_C2C, sizex);

        KCrossFrame16<<<blocks,threads>>>(f.gpuptr, g.gpuptr, frame0, frame180, dark, flat, sizex);
        
        cufftExecC2C(plan, f.gpuptr, f.gpuptr, CUFFT_FORWARD);
        cufftExecC2C(plan, g.gpuptr, g.gpuptr, CUFFT_FORWARD);

        cufftExecC2C(plan2, f.gpuptr, f.gpuptr, CUFFT_FORWARD);
        cufftExecC2C(plan2, g.gpuptr, g.gpuptr, CUFFT_FORWARD);

        KCrossCorrelation<<<blocks,threads>>>(f.gpuptr, g.gpuptr, sizex);

        cufftExecC2C(plan, f.gpuptr, f.gpuptr, CUFFT_INVERSE);
        cufftExecC2C(plan2, f.gpuptr, f.gpuptr, CUFFT_INVERSE);
        
        f.LoadFromGPU();
        cudaDeviceSynchronize();

        cufftDestroy(plan);
        cufftDestroy(plan2);
        
        float maxx = 0;
        int posx = 0;
        const size_t irange = (sizex>>2)<<2;
        for(size_t j=0; j<sizey; j++){
            for(size_t i=0; i<irange; i+=4){
                float bbs = f.cpuptr[j*sizex + i].abs2();
                if(bbs > maxx)
                {
                    maxx = bbs;
                    posx = int(i);
                }
            }
            for(size_t i=irange; i<sizex; i++){
                float bbs = f.cpuptr[j*sizex + i].abs2();
                if(bbs > maxx){
                    maxx = bbs;
                    posx = int(i);
                }
            }
        }
        
        if(posx > (float)sizex/2)
            posx -= sizex;
            
        return -posx/2;
    }

    __global__ void KCrossFrame(complex* f, complex* g, const float* frame0, const float* frame1, const float* dark, const float* flat, size_t sizex)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t idy = blockIdx.y;
        const size_t index = idy*sizex + idx;

        const size_t forw = sizex*idy+idx;
        const size_t invs = sizex*idy+sizex-1-idx;
        
        if(idx < sizex){
            float dk = (float) dark[index];
            float ft = (float) flat[index];
            
            f[forw] = complex(-logf(fmaxf(frame0[index]-dk,0.5f)/fmaxf(ft-dk,0.5f)));
            g[invs] = complex(-logf(fmaxf(frame1[index]-dk,0.5f)/fmaxf(ft-dk,0.5f)));
        }
    }

    int getCentersino(float* frame0, float* frame180, 
    float* dark, float* flat, 
    size_t sizex, size_t sizey)
    {

        dim3 blocks((sizex+127)/128,sizey,1);
        dim3 threads(fminf(sizex,128),1,1);

        cImage f(sizex,sizey),g(sizex,sizey);

        int n[] = {(int)sizey};
        int inembed[] = {(int)sizey};

        cufftHandle plan, plan2;
        cufftPlan1d(&plan, sizex, CUFFT_C2C, sizey);
        cufftPlanMany(&plan2, 1, n, inembed, sizex, 1, inembed, sizex, 1, CUFFT_C2C, sizex);

        KCrossFrame<<<blocks,threads>>>(f.gpuptr, g.gpuptr, frame0, frame180, dark, flat, sizex);
        
        cufftExecC2C(plan, f.gpuptr, f.gpuptr, CUFFT_FORWARD);
        cufftExecC2C(plan, g.gpuptr, g.gpuptr, CUFFT_FORWARD);

        cufftExecC2C(plan2, f.gpuptr, f.gpuptr, CUFFT_FORWARD);
        cufftExecC2C(plan2, g.gpuptr, g.gpuptr, CUFFT_FORWARD);

        KCrossCorrelation<<<blocks,threads>>>(f.gpuptr, g.gpuptr, sizex);

        cufftExecC2C(plan, f.gpuptr, f.gpuptr, CUFFT_INVERSE);
        cufftExecC2C(plan2, f.gpuptr, f.gpuptr, CUFFT_INVERSE);
        
        f.LoadFromGPU();
        cudaDeviceSynchronize();

        cufftDestroy(plan);
        cufftDestroy(plan2);
        
        float maxx = 0;
        int posx = 0;
        const size_t irange = (sizex>>2)<<2;
        for(size_t j=0; j<sizey; j++){
            for(size_t i=0; i<irange; i+=4){
                float bbs = f.cpuptr[j*sizex + i].abs2();
                if(bbs > maxx)
                {
                    maxx = bbs;
                    posx = int(i);
                }
            }
            for(size_t i=irange; i<sizex; i++){
                float bbs = f.cpuptr[j*sizex + i].abs2();
                if(bbs > maxx){
                    maxx = bbs;
                    posx = int(i);
                }
            }
        }
        
        if(posx > (float)sizex/2)
            posx -= sizex;
            
        return -posx/2;
    }

    __global__ void KCorrectRotationAxis(float* tomoin, float* tomoout,
            int sizex, int sizey, int sizez, int deviation) {
        const size_t idz = threadIdx.z + blockIdx.z * blockDim.z;
        const size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx != 0) //use gridDim.x == 1 to simplify kernel implementation
            return;

        const int offset = abs(deviation);

        float* in = &tomoin[idz * sizex * sizey + idy * sizex];
        float* out = &tomoout[idz * sizex * sizey + idy * sizex];

        if (deviation > 0) { //shift right
            for (int x = sizex - 1; x >= offset; --x) out[x] = in[x - offset];
            for (int x = offset - 1; x >= 0; --x) out[x] = 0.0f;
        } else if (deviation < 0) { //shift left
            for (int x = 0; x < sizex - offset; ++x) out[x] = in[x + offset];
            for (int x = sizex - offset; x < sizex; ++x) out[x] = 0.0f;
        }
    }

    void getCorrectRotationAxis(float* d_tomo_in, float* d_tomo_out,
            dim3 tomo_size, int deviation) {
        dim3 gridDim(1, 32, 32);
        dim3 blockDim(tomo_size.x,
                tomo_size.y / 32 + tomo_size.y % 32,
                tomo_size.z / 32 + tomo_size.z % 32);
        KCorrectRotationAxis<<<blockDim, gridDim>>>(d_tomo_in, d_tomo_out,
                tomo_size.x, tomo_size.y, tomo_size.z, deviation);
    }

    /**
      * Shifts the x axis according to specified deviation
      * To do the shift inplace, use tomoin = tomoout
      */
    void correctRotationAxis(float* tomoin, float* tomoout,
            int sizex, int sizey, int sizez, int deviation) {
        const int offset = abs(deviation);

        if (deviation > 0) { //shift right
            for (size_t z = 0; z < sizez; ++z) {
                for (size_t y = 0; y < sizey; ++y) {
                    float* in = &tomoin[z * sizex * sizey + y * sizex];
                    float* out = &tomoout[z * sizex * sizey + y * sizex];
                    for (int x = sizex - 1; x >= offset; --x) out[x] = in[x - offset];
                    for (int x = offset - 1; x >= 0; --x) out[x] = 0.0f;
                }
            }
        } else if (deviation < 0) { //shift left
            for (size_t z = 0; z < sizez; ++z) {
                for (size_t y = 0; y < sizey; ++y) {
                    float* in = &tomoin[z * sizex * sizey + y * sizex];
                    float* out = &tomoout[z * sizex * sizey + y * sizex];
                    for (int x = 0; x < sizex - offset; ++x) out[x] = in[x + offset];
                    for (int x = sizex - offset; x < sizex; ++x) out[x] = 0.0f;
                }
            }
        }
    }

}

extern "C"{
    int findcentersino16(uint16_t* frame0, uint16_t* frame180, 
    uint16_t* dark, uint16_t* flat, int sizex, int sizey)
    {
        Image2D<uint16_t> fr0(frame0,sizex,sizey), fr180(frame180,sizex,sizey), dk(dark,sizex,sizey), ft(flat,sizex,sizey);
        return getCentersino16(fr0.gpuptr, fr180.gpuptr, dk.gpuptr, ft.gpuptr, sizex, sizey);
    }

    int findcentersino(float* frame0, float* frame180,
    float* dark, float* flat, int sizex, int sizey)
    {
        Image2D<float> fr0(frame0,sizex,sizey), fr180(frame180,sizex,sizey), dk(dark,sizex,sizey), ft(flat,sizex,sizey);
        return getCentersino(fr0.gpuptr, fr180.gpuptr, dk.gpuptr, ft.gpuptr, sizex, sizey);
    }
}
