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
#include "common/opt.hpp"

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

    float abs2Interp(complex* data,
            size_t sizex, size_t sizey,
            float posx, float posy) 
    {
        const int x0 = floor(posx);
        const int x1 = x0 + 1;
        const int y0 = floor(posy);
        const int y1 = y0 + 1;

        float v00 = 0.0f, v10 = 0.0f, v01 = 0.0f, v11 = 0.0f;

        if (y0 >= 0 && y0 < sizey && x0 >= 0 && x0 < sizex)  v00 = data[y0 * sizex + x0].abs2();
        if (y0 >= 0 && y0 < sizey && x0 >= 0 && x0 < sizex)  v01 = data[y0 * sizex + x1].abs2();
        if (y0 >= 0 && y0 < sizey && x0 >= 0 && x0 < sizex)  v10 = data[y1 * sizex + x0].abs2();
        if (y0 >= 0 && y0 < sizey && x0 >= 0 && x0 < sizex)  v11 = data[y1 * sizex + x1].abs2();

        const float interp =
            (float((x1 - posx) * (y1 - posy)) / float((x1 - x0) * (y1 - y0))) * v00 +
            (float((x1 - posx) * (y0 - posy)) / float((x1 - x0) * (y0 - y1))) * v10 +
            (float((x0 - posx) * (y1 - posy)) / float((x0 - x1) * (y1 - y0))) * v01 +
            (float((x0 - posx) * (y0 - posy)) / float((x0 - x1) * (y1 - y0))) * v11;

        return interp;
    }


    float abs2InterpCubic(complex* data,
            size_t sizex, size_t sizey,
            float posx, float posy) 
    {
        // Lagrange polinomials
        const int x0 = floor(posx);
        const int x1 = x0 + 1;
        const int x2 = x0 + 2;
        const int x3 = x0 + 3;
        const int y0 = floor(posy);
        const int y1 = y0 + 1;
        const int y2 = y0 + 2;
        const int y3 = y0 + 3;

        float v00 = 0.0f, v10 = 0.0f, v20 = 0.0f, v30 = 0.0f;
        float v01 = 0.0f, v11 = 0.0f, v21 = 0.0f, v31 = 0.0f;
        float v02 = 0.0f, v12 = 0.0f, v22 = 0.0f, v32 = 0.0f;
        float v03 = 0.0f, v13 = 0.0f, v23 = 0.0f, v33 = 0.0f;

        if (y0 >= 0 && y0 < sizey && x0 >= 0 && x0 < sizex)  v00 = data[y0 * sizex + x0].abs2();
        if (y0 >= 0 && y0 < sizey && x1 >= 0 && x1 < sizex)  v01 = data[y0 * sizex + x1].abs2();
        if (y0 >= 0 && y0 < sizey && x2 >= 0 && x2 < sizex)  v02 = data[y0 * sizex + x2].abs2();
        if (y0 >= 0 && y0 < sizey && x3 >= 0 && x3 < sizex)  v03 = data[y0 * sizex + x3].abs2();

        if (y1 >= 0 && y1 < sizey && x0 >= 0 && x0 < sizex)  v10 = data[y1 * sizex + x0].abs2();
        if (y1 >= 0 && y1 < sizey && x1 >= 0 && x1 < sizex)  v11 = data[y1 * sizex + x1].abs2();
        if (y1 >= 0 && y1 < sizey && x2 >= 0 && x2 < sizex)  v12 = data[y1 * sizex + x2].abs2();
        if (y1 >= 0 && y1 < sizey && x3 >= 0 && x3 < sizex)  v13 = data[y1 * sizex + x3].abs2();

        if (y2 >= 0 && y2 < sizey && x0 >= 0 && x0 < sizex)  v20 = data[y2 * sizex + x0].abs2();
        if (y2 >= 0 && y2 < sizey && x1 >= 0 && x1 < sizex)  v21 = data[y2 * sizex + x1].abs2();
        if (y2 >= 0 && y2 < sizey && x2 >= 0 && x2 < sizex)  v22 = data[y2 * sizex + x2].abs2();
        if (y2 >= 0 && y2 < sizey && x3 >= 0 && x3 < sizex)  v23 = data[y2 * sizex + x3].abs2();

        if (y3 >= 0 && y3 < sizey && x0 >= 0 && x0 < sizex)  v30 = data[y3 * sizex + x0].abs2();
        if (y3 >= 0 && y3 < sizey && x1 >= 0 && x1 < sizex)  v31 = data[y3 * sizex + x1].abs2();
        if (y3 >= 0 && y3 < sizey && x2 >= 0 && x2 < sizex)  v32 = data[y3 * sizex + x2].abs2();
        if (y3 >= 0 && y3 < sizey && x3 >= 0 && x3 < sizex)  v33 = data[y3 * sizex + x3].abs2();


        const float interp =
            (float((x1 - posx) * (x2 - posx) * (x3 - posx) * (y1 - posy) * (y2 - posy) * (y3 - posy) ) / float((x1 - x0) * (x2 - x0) * (x3 - x0) * (y1 - y0) * (y2 - y0) * (y3 - y0) )) * v00 +
            (float((x1 - posx) * (x2 - posx) * (x3 - posx) * (y0 - posy) * (y2 - posy) * (y3 - posy) ) / float((x1 - x0) * (x2 - x0) * (x3 - x0) * (y0 - y1) * (y2 - y1) * (y3 - y1) )) * v10 +
            (float((x1 - posx) * (x2 - posx) * (x3 - posx) * (y0 - posy) * (y1 - posy) * (y3 - posy) ) / float((x1 - x0) * (x2 - x0) * (x3 - x0) * (y0 - y2) * (y1 - y2) * (y3 - y2) )) * v20 +
            (float((x1 - posx) * (x2 - posx) * (x3 - posx) * (y0 - posy) * (y1 - posy) * (y2 - posy) ) / float((x1 - x0) * (x2 - x0) * (x3 - x0) * (y0 - y3) * (y1 - y3) * (y2 - y3) )) * v30 +

            (float((x0 - posx) * (x2 - posx) * (x3 - posx) * (y1 - posy) * (y2 - posy) * (y3 - posy) ) / float((x0 - x1) * (x2 - x1) * (x3 - x1) * (y1 - y0) * (y2 - y0) * (y3 - y0) )) * v01 +
            (float((x0 - posx) * (x2 - posx) * (x3 - posx) * (y0 - posy) * (y2 - posy) * (y3 - posy) ) / float((x0 - x1) * (x2 - x1) * (x3 - x1) * (y0 - y1) * (y2 - y1) * (y3 - y1) )) * v11 +
            (float((x0 - posx) * (x2 - posx) * (x3 - posx) * (y0 - posy) * (y1 - posy) * (y3 - posy) ) / float((x0 - x1) * (x2 - x1) * (x3 - x1) * (y0 - y2) * (y1 - y2) * (y3 - y2) )) * v21 +
            (float((x0 - posx) * (x2 - posx) * (x3 - posx) * (y0 - posy) * (y1 - posy) * (y2 - posy) ) / float((x0 - x1) * (x2 - x1) * (x3 - x1) * (y0 - y3) * (y1 - y3) * (y2 - y3) )) * v31 +

            (float((x0 - posx) * (x1 - posx) * (x3 - posx) * (y1 - posy) * (y2 - posy) * (y3 - posy) ) / float((x0 - x2) * (x1 - x2) * (x3 - x2) * (y1 - y0) * (y2 - y0) * (y3 - y0) )) * v02 +
            (float((x0 - posx) * (x1 - posx) * (x3 - posx) * (y0 - posy) * (y2 - posy) * (y3 - posy) ) / float((x0 - x2) * (x1 - x2) * (x3 - x2) * (y0 - y1) * (y2 - y1) * (y3 - y1) )) * v12 +
            (float((x0 - posx) * (x1 - posx) * (x3 - posx) * (y0 - posy) * (y1 - posy) * (y3 - posy) ) / float((x0 - x2) * (x1 - x2) * (x3 - x2) * (y0 - y2) * (y1 - y2) * (y3 - y2) )) * v22 +
            (float((x0 - posx) * (x1 - posx) * (x3 - posx) * (y0 - posy) * (y1 - posy) * (y2 - posy) ) / float((x0 - x2) * (x1 - x2) * (x3 - x2) * (y0 - y3) * (y1 - y3) * (y2 - y3) )) * v32 +

            (float((x0 - posx) * (x1 - posx) * (x2 - posx) * (y1 - posy) * (y2 - posy) * (y3 - posy) ) / float((x0 - x3) * (x1 - x3) * (x2 - x3) * (y1 - y0) * (y2 - y0) * (y3 - y0) )) * v03 +
            (float((x0 - posx) * (x1 - posx) * (x2 - posx) * (y0 - posy) * (y2 - posy) * (y3 - posy) ) / float((x0 - x3) * (x1 - x3) * (x2 - x3) * (y0 - y1) * (y2 - y1) * (y3 - y1) )) * v13 +
            (float((x0 - posx) * (x1 - posx) * (x2 - posx) * (y0 - posy) * (y1 - posy) * (y3 - posy) ) / float((x0 - x3) * (x1 - x3) * (x2 - x3) * (y0 - y2) * (y1 - y2) * (y3 - y2) )) * v23 +
            (float((x0 - posx) * (x1 - posx) * (x2 - posx) * (y0 - posy) * (y1 - posy) * (y2 - posy) ) / float((x0 - x3) * (x1 - x3) * (x2 - x3) * (y0 - y3) * (y1 - y3) * (y2 - y3) )) * v33;


        return interp;
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

    float getCentersino_subpixel(float* frame0, float* frame180, 
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

        float bestpos = 0.0f;

        const float di = 0.01f;
        //const size_t irange = size_t(sizex * (1.0f/di));
        for(size_t j=0; j<sizey; j++) {

            for(float i=0; i < irange; i++) {
                // const float bbs = abs2Interp(f.cpuptr, sizex, sizey, i * di, j);
                const float bbs = abs2InterpCubic(f.cpuptr, sizex, sizey, i * di, j);

                if(bbs > maxx) {
                    maxx = bbs;
                    posx = int(round(i * di));
                    bestpos = i * di;
                }
            }

            for(float i=irange; i < sizex; i++) {
                // const float bbs = abs2Interp(f.cpuptr, sizex, sizey, i * di, j);
                const float bbs = abs2InterpCubic(f.cpuptr, sizex, sizey, i * di, j);

                if(bbs > maxx) {
                    maxx = bbs;
                    posx = int(round(i * di));
                    bestpos = i * di;
                }
            }
        }
        // printf("bestpos = %f \n", bestpos);
        fflush(stdout);
        if(bestpos > (float)sizex/2.0f) // look into it!
            bestpos -= (float)sizex;

        // printf("After bestpos = %f \n", bestpos);
        fflush(stdout);
        // printf("bestpos/2 = %f \n", bestpos/2.0f);
        fflush(stdout);
        return -bestpos/2.0f;
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

    float findcentersino_subpixel(float* frame0, float* frame180,
    float* dark, float* flat, int sizex, int sizey)
    {
        Image2D<float> fr0(frame0,sizex,sizey), fr180(frame180,sizex,sizey), dk(dark,sizex,sizey), ft(flat,sizex,sizey);
        return getCentersino_subpixel(fr0.gpuptr, fr180.gpuptr, dk.gpuptr, ft.gpuptr, sizex, sizey);
    }

    int findcentersino(float* frame0, float* frame180,
    float* dark, float* flat, int sizex, int sizey)
    {
        Image2D<float> fr0(frame0,sizex,sizey), fr180(frame180,sizex,sizey), dk(dark,sizex,sizey), ft(flat,sizex,sizey);
        return getCentersino(fr0.gpuptr, fr180.gpuptr, dk.gpuptr, ft.gpuptr, sizex, sizey);
    }
}

__global__ void rot_axis_correction_kernel(complex *kernel, float axis_offset, dim3 size)
{
    size_t i = blockIdx.x*blockDim.x + threadIdx.x;
    size_t j = blockIdx.y*blockDim.y + threadIdx.y;
    size_t k = blockIdx.z*blockDim.z + threadIdx.z;

    size_t index = IND(i,j,k,size.x,size.y);
    
    if ( i >= size.x  || (j >= size.y) ) return;

    float expoent = 2.0f * float(M_PI)/(float)( 2 * size.x - 2) * axis_offset * i;

    kernel[index] *= exp1j(- expoent );
}

extern "C"{
    void getRotAxisCorrection(GPU gpus, float *tomogram, 
    float axis_offset, dim3 tomo_size)
    {
        /* Projection data sizes */
        int padx    = 2;
        int nrays   = tomo_size.x * ( 1.0 + padx );
        int nangles = tomo_size.y;
        int nslices = tomo_size.z;

        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock( (int)ceil( nrays   / TPBX ) + 1,
                        (int)ceil( nangles / TPBY ) + 1,
                        (int)ceil( nslices / TPBZ ) + 1);
        
        dim3 threadsPerBlockFFT(TPBX,TPBY,1);
        dim3 gridBlockFFT( (int)ceil( nrays   / threadsPerBlock.x ) + 1, 
                           (int)ceil( nangles / threadsPerBlock.y ) + 1, 
                           1);

        dim3 fft_size = dim3( nrays / 2 + 1, nangles, 1 );
        size_t nfft   = opt::get_total_points(fft_size);
        size_t npad   = nrays * nangles * nslices;

		cufftPlan1d(&gpus.mplan , nrays, CUFFT_R2C, nangles);
		cufftPlan1d(&gpus.mplanI, nrays, CUFFT_C2R, nangles);

        cufftComplex *fft = opt::allocGPU<cufftComplex>(nfft);
        float *dataPadded = opt::allocGPU<float>(npad);

        opt::paddR2R<<<gridBlock,threadsPerBlock>>>(tomogram, dataPadded, tomo_size,
                                                    dim3(padx,0,0));

        size_t offset; 
        for( int k = 0; k < nslices; k++){  
            
            offset = (size_t)k * nrays * nangles;

            HANDLE_FFTERROR(cufftExecR2C(gpus.mplan, dataPadded + offset, fft));
                    
            rot_axis_correction_kernel<<<gridBlockFFT,threadsPerBlockFFT>>>((complex*)fft, axis_offset, fft_size);

            HANDLE_FFTERROR(cufftExecC2R(gpus.mplanI, fft, dataPadded + offset));

        }
        
        opt::remove_paddR2R<<<gridBlock,threadsPerBlock>>>(dataPadded, tomogram, 
                                                            tomo_size, dim3(padx,0,0));

        float scale = (float)(nrays);

        opt::scale<<<gridBlock,threadsPerBlock>>>(tomogram, tomo_size, scale);

        HANDLE_ERROR(cudaFree(dataPadded));
        HANDLE_ERROR(cudaFree(fft));
		HANDLE_FFTERROR(cufftDestroy(gpus.mplan));
        HANDLE_FFTERROR(cufftDestroy(gpus.mplanI));

        HANDLE_ERROR(cudaDeviceSynchronize());   
    }
}

extern "C"{   

    void getRotAxisCorrectionGPU(GPU gpus, float *tomogram, 
    float axis_offset, dim3 tomo_size, int ngpu, int blocksize)
    {
        HANDLE_ERROR(cudaSetDevice(ngpu));

        /* Projection data sizes */
        int nrays   = tomo_size.x;
        int nangles = tomo_size.y;
        int sizez   = tomo_size.z;

        int i; 

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize(sizez, nangles * nrays * 6, true, BYTES_TO_GB * getTotalDeviceMemory());
            blocksize          = min(sizez, blocksize_aux);
        }

        int ind_block = (int)ceil( (float) sizez / blocksize );

        float *dtomo  = opt::allocGPU<float>((size_t) nrays * nangles * blocksize);

        /* Loop for each batch of size 'batch' in threads */
		int ptr = 0, subblock; size_t ptr_block_tomo = 0;

        for (i = 0; i < ind_block; i++){

			subblock       = min(sizez - ptr, blocksize);

			ptr_block_tomo = (size_t)nrays * nangles * ptr;

			/* Update pointer */
			ptr = ptr + subblock;
			
            opt::CPUToGPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)nrays * nangles * subblock);

            getRotAxisCorrection(gpus, dtomo, axis_offset, 
                                dim3(nrays, nangles, subblock));  /* Tomogram size */

            opt::GPUToCPU<float>(tomogram + ptr_block_tomo, dtomo, 
                                (size_t)nrays * nangles * subblock);

        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        HANDLE_ERROR(cudaFree(dtomo));
    }

    void getRotAxisCorrectionMultiGPU(int* gpus, int ngpus, 
    float* tomogram, float axis_offset, 
    int nrays, int nangles, int nslices, int blocksize)
    {
        int i, Maxgpudev;

		/* Multiples devices */
		HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

		/* If devices input are larger than actual devices on GPU, exit */
		for(i = 0; i < ngpus; i++) 
			assert(gpus[i] < Maxgpudev && "Invalid device number.");

		GPU gpu_parameters;

        dim3 tomo_size = dim3(nrays, nangles, nslices);

        setGPUParameters(&gpu_parameters, tomo_size, ngpus, gpus);

		int subvolume = (nslices + ngpus - 1) / ngpus;
		int subblock, ptr = 0; 

		if (ngpus == 1){ /* 1 device */

			getRotAxisCorrectionGPU(gpu_parameters, tomogram, axis_offset, tomo_size, gpus[0], blocksize);

		}else{
		/* Launch async Threads for each device.
			Each device solves a block of 'nrays * nangles' size.
		*/
			// See future c++ async launch
			std::vector<std::future<void>> threads = {};
            threads.reserve(ngpus);

			for (i = 0; i < ngpus; i++){
				
				subblock   = min(nslices - ptr, subvolume);

				threads.push_back( std::async( std::launch::async, 
                    getRotAxisCorrectionGPU, 
                    gpu_parameters, 
                    tomogram + (size_t)nrays * nangles * ptr, 
                    axis_offset,
                    dim3(nrays, nangles, subblock),
                    gpus[i],
                    blocksize));

                /* Update pointer */
				ptr = ptr + subblock;		

			}
			for (i = 0; i < ngpus; i++)
				threads[i].get();
		}
    }

}
