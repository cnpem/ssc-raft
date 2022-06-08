// Authors: Giovanni Baraldi, Gilberto Martinez, Eduardo Miqueles
// Sinogram centering

#include "../../inc/include.h"
#include "../../inc/common/kernel_operators.hpp"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"

extern "C"{
    __global__ void KCrossFrame(complex* f, complex* g, const uint16_t* frame0, const uint16_t* frame1, const uint16_t* dark, const uint16_t* flat, size_t sizex)
    {
        const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
        const size_t idy = blockIdx.y;
        const size_t index = idy*sizex + idx;

        const size_t forw = sizex*idy+idx;
        const size_t invs = sizex*idy+sizex-1-idx;
        
        if(idx < sizex){
            float dk = (float) dark[index];
            float ft = (float) flat[index];
            
            f[forw] = -logf(fmaxf(frame0[index]-dk,0.5f)/fmaxf(ft-dk,0.5f));
            g[invs] = -logf(fmaxf(frame1[index]-dk,0.5f)/fmaxf(ft-dk,0.5f));
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

    int Centersino(uint16_t* frame0, uint16_t* frame1, uint16_t* dark, uint16_t* flat, size_t sizex, size_t sizey)
    {

        dim3 blocks((sizex+127)/128,sizey,1);
        dim3 threads(fminf(sizex,128),1,1);

        cImage f(sizex,sizey),g(sizex,sizey);

        int n[] = {(int)sizey};
        int inembed[] = {(int)sizey};

        cufftHandle plan, plan2;
        cufftPlan1d(&plan, sizex, CUFFT_C2C, sizey);
        cufftPlanMany(&plan2, 1, n, inembed, sizex, 1, inembed, sizex, 1, CUFFT_C2C, sizex);

        KCrossFrame<<<blocks,threads>>>(f.gpuptr, g.gpuptr, frame0, frame1, dark, flat, sizex);
        
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
}

extern "C"{
    int find_centersino16(uint16_t* frame0, uint16_t* frame1, uint16_t* dark, uint16_t* flat, int sizex, int sizey)
    {
        Image2D<uint16_t> fr0(frame0,sizex,sizey), fr1(frame1,sizex,sizey), dk(dark,sizex,sizey), ft(flat,sizex,sizey);	
        return Centersino(fr0.gpuptr, fr1.gpuptr, dk.gpuptr, ft.gpuptr, sizex, sizey);
    }

}
