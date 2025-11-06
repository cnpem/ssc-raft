// Authors: Giovanni Baraldi, Eduardo X. Miqueles

#include <cuda_runtime_api.h>
#include <cufft.h>
#include <driver_types.h>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <future>
#include <ratio>
#include <vector>

#include "common/complex.hpp"
#include "common/configs.hpp"
#include "common/logerror.hpp"
#include "common/opt.hpp"
#include "common/types.hpp"
#include "geometries/parallel/bst.hpp"
#include "processing/filters.hpp"

extern "C"{
    WBST *InitializeBST_workspace(dim3 tomo_size, dim3 obj_size, int bst_padd, int blocksize_bst)
    {  
        /* Allocate the local GPU variables:
        tomo_size = (nrays,nangles,nslices_gpu_block) = (tomo_size.x, tomo_size.y, tomo_size.z)
        obj_size  = (nrays,  nrays,nslices_gpu_block) = ( obj_size.x,  obj_size.y,  obj_size.z)
        */
        WBST *workspace = (WBST *)malloc(sizeof(WBST));

        int nrays      = tomo_size.x;
        int nangles    = tomo_size.y;
        int sizeImagex = obj_size.x;

        int dimmsfilter[] = {nrays};
        int dimms1d[]     = {(int)nrays * bst_padd / 2};
        int dimms2d[]     = {(int)sizeImagex, (int)sizeImagex};
        int beds[]        = {nrays * bst_padd / 2};

        HANDLE_FFTERROR(cufftPlanMany(&workspace->plan1d, 1, dimms1d, beds, 1, nrays * bst_padd / 2, beds, 1, nrays * bst_padd / 2, CUFFT_C2C, nangles * blocksize_bst * 2));
        HANDLE_FFTERROR(cufftPlanMany(&workspace->plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize_bst));
        HANDLE_FFTERROR(cufftPlanMany(&workspace->filterplan, 1, dimmsfilter, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, nangles * blocksize_bst));

        workspace->filtersino     = new cImage(           nrays,    nangles * blocksize_bst, 1, MemoryType::EAllocGPU);
        workspace->cartesianblock = new cImage(      sizeImagex, sizeImagex * blocksize_bst, 1, MemoryType::EAllocGPU);
        workspace->polarblock     = new cImage(nrays * bst_padd,    nangles * blocksize_bst, 1, MemoryType::EAllocGPU);
        workspace->realpolar      = new cImage(nrays * bst_padd,    nangles * blocksize_bst, 1, MemoryType::EAllocGPU);

        return workspace;
    }

    void freeBSTWorkspace(WBST *workspace)
    {  /* Deallocate the GPU variables */

        /* GPU */
        delete workspace->filtersino;
        delete workspace->cartesianblock;
        delete workspace->polarblock;
        delete workspace->realpolar;

        HANDLE_FFTERROR(cufftDestroy(workspace->plan1d));
        HANDLE_FFTERROR(cufftDestroy(workspace->plan2d));
        HANDLE_FFTERROR(cufftDestroy(workspace->filterplan));

        free(workspace);
    }
}
__global__ void sino2p(complex* padded, float* in, size_t nrays, size_t nangles, int pad0, int csino) {
    int center = nrays / 2 - csino;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nrays / 2) {
        size_t fory = blockIdx.y;
        size_t revy = blockIdx.y + nangles;
        size_t slicez = blockIdx.z * nrays * nangles;
        // float Arg2 = (2.0f*idx - pad0*nrays/2 + 1.0f)/(pad0*nrays/2 - 1.0f);
        // double b1 = cyl_bessel_i0f(sqrtf(fmaxf(1.0f - Arg2 * Arg2,0.0f)));
        // double b2 = cyl_bessel_i0f(1.0f);
        float w_bessel = 1;  // fabsf(b1/b2);
        if (idx == 0) w_bessel *= 0.5f;

        w_bessel *= sq(pad0);

        if (center - 1 - idx >= 0)
            padded[pad0 * slicez + pad0 * fory * nrays / 2 + idx] =
                complex(w_bessel * in[slicez + fory * nrays + center - 1 - idx]);
        else
            padded[pad0 * slicez + pad0 * fory * nrays / 2 + idx] = complex(w_bessel * in[slicez + fory * nrays]);
        if (center + 0 + idx >= 0)
            padded[pad0 * slicez + pad0 * revy * nrays / 2 + idx] =
                complex(w_bessel * in[slicez + fory * nrays + center + 0 + idx]);
        else
            padded[pad0 * slicez + pad0 * revy * nrays / 2 + idx] = complex(w_bessel * in[slicez + fory * nrays]);
    }
}

__global__ void convBST(complex* block, size_t nrays, size_t nangles) {
    /* Convolution BST with kernel = sigma = 2 /( Nx * max( min(i,Nx-i), 0.5) ) ) */
    size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t ty = blockIdx.y;
    size_t tz = blockIdx.z;

    float sigma = 2.0f / (nrays * (fmaxf(fminf(tx, nrays - tx), 0.5f)));
    size_t offset = tz * nangles * nrays + ty * nrays + tx;

    if (tx < nrays) block[offset] *= sigma;
}

__global__ void polar2cartesian_fourier(complex* cartesian, complex* polar, float* angles, size_t nrays, size_t nangles,
                                        size_t sizeimage) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;

    if (tx < sizeimage) {
        size_t cartplane = blockIdx.z * sizeimage * sizeimage;
        polar += blockIdx.z * nrays * nangles;

        int posx = tx - sizeimage / 2;
        int posy = ty - sizeimage / 2;

        float rho = nrays * hypotf(posx, posy) / sizeimage;
        float angle = (nangles) * (0.5f * atan2f(posy, posx) / float(M_PI) + 0.5f);

        size_t irho = size_t(rho);
        int iarc = int(angle);
        complex interped = complex(0.0f);

        if (irho < nrays / 2 - 1) {
            float pfrac = rho - irho;
            float tfrac = iarc - angle;

            iarc = iarc % (nangles);

            int uarc = (iarc + 1) % (nangles);

            complex interp0 = polar[iarc * nrays + irho] * (1.0f - pfrac) + polar[iarc * nrays + irho + 1] * pfrac;
            complex interp1 = polar[uarc * nrays + irho] * (1.0f - pfrac) + polar[uarc * nrays + irho + 1] * pfrac;

            interped = interp0 * tfrac + interp1 * (1.0f - tfrac);
        }

        cartesian[cartplane + sizeimage * ((ty + sizeimage / 2) % sizeimage) + (tx + sizeimage / 2) % sizeimage] =
            interped * (4 * (tx % 2 - 0.5f) * (ty % 2 - 0.5f));
    }
}

__global__ void polar2cartesian_fourier_angle(complex* cartesian, complex* polar, float* angles, 
                                                size_t nrays, size_t nangles, size_t sizeimage) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y;

    if (tx < sizeimage) {
        size_t cartplane = blockIdx.z * sizeimage * sizeimage;
        polar += blockIdx.z * nrays * nangles;

        int posx = tx - sizeimage / 2;
        int posy = ty - sizeimage / 2;

        float rho = nrays * hypotf(posx, posy) / sizeimage;
        float angle = (nangles) * (0.5f * atan2f(posy, posx) / float(M_PI) + 0.5f);

        size_t irho = size_t(rho);
        int iarc = int(angle);
        complex interped = complex(0.0f);

        if (irho < nrays / 2 - 1) {
            float pfrac = rho - irho;
            float tfrac = iarc - angle;

            iarc = iarc % (nangles);

            int uarc = (iarc + 1) % (nangles);

            complex interp0 = polar[iarc * nrays + irho] * (1.0f - pfrac) + polar[iarc * nrays + irho + 1] * pfrac;
            complex interp1 = polar[uarc * nrays + irho] * (1.0f - pfrac) + polar[uarc * nrays + irho + 1] * pfrac;

            interped = interp0 * tfrac + interp1 * (1.0f - tfrac);
        }

        cartesian[cartplane + sizeimage * ((ty + sizeimage / 2) % sizeimage) + (tx + sizeimage / 2) % sizeimage] =
            interped * (4 * (tx % 2 - 0.5f) * (ty % 2 - 0.5f));
    }
}

void EMFQ_BST(float* blockRecon, float* wholesinoblock, float* angles, int Nrays, int Nangles, int trueblocksize,
              int sizeimage, int pad0) {
    int blocksize = 1;

    cImage cartesianblock(sizeimage, sizeimage * blocksize);
    cImage polarblock(Nrays * pad0, Nangles * blocksize);
    cImage realpolar(Nrays * pad0, Nangles * blocksize);

    cufftHandle plan1d;
    cufftHandle plan2d;

    int dimms1d[] = {(int)Nrays * pad0 / 2};
    int dimms2d[] = {(int)sizeimage, (int)sizeimage};
    int beds[] = {(int)Nrays * pad0 / 2};

    HANDLE_FFTERROR(cufftPlanMany(&plan1d, 1, dimms1d, beds, 1, Nrays * pad0 / 2, beds, 1, Nrays * pad0 / 2, CUFFT_C2C,
                                  Nangles * blocksize * 2));
    HANDLE_FFTERROR(cufftPlanMany(&plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize));

    size_t insize = Nrays * Nangles;
    size_t outsize = sizeimage * sizeimage;

    for (size_t zoff = 0; zoff < (size_t)trueblocksize; zoff += blocksize) {
        float* sinoblock = wholesinoblock + insize * zoff;

        dim3 blocks((Nrays + 255) / 256, Nangles, blocksize);
        dim3 threads(128, 1, 1);

        sino2p<<<blocks, threads>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, pad0, 0);

        Nangles *= 2;
        Nrays *= pad0;
        Nrays /= 2;

        blocks.y *= 2;
        blocks.x *= pad0;
        blocks.x /= 2;

        HANDLE_FFTERROR(cufftExecC2C(plan1d, realpolar.gpuptr, polarblock.gpuptr, CUFFT_FORWARD));
        convBST<<<blocks, threads>>>(polarblock.gpuptr, Nrays, Nangles);

        blocks = dim3((sizeimage + 255) / 256, sizeimage, blocksize);
        threads = dim3(256, 1, 1);

        HANDLE_ERROR(cudaPeekAtLastError());
        polar2cartesian_fourier<<<blocks, threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, angles, Nrays, Nangles,
                                                     sizeimage);

        HANDLE_FFTERROR(cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE));

        cudaDeviceSynchronize();

        GetX<<<dim3((sizeimage + 127) / 128, sizeimage), 128>>>(blockRecon + outsize * zoff, cartesianblock.gpuptr,
                                                                sizeimage, 1.0f);

        HANDLE_ERROR(cudaPeekAtLastError());

        Nangles /= 2;
        Nrays *= 2;
        Nrays /= pad0;
    }
    cufftDestroy(plan1d);
    cufftDestroy(plan2d);
}

void EMFQ_BST_ITER(float* blockRecon, float* wholesinoblock, float* angles, cImage& cartesianblock, cImage& polarblock,
                   cImage& realpolar, cufftHandle plan1d, cufftHandle plan2d, int Nrays, int Nangles, int trueblocksize,
                   int blocksize, int sizeimage, int pad0) {
    size_t insize = Nrays * Nangles;
    size_t outsize = sizeimage * sizeimage;

    for (size_t zoff = 0; zoff < (size_t)trueblocksize; zoff += blocksize) {
        float* sinoblock = wholesinoblock + insize * zoff;

        dim3 blocks((Nrays + 255) / 256, Nangles, blocksize);
        dim3 threads(128, 1, 1);

        sino2p<<<blocks, threads>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, pad0, 0);

        Nangles *= 2;
        Nrays *= pad0;
        Nrays /= 2;

        blocks.y *= 2;
        blocks.x *= pad0;
        blocks.x /= 2;

        HANDLE_FFTERROR(cufftExecC2C(plan1d, realpolar.gpuptr, polarblock.gpuptr, CUFFT_FORWARD));
        convBST<<<blocks, threads>>>(polarblock.gpuptr, Nrays, Nangles);

        blocks = dim3((sizeimage + 255) / 256, sizeimage, blocksize);
        threads = dim3(256, 1, 1);

        polar2cartesian_fourier<<<blocks, threads>>>(cartesianblock.gpuptr, polarblock.gpuptr, angles, Nrays, Nangles,
                                                     sizeimage);

        HANDLE_FFTERROR(cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE));

        cudaDeviceSynchronize();

        GetX<<<dim3((sizeimage + 127) / 128, sizeimage), 128>>>(blockRecon + outsize * zoff, cartesianblock.gpuptr,
                                                                sizeimage, 1.0f);

        HANDLE_ERROR(cudaPeekAtLastError());

        Nangles /= 2;
        Nrays *= 2;
        Nrays /= pad0;
    } 
}


void getBST(float* blockRecon, float* wholesinoblock, float* angles, 
int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0, 
float reg, float paganin, int filter_type, float offset, float pixel, 
WBST *bst_workspace, cudaStream_t stream) 
{
    int blocksize_bst = 1;

    size_t insize  =     Nrays *   Nangles;
    size_t outsize = sizeimage * sizeimage;

    Filter filter(filter_type, paganin, reg, offset, pixel, 0, 1);

    /* BST initialization finishes here */

    for (size_t zoff = 0; zoff < (size_t)trueblocksize; zoff += blocksize_bst) {

        float* sinoblock = wholesinoblock + insize * zoff;

        if (filter.type != Filter::EType::none)
            BSTFilter_pad(bst_workspace->filterplan, bst_workspace->filtersino->gpuptr, sinoblock, Nrays, Nangles, filter, stream);

        dim3 blocks((Nrays + 255) / 256, Nangles, blocksize_bst);
        dim3 threads(128, 1, 1);

        sino2p<<<blocks, threads, 0, stream>>>(bst_workspace->realpolar->gpuptr, sinoblock, Nrays, Nangles, pad0, 0);

        Nangles *= 2;
        Nrays *= pad0;
        Nrays /= 2;

        blocks.y *= 2;
        blocks.x *= pad0;
        blocks.x /= 2;

        HANDLE_FFTERROR(cufftExecC2C(bst_workspace->plan1d, bst_workspace->realpolar->gpuptr, bst_workspace->polarblock->gpuptr, CUFFT_FORWARD));
        convBST<<<blocks, threads, 0, stream>>>(bst_workspace->polarblock->gpuptr, Nrays, Nangles);

        blocks = dim3((sizeimage + 255) / 256, sizeimage, blocksize_bst);
        threads = dim3(256, 1, 1);

        polar2cartesian_fourier<<<blocks, threads, 0, stream>>>(bst_workspace->cartesianblock->gpuptr, bst_workspace->polarblock->gpuptr, angles,
                                                                Nrays, Nangles, sizeimage);

        HANDLE_FFTERROR(cufftExecC2C(bst_workspace->plan2d, bst_workspace->cartesianblock->gpuptr, bst_workspace->cartesianblock->gpuptr, CUFFT_INVERSE));

        // cudaDeviceSynchronize();
        Nangles /= 2;
        Nrays *= 2;
        Nrays /= pad0;

        float scale = (float)Nrays * pixel * 4.0f;

        GetX<<<dim3((sizeimage + 127) / 128, sizeimage), 128, 0, stream>>>( blockRecon + outsize * zoff,
                                                                            bst_workspace->cartesianblock->gpuptr, 
                                                                            sizeimage,  scale);

        HANDLE_ERROR(cudaPeekAtLastError());
    }
}

extern "C" {

    void getBSTGPU(DIM tomo, DIM obj, GEO geometry, REC ReconParam,
    float* object, float* tomogram, float* angles, 
    int blockgpu, int gpu, int nstreams) 
    {
        HANDLE_ERROR(cudaSetDevice(gpu));

        const int blocksize_bst = 1;
        int blocksize           = tomo.blocksize;

        size_t total_required_mem_per_slice_bytes = (
            calcSliceMemoryBytes(tomo)           + // Tomo slice
            2 * calcSliceMemoryBytes(obj)        + // Reconstructed object slice
            2 * calcPaddedSliceMemoryBytes(obj)  + // Reconstructed object padded slice
            2 * calcPaddedSliceMemoryBytes(tomo) + // Tomo padded slice 
            tomo.size.y * sizeof(float)            // angles
            ); 

        if (blocksize == 0) {
            int blocksize_aux = compute_GPU_blocksize(  blockgpu, 
                                                        (size_t)nstreams * total_required_mem_per_slice_bytes, 
                                                        true, 
                                                        BYTES_TO_GB * getTotalDeviceMemory());
            blocksize = min(blockgpu, blocksize_aux);
        }
        int ind_block = (int)ceil((float)blockgpu / blocksize);
        int ptr = 0, subblock;

        /* Projection data sizes */
        int nrays    = PDIM(tomo.size.x,tomo.pad.x); // tomo.size.x * (1 + tomo.pad.x);
        int nangles  = tomo.size.y;

        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock( (int)ceil(     nrays / TPBX ) + 1,
                            (int)ceil(   nangles / TPBY ) + 1,
                            (int)ceil( blocksize / TPBZ ) + 1);

        /* Reconstruction sizes */
        int sizeImagex = PDIM(obj.size.x,obj.pad.x); // obj.size.x * (1 + obj.pad.x);

        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 ObjthreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 ObjgridBlock(  (int)ceil( sizeImagex / TPBX ) + 1,
                            (int)ceil( sizeImagex / TPBY ) + 1,
                            (int)ceil(  blocksize / TPBZ ) + 1);

        // int padx  = PADS(obj.size.x,obj.pad.x); 
        // int padt  = PADS(tomo.size.x,tomo.pad.x);
        // Log("Streams: Size tomo");
        // printDim(tomo.size);
        // Log("Pad tomo");
        // printDim(tomo.pad);
        // printf("TOMO: nrayspad = %d\n", nrays);
        // printf("TOMO: padx = %d\n", padt);
        // Log("Size obj");
        // printDim(obj.size);
        // Log("Pad obj");
        // printDim(obj.pad);
        // printf("OBJ: padImagex = %d \n", sizeImagex);
        // printf("OBJ: padx = %d \n", padx);
        // printf("padding_mode = %d \n", tomo.padding_mode);
        // fflush(stdout);

                            
        int bst_padd      = 8; /* Fix this padding for we will padd the data before this */
        int filter_type   = ReconParam.filter;
        float paganin_reg = ReconParam.paganin_slices;
        float filter_reg  = ReconParam.filter_reg;
        float axis_offset = ReconParam.rotation_axis_offset;
        float pixel       = geometry.obj_pixel.x;

        float* dangles = opt::allocGPU<float>(nangles);

        opt::CPUToGPU<float>(angles, dangles, nangles);

        float* dtomo[nstreams];
        float* dobj[nstreams];
        float* dtomoPadded[nstreams];
        float* dobjPadded[nstreams];
        cudaStream_t streams[nstreams];

        WBST **bst_workspace = (WBST**)malloc(sizeof(WBST*) * nstreams);

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamCreate(&streams[st]);

            bst_workspace[st] = InitializeBST_workspace(dim3(     nrays,   nangles,blocksize), 
                                                         dim3(sizeImagex,sizeImagex,blocksize), 
                                                         bst_padd, blocksize_bst);

            cufftSetStream(bst_workspace[st]->plan1d    , streams[st]);
            cufftSetStream(bst_workspace[st]->plan2d    , streams[st]);
            cufftSetStream(bst_workspace[st]->filterplan, streams[st]);

            dtomo[st] = opt::allocGPU<float>((size_t)tomo.size.x *     nangles * blocksize, streams[st]);
            dobj[st]  = opt::allocGPU<float>((size_t) obj.size.x *  obj.size.y * blocksize, streams[st]);

            dtomoPadded[st] = opt::allocGPU<float>((size_t)     nrays *    nangles * blocksize, streams[st]);
            dobjPadded[st]  = opt::allocGPU<float>((size_t)sizeImagex * sizeImagex * blocksize, streams[st]);
        }
 
        for (int i = 0; i < ind_block; ++i){
            int st = i % nstreams;
            cudaStream_t stream = streams[i % nstreams];

            subblock = min(blockgpu - ptr, (int)blocksize);

            opt::CPUToGPU<float>(tomogram + (size_t)ptr * tomo.size.x * nangles, 
                                dtomo[st], 
                                (size_t)tomo.size.x * nangles * subblock,
                                stream);

            /* Padding the tomogram data */
            TomogridBlock.z = (int)ceil( subblock / TPBZ ) + 1;
            opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock,0,stream>>>(   dtomo[st], dtomoPadded[st], tomo.padding_mode,
                                                                            dim3(tomo.size.x, tomo.size.x, subblock),
                                                                            tomo.pad);
            getBST( dobjPadded[st], dtomoPadded[st],
                    dangles, nrays, nangles, subblock, sizeImagex, 
                    bst_padd, filter_reg, paganin_reg,
                    filter_type, axis_offset, pixel, bst_workspace[st], stream);

            /* Remove padd from the object (reconstruction) */
            ObjgridBlock.z = TomogridBlock.z;
            opt::remove_paddR2R<<<ObjgridBlock,ObjthreadsPerBlock,0,stream>>>(  dobjPadded[st], dobj[st], 
                                                                                dim3(obj.size.x, obj.size.x, subblock), 
                                                                                obj.pad);

            opt::GPUToCPU<float>(object +  size_t(ptr * obj.size.x * obj.size.y), 
                                dobj[st],
                                size_t(obj.size.x * obj.size.y * subblock), 
                                stream);

            /* Update pointer */
            ptr = ptr + subblock;
        }

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamSynchronize(streams[st]);

            freeBSTWorkspace(bst_workspace[st]);

            HANDLE_ERROR(cudaFreeAsync(dtomo[st], streams[st]));
            HANDLE_ERROR(cudaFreeAsync(dobj[st], streams[st]));
            HANDLE_ERROR(cudaFreeAsync(dtomoPadded[st], streams[st]));
            HANDLE_ERROR(cudaFreeAsync(dobjPadded[st], streams[st]));

            cudaStreamDestroy(streams[st]);
        }
        free(bst_workspace);
        HANDLE_ERROR(cudaFree(dangles));
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    void getBSTMultiGPU(DIM tomo, DIM obj, GEO geometry, REC ReconParam,
    int* gpus, int ngpus, float* object, float* tomogram, float* angles,
    int nstreams) 
    {
        int i, Maxgpudev;

        /* Multiples devices */
        HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

        /* If devices input are larger than actual devices on GPU, exit */
        for (i = 0; i < ngpus; i++) assert(gpus[i] < Maxgpudev && "Invalid device number.");

        /* Projection data sizes */
        int nrays   = tomo.size.x;
        int nangles = tomo.size.y;
        int nslices = tomo.size.z;

        /* Reconstruction sizes */
        int sizeImagex = obj.size.x;

        int blockgpu = (nslices + ngpus - 1) / ngpus;
        int subblock, ptr = 0;
        std::vector<std::future<void>> threads;
        threads.reserve(ngpus);


        if (  ngpus == 1 ){

            getBSTGPU(tomo, obj, geometry, ReconParam, object, tomogram, angles, nslices, gpus[0], nstreams);
        
        }else{
            
            for (i = 0; i < ngpus; i++) {
                subblock = min(nslices - ptr, blockgpu);

                threads.push_back(std::async(std::launch::async, getBSTGPU, tomo, obj, geometry, ReconParam,
                                            object   + (size_t)ptr * sizeImagex * sizeImagex,
                                            tomogram + (size_t)ptr *      nrays * nangles, 
                                            angles, subblock, gpus[i], nstreams));
                /* Update pointer */
                ptr = ptr + subblock;
            }
            for (i = 0; i < ngpus; i++) threads[i].get();
        }
    }
}

