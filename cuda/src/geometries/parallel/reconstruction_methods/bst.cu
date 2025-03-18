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

// void getBST(CFG configs, GPU gpus,
//         float* obj, float* tomo, float* angles,
//         dim3 tomo_size, dim3 tomo_pad, dim3 obj_size, cudaStream_t stream) {

//     int blocksize_bst = 1;

//     const int filter_type   = configs.reconstruction_filter_type;
//     int Nrays               = tomo_size.x;
//     int Nangles             = tomo_size.y;
//     const int sizeimage     = obj_size.x;
//     const float reg         = configs.reconstruction_reg;
//     const float paganin     = configs.reconstruction_paganin;
//     const float axis_offset = configs.rotation_axis_offset;
//     const int trueblocksize = tomo_size.z;
//     const int padding       = configs.tomo.pad.x;
//     float pixel             = configs.geometry.obj_pixel_x;

//     size_t insize  = Nrays * Nangles;
//     size_t outsize = sizeimage * sizeimage;


//     int dimmsfilter[] = {Nrays};
//     int dimms1d[] = {(int)Nrays * padding / 2};
//     int dimms2d[] = {(int)sizeimage, (int)sizeimage};
//     int beds[] = {Nrays * padding / 2};


//     cufftHandle plan1d;
//     cufftHandle plan2d;
//     cufftHandle filterplan;

//     HANDLE_FFTERROR(cufftPlanMany(&plan1d, 1, dimms1d, beds, 1, Nrays * padding / 2, beds, 1,
//                                   Nrays * padding / 2, CUFFT_C2C, Nangles * blocksize_bst * 2));
//     HANDLE_FFTERROR(
//         cufftPlanMany(&plan2d, 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize_bst));
//     HANDLE_FFTERROR(cufftPlanMany(&filterplan, 1, dimmsfilter, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C,
//                                   Nangles * blocksize_bst));

//     cufftSetStream(plan1d, stream);
//     cufftSetStream(plan2d, stream);
//     cufftSetStream(filterplan, stream);

//     cImage filtersino(Nrays, Nangles * blocksize_bst, 1, MemoryType::EAllocGPU, stream);
//     cImage cartesianblock(sizeimage, sizeimage * blocksize_bst, 1, MemoryType::EAllocGPU, stream);
//     cImage polarblock(Nrays * padding, Nangles * blocksize_bst, 1, MemoryType::EAllocGPU, stream);
//     cImage realpolar(Nrays * padding, Nangles * blocksize_bst, 1, MemoryType::EAllocGPU, stream);

//     Filter filter(filter_type, paganin, reg, axis_offset, pixel);

//     // BST initialization finishes here.

//     for (size_t zoff = 0; zoff < (size_t)trueblocksize; zoff += blocksize_bst) {
//         float* sinoblock = tomo + insize * zoff;

//         if (filter.type != Filter::EType::none)
//             BSTFilter(filterplan, filtersino.gpuptr, sinoblock, Nrays, Nangles, axis_offset, filter, pixel, stream);

//         dim3 blocks((Nrays + 255) / 256, Nangles, blocksize_bst);
//         dim3 threads(128, 1, 1);

//         sino2p<<<blocks, threads, 0, stream>>>(realpolar.gpuptr, sinoblock, Nrays, Nangles, padding, 0);

//         Nangles *= 2;
//         Nrays *= padding;
//         Nrays /= 2;

//         blocks.y *= 2;
//         blocks.x *= padding;
//         blocks.x /= 2;

//         HANDLE_FFTERROR(cufftExecC2C(plan1d, realpolar.gpuptr, polarblock.gpuptr, CUFFT_FORWARD));
//         convBST<<<blocks, threads, 0, stream>>>(polarblock.gpuptr, Nrays, Nangles);

//         blocks = dim3((sizeimage + 255) / 256, sizeimage, blocksize_bst);
//         threads = dim3(256, 1, 1);

//         polar2cartesian_fourier<<<blocks, threads, 0, stream>>>(cartesianblock.gpuptr, polarblock.gpuptr, angles,
//                                                                 Nrays, Nangles, sizeimage);

//         HANDLE_FFTERROR(cufftExecC2C(plan2d, cartesianblock.gpuptr, cartesianblock.gpuptr, CUFFT_INVERSE));

//         // cudaDeviceSynchronize();

//         Nangles /= 2;
//         Nrays *= 2;
//         Nrays /= padding;

//         float scale = (float)Nrays * pixel * 4.0f;

//         GetX<<<dim3((sizeimage + 127) / 128, sizeimage), 128, 0, stream>>>(obj + outsize * zoff,
//                                                                            cartesianblock.gpuptr, 
//                                                                            sizeimage, scale);

//         HANDLE_ERROR(cudaPeekAtLastError());
//     }

// }

    void getBST(float* blockRecon, float* wholesinoblock, float* angles, 
    int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0, 
    float reg, float paganin, int filter_type, float offset, float pixel, 
    cufftHandle plan1d, cufftHandle plan2d, cufftHandle filterplan, 
    cImage* filtersino, cImage* cartesianblock, cImage* polarblock, cImage* realpolar, 
    int gpu, cudaStream_t stream = 0) 
    {
        // HANDLE_ERROR(cudaSetDevice(gpu));

        int blocksize_bst = 1;

        size_t insize  =     Nrays *   Nangles;
        size_t outsize = sizeimage * sizeimage;

        Filter filter(filter_type, paganin, reg, offset, pixel);

        /* BST initialization finishes here */

        for (size_t zoff = 0; zoff < (size_t)trueblocksize; zoff += blocksize_bst) {
            float* sinoblock = wholesinoblock + insize * zoff;

            if (filter.type != Filter::EType::none)
                BSTFilter(filterplan, filtersino->gpuptr, sinoblock, Nrays, Nangles, offset, filter, pixel, stream);

            dim3 blocks((Nrays + 255) / 256, Nangles, blocksize_bst);
            dim3 threads(128, 1, 1);

            sino2p<<<blocks, threads, 0, stream>>>(realpolar->gpuptr, sinoblock, Nrays, Nangles, pad0, 0);

            Nangles *= 2;
            Nrays *= pad0;
            Nrays /= 2;

            blocks.y *= 2;
            blocks.x *= pad0;
            blocks.x /= 2;

            HANDLE_FFTERROR(cufftExecC2C(plan1d, realpolar->gpuptr, polarblock->gpuptr, CUFFT_FORWARD));
            convBST<<<blocks, threads, 0, stream>>>(polarblock->gpuptr, Nrays, Nangles);

            blocks = dim3((sizeimage + 255) / 256, sizeimage, blocksize_bst);
            threads = dim3(256, 1, 1);

            polar2cartesian_fourier<<<blocks, threads, 0, stream>>>(cartesianblock->gpuptr, polarblock->gpuptr, angles,
                                                                    Nrays, Nangles, sizeimage);

            HANDLE_FFTERROR(cufftExecC2C(plan2d, cartesianblock->gpuptr, cartesianblock->gpuptr, CUFFT_INVERSE));

            // cudaDeviceSynchronize();
            Nangles /= 2;
            Nrays *= 2;
            Nrays /= pad0;

            float scale = (float)Nrays * pixel * 4.0f;

            GetX<<<dim3((sizeimage + 127) / 128, sizeimage), 128, 0, stream>>>( blockRecon + outsize * zoff,
                                                                                cartesianblock->gpuptr, 
                                                                                sizeimage,  scale);

            HANDLE_ERROR(cudaPeekAtLastError());
        }
    }

extern "C" {

    void getBSTGPU(CFG configs, 
    float* obj, float* tomo, float* angles, 
    int blockgpu, int gpu) 
    {
        HANDLE_ERROR(cudaSetDevice(gpu));

        const int blocksize_bst = 1;

        /* Projection data sizes */
        int nrays   = configs.tomo.padsize.x;
        int nangles = configs.tomo.padsize.y;

        /* Projection GPUs padded Grd and Blocks */
        dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 TomogridBlock( (int)ceil( configs.tomo.padsize.x / TPBX ) + 1,
                            (int)ceil( configs.tomo.padsize.y / TPBY ) + 1,
                            (int)ceil( configs.tomo.padsize.z / TPBZ ) + 1);

        /* Reconstruction sizes */
        // int sizeImagex = configs.obj.padsize.x;
        int sizeImagex = configs.obj.size.x;

        /* Reconstruction GPUs padded Grd and Blocks */
        dim3 ObjthreadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 ObjgridBlock(  (int)ceil( configs.obj.padsize.x / TPBX ) + 1,
                            (int)ceil( configs.obj.padsize.y / TPBY ) + 1,
                            (int)ceil( configs.obj.padsize.z / TPBZ ) + 1);

        int bst_padd         = 2; /* Fix this padding for we will padd the data before this */
        int filter_type      = configs.reconstruction_filter_type;
        float paganin_reg    = configs.reconstruction_paganin;
        float regularization = configs.reconstruction_reg;
        float axis_offset    = configs.rotation_axis_offset;
        float pixel          = configs.geometry.obj_pixel_x;

        int blocksize = configs.blocksize;

        if (blocksize == 0) {
            int blocksize_aux = compute_GPU_blocksize(blockgpu, configs.total_required_mem_per_slice_bytes, true, A100_MEM);
            blocksize = min(blockgpu, blocksize_aux);
        }
        int ind_block = (int)ceil((float)blockgpu / blocksize);
        int ptr = 0, subblock;

        float* dangles = opt::allocGPU<float>(nangles);

        opt::CPUToGPU<float>(angles, dangles, nangles);

        int dimmsfilter[] = {nrays};
        int dimms1d[]     = {(int)nrays * bst_padd / 2};
        int dimms2d[]     = {(int)sizeImagex, (int)sizeImagex};
        int beds[]        = {nrays * bst_padd / 2};

        const int nstreams = 2;
        float* dtomo[nstreams];
        float* dobj[nstreams];
        float* dtomoPadded[nstreams];
        float* dobjPadded[nstreams];
        cudaStream_t streams[nstreams];
        cufftHandle plans1d[nstreams];
        cufftHandle plans2d[nstreams];
        cufftHandle filterplans[nstreams];

        cImage* filtersino[nstreams];
        cImage* cartesianblock[nstreams];
        cImage* polarblock[nstreams];
        cImage* realpolar[nstreams];

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamCreate(&streams[st]);

            HANDLE_FFTERROR(cufftPlanMany(&plans1d[st], 1, dimms1d, beds, 1, nrays * bst_padd / 2, beds, 1, nrays * bst_padd / 2, CUFFT_C2C, nangles * blocksize_bst * 2));
            HANDLE_FFTERROR(cufftPlanMany(&plans2d[st], 2, dimms2d, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, blocksize_bst));
            HANDLE_FFTERROR(cufftPlanMany(&filterplans[st], 1, dimmsfilter, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2C, nangles * blocksize_bst));

            cufftSetStream(    plans1d[st], streams[st]);
            cufftSetStream(    plans2d[st], streams[st]);
            cufftSetStream(filterplans[st], streams[st]);

            filtersino[st]     = new cImage(           nrays,    nangles * blocksize_bst, 1, MemoryType::EAllocGPU, streams[st]);
            cartesianblock[st] = new cImage(      sizeImagex, sizeImagex * blocksize_bst, 1, MemoryType::EAllocGPU, streams[st]);
            polarblock[st]     = new cImage(nrays * bst_padd,    nangles * blocksize_bst, 1, MemoryType::EAllocGPU, streams[st]);
            realpolar[st]      = new cImage(nrays * bst_padd,    nangles * blocksize_bst, 1, MemoryType::EAllocGPU, streams[st]);

            dtomo[st] = opt::allocGPU<float>((size_t)configs.tomo.size.x *            nangles * blocksize, streams[st]);
            dobj[st]  = opt::allocGPU<float>((size_t) configs.obj.size.x * configs.obj.size.y * blocksize, streams[st]);

            dtomoPadded[st] = opt::allocGPU<float>((size_t)     nrays *    nangles * blocksize, streams[st]);
            dobjPadded[st]  = opt::allocGPU<float>((size_t)sizeImagex * sizeImagex * blocksize, streams[st]);
        }
 
        for (int i = 0; i < ind_block; ++i){
            int st = i % nstreams;
            cudaStream_t stream = streams[i % nstreams];

            subblock = min(blockgpu - ptr, (int)blocksize);

            opt::CPUToGPU<float>(tomo + (size_t)ptr * configs.tomo.size.x * nangles, 
                                dtomo[st], 
                                (size_t)configs.tomo.size.x * nangles * subblock,
                                stream);

            /* Padding the tomogram data */
            TomogridBlock.z = (int)ceil( subblock / TPBZ ) + 1;
            opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock,0,stream>>>(   dtomo[st], dtomoPadded[st], 
                                                                            dim3(configs.tomo.size.x, configs.tomo.size.x, subblock),
                                                                            configs.tomo.pad);
            getBST( dobj[st], dtomoPadded[st],
                    dangles, nrays, nangles, subblock, sizeImagex, 
                    bst_padd, regularization, paganin_reg,
                    filter_type, axis_offset, pixel, 
                    plans1d[st], plans2d[st], filterplans[st],
                    filtersino[st], cartesianblock[st], polarblock[st], realpolar[st],
                    gpu, stream);

            /* Remove padd from the object (reconstruction) */
            // ObjgridBlock.z = TomogridBlock.z;
            // opt::remove_paddR2R<<<ObjgridBlock,ObjthreadsPerBlock,0,stream>>>(  dobjPadded[st], dobj[st], 
            //                                                                     dim3(configs.obj.size.x, configs.obj.size.x, subblock), 
            //                                                                     configs.obj.pad);

            opt::GPUToCPU<float>(obj +  size_t(ptr * configs.obj.size.x * configs.obj.size.y), 
                                dobj[st],
                                size_t(configs.obj.size.x * configs.obj.size.y * subblock), 
                                stream);

            /* Update pointer */
            ptr = ptr + subblock;
        }

        for (int st = 0; st < nstreams; ++st) {
            cudaStreamSynchronize(streams[st]);

            HANDLE_FFTERROR(cufftDestroy(plans1d[st]));
            HANDLE_FFTERROR(cufftDestroy(plans2d[st]));
            HANDLE_FFTERROR(cufftDestroy(filterplans[st]));

            HANDLE_ERROR(cudaFreeAsync(dtomo[st], streams[st]));
            HANDLE_ERROR(cudaFreeAsync(dobj[st], streams[st]));
            HANDLE_ERROR(cudaFreeAsync(dtomoPadded[st], streams[st]));
            HANDLE_ERROR(cudaFreeAsync(dobjPadded[st], streams[st]));

            delete filtersino[st];
            delete cartesianblock[st];
            delete polarblock[st];
            delete realpolar[st];

            cudaStreamDestroy(streams[st]);
        }

        HANDLE_ERROR(cudaFree(dangles));
        HANDLE_ERROR(cudaDeviceSynchronize());
    }

    void getBSTMultiGPU(int* gpus, int ngpus, 
    float* obj, float* tomogram, float* angles, 
    float* paramf, int* parami) 
    {
        int i, Maxgpudev;

        /* Multiples devices */
        HANDLE_ERROR(cudaGetDeviceCount(&Maxgpudev));

        /* If devices input are larger than actual devices on GPU, exit */
        for (i = 0; i < ngpus; i++) assert(gpus[i] < Maxgpudev && "Invalid device number.");
        CFG configs;
        GPU gpu_parameters;

        setBSTParameters(&configs, paramf, parami);
        // printBSTParameters(&configs);

        /* Projection data sizes */
        int nrays   = configs.tomo.size.x;
        int nangles = configs.tomo.size.y;
        int nslices = configs.tomo.size.z;

        /* Reconstruction sizes */
        int sizeImagex = configs.obj.size.x;

        int blockgpu = (nslices + ngpus - 1) / ngpus;
        int subblock, ptr = 0;
        std::vector<std::future<void>> threads;
        threads.reserve(ngpus);

        for (i = 0; i < ngpus; i++) {
            subblock = min(nslices - ptr, blockgpu);

            threads.push_back(std::async(std::launch::async, getBSTGPU, configs,
                                        obj      + (size_t)ptr * sizeImagex * sizeImagex,
                                        tomogram + (size_t)ptr *      nrays * nangles, 
                                        angles, subblock, gpus[i]));

            /* Update pointer */
            ptr = ptr + subblock;
        }
        for (i = 0; i < ngpus; i++) threads[i].get();
    }
}

