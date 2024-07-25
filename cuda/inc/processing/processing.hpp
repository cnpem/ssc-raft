#ifndef RAFT_PROCESSING_H
#define RAFT_PROCESSING_H

#include <driver_types.h>
#define PADDING 32

#include "common/complex.hpp"
#include "common/configs.hpp"

/* Background Correction */
extern "C"{

	void getBackgroundCorrectionMultiGPU(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nangles, int nslices, int numflats, int is_log);

	void getBackgroundCorrectionGPU(GPU gpus, int gpu, float* frames, float* flat, float* dark, dim3 size, int numflats, int is_log);

	void getBackgroundCorrection(GPU gpus, float* frames, float* flat, float* dark, dim3 size, int numflats, cudaStream_t stream = 0);
}

/* Rings */
extern "C"{
    
    void getTitarenkoRingsMultiGPU(int *gpus, int ngpus, float *data, int nrays, int nangles, int nslices, float lambda_rings, int ring_blocks);

    void getTitarenkoRingsGPU(GPU gpus, int gpu, float *data, dim3 size, float lambda_rings, int ring_blocks);

    void getTitarenkoRings(GPU gpus, float *tomogram, dim3 size, float lambda_rings, int ring_blocks, cudaStream_t stream = 0);
}

/* Rotation Axis */

extern "C" {
    void getCorrectRotationAxis(float* d_tomo_in, float* d_tomo_out,
            dim3 tomo_size, int deviation);
}

/* Centersino - Find offset for 180 degrees parallel tomogram */
extern "C"{

    int findcentersino(float* frame0, float* frame180, 
    float* dark, float* flat, int sizex, int sizey);

    int getCentersino(float* frame0, float* frame180, 
    float* dark, float* flat, size_t sizex, size_t sizey);
}

/* Phase retrieval Functions */
extern "C"{

    void getPhaseMultiGPU(int *gpus, int ngpus, 
    float *projections, float *paramf, int *parami);

    void getPhase(CFG configs, GPU gpus, float *kernel, float *projections, 
    dim3 size, dim3 size_pad);
}

namespace phase_pag{ // Phase retrieval Paganin

    __global__ void padding(float *in, cufftComplex *inpadded, dim3 size, dim3 pad);
    __global__ void recuperate_padding(cufftComplex *inpadded, float *in, dim3 size, dim3 pad);

    __global__ void paganinKernel(float *kernel, float beta_delta, float wavelength, 
    float pixel_objx, float pixel_objy, float z2, dim3 size);

    void apply_paganin_filter(CFG configs, GPU gpus, float *projections, float *kernel,
    dim3 size, dim3 size_pad, dim3 pad);

    __global__ void mult(cufftComplex *a, float *b, cufftComplex *ans, dim3 size);

    __global__ void copy(float *projection, float *kernel, dim3 size);
}

#endif
