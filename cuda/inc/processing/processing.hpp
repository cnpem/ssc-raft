#ifndef RAFT_PROCESSING_H
#define RAFT_PROCESSING_H

#include <driver_types.h>
#define PADDING 32

#include "common/complex.hpp"
#include "common/configs.hpp"

/* Background Correction */
extern "C"{
	void getBackgroundCorrectionMultiGPU(int* gpus, int ngpus, float* frames, float* flat, float* dark, 
    int nrays, int nangles, int nslices, int numflats, int is_log, int input_slices, int blocksize);

	void getBackgroundCorrectionGPU(int gpu, float* frames, float* flat, float* dark, 
    dim3 size, int numflats, int is_log, int input_slices, int blocksize);

    void getBackgroundCorrection_slices(float* frames, float* flat, float* dark, 
        dim3 size, int numflats, int is_log, cudaStream_t stream);
    
    void getBackgroundCorrection_frames(float* frames, float* flat, float* dark, 
            dim3 size, int numflats, int is_log, cudaStream_t stream);
}

/* Rings */
extern "C"{
    void getTitarenkoRingsMultiGPU(int *gpus, int ngpus, float *data, 
    int nrays, int nangles, int nslices, 
    float lambda_rings, int ring_blocks, int blocksize);

    void getTitarenkoRingsGPU(int gpu, float *data, dim3 size, 
    float lambda_rings, int ring_blocks, int blocksize);

    void getTitarenkoRings(float *tomogram, dim3 size, 
    float lambda_rings, int ring_blocks, cudaStream_t stream = 0);
}

/* Rotation Axis */

extern "C" {
    void getCorrectRotationAxis(float* d_tomo_in, float* d_tomo_out,
            dim3 tomo_size, int deviation);

    void getRotAxisCorrectionMultiGPU(int* gpus, int ngpus, 
    float* tomogram, float axis_offset, 
    int nrays, int nangles, int nslices, int blocksize);

    void getRotAxisCorrectionGPU(float *tomogram, 
    float axis_offset, dim3 tomo_size, int ngpu, int blocksize);
}

/* Centersino - Find offset for 180 degrees parallel tomogram */
extern "C"{

    float findcentersino_subpixel(float* frame0, float* frame180, 
    float* dark, float* flat, int sizex, int sizey);

    float getCentersino_subpixel(float* frame0, float* frame180, 
    float* dark, float* flat, size_t sizex, size_t sizey);

    int getCentersino(float* frame0, float* frame180, 
    float* dark, float* flat, size_t sizex, size_t sizey);
}

/* Contrast Enhancement Functions */
extern "C"{

    void getContrastEnhencementMultiGPU(DIM tomo, GEO geometry, CEF ContrastFilter,
    int *gpus, int ngpus, float *projections);

    void getContrastEnhencementGPU(DIM tomo, GEO geometry, CEF ContrastFilter,
    float *projections, int sizez, int ngpu);
}

namespace contrast_enhance{ // Phase retrieval Paganin

    enum ContrastEnhanceType
    {
        none           = 0,
        paganin        = 1,
        paganin_slices = 2,
        contrast       = 3
    };

    __global__ void paganinKernel(float *kernel, float beta_delta, float wavelength, 
    float pixel_objx, float pixel_objy, float z2, dim3 size);

    __global__ void multiplication(cufftComplex *a, float *b, cufftComplex *ans, dim3 size);

    __global__ void contrast_paganin_based_Kernel(float *kernel, float regularization, dim3 size);
}

namespace denoise{}

#endif
