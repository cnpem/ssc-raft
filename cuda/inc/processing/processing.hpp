#ifndef RAFT_PROCESSING_H
#define RAFT_PROCESSING_H

#define PADDING 32

#include "common/complex.hpp"
#include "common/configs.hpp"

/* Background Correction */
extern "C"{

	void getBackgroundCorrectionMultiGPU(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nangles, int nslices, int numflats, int is_log);

	void getBackgroundCorrectionGPU(GPU gpus, int gpu, float* frames, float* flat, float* dark, dim3 size, int numflats, int is_log);

	void getBackgroundCorrection(GPU gpus, float* frames, float* flat, float* dark, dim3 size, int numflats);
}

/* Rings */
extern "C"{
    
    void getTitarenkoRingsMultiGPU(int *gpus, int ngpus, float *data, int nrays, int nangles, int nslices, float lambda_rings, int ring_blocks);

    void getTitarenkoRingsGPU(GPU gpus, int gpu, float *data, dim3 size, float lambda_rings, int ring_blocks);

    void getTitarenkoRings(GPU gpus, float *tomogram, dim3 size, float lambda_rings, int ring_blocks);
}

/* Rotation Axis */

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

    void getPhase(CFG configs, GPU gpus, float *projections, 
    dim3 size, dim3 size_pad);

    void _paganin_gpu(CFG configs, GPU gpus, float *projections,
    dim3 size, dim3 size_pad, dim3 pad);

    void _paganin_gpu_tomopy(CFG configs, GPU gpus, float *projections,
    dim3 size, dim3 size_pad, dim3 pad);

    void _paganin_gpu_v0(CFG configs, GPU gpus, float *projections,
    dim3 size, dim3 size_pad, dim3 pad);

}


#endif