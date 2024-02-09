#ifndef RAFT_PROCESSING_H
#define RAFT_PROCESSING_H

#define PADDING 32

#include "common/complex.hpp"
#include "common/configs.hpp"

/* Flat/Dark Correction */
extern "C"{

	void getFlatDarkMultiGPU(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nangles, int nslices, int numflats, int is_log);

	void getFlatDarkGPU(GPU gpus, int gpu, float* frames, float* flat, float* dark, dim3 size, int numflats, int is_log);

	void getFlatDarkCorrection(float* frames, float* flat, float* dark, dim3 size, int numflats, GPU gpus);
}

/* Rings */
extern "C"{
    
    void getRingsMultiGPU(int *gpus, int ngpus, float *data, float *lambda_computed, int nrays, int nangles, int nslices, float lambda_rings, int ring_blocks);

    void getRingsGPU(GPU gpus, int gpu, float *data, float *lambda_computed, dim3 size, float lambda_rings, int ring_blocks);

    float getRings(float *tomogram, dim3 size, float lambda_rings, int ring_blocks, GPU gpus);
}

/* Rotation Axis */

/* Centersino - Find offset for 180 degrees parallel tomogram */
extern "C"{

    int findcentersino(float* frame0, float* frame180, 
    float* dark, float* flat, int sizex, int sizey);

    int getCentersino(float* frame0, float* frame180, 
    float* dark, float* flat, size_t sizex, size_t sizey);
}


#endif