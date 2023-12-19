#ifndef RAFT_PROCESSING_H
#define RAFT_PROCESSING_H

#define PADDING 32

#include "./common/complex.hpp"
#include "./common/configs.h"

/* Flat/Dark Correction */
extern "C"{

	void getFlatDarkMultiGPU(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nangles, int nslices, int numflats, int is_log);

	void getFlatDarkGPU(GPU gpus, int gpu, float* frames, float* flat, float* dark, dim3 size, int numflats, int is_log);

	void getFlatDarkCorrection(float* frames, float* flat, float* dark, dim3 size, int numflats, GPU gpus);

	void getLog(float *tomogram, dim3 size, GPU gpus);

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
    int getRotAxisOfsset(float *start_frame, float *end_frame, float *flat, float *dark, dim3 size, int rotation_axis_method);

    int getCentersino(float* frame0, float* frame1, float* dark, float* flat, int sizex, int sizey);

    int Centersino16(uint16_t* frame0, uint16_t* frame1, uint16_t* dark, uint16_t* flat, size_t sizex, size_t sizey);

    int Centersino(float* frame0, float* frame1, float* dark, float* flat, size_t sizex, size_t sizey);

    int findcentersino16(uint16_t* frame0, uint16_t* frame1, uint16_t* dark, uint16_t* flat, int sizex, int sizey);

    int findcentersino(float* frame0, float* frame1, float* dark, float* flat, int sizex, int sizey);

    __global__ void KCrossCorrelation(complex* F, const complex* G, size_t sizex);

    __global__ void KCrossFrame16(complex* f, complex* g, const uint16_t* frame0, const uint16_t* frame1, const uint16_t* dark, const uint16_t* flat, size_t sizex);

    __global__ void KCrossFrame(complex* f, complex* g, const float* frame0, const float* frame1, const float* dark, const float* flat, size_t sizex);

}

/* 360 offset - Find offset for 360 degrees parallel tomogram in a "panoramic" acquisition */
extern "C"{
	int ComputeTomo360Offsetgpu(int gpu, float* cpusinograms, int sizex, int sizey, int sizez);
	
	void Tomo360To180gpu(int gpu, float* data, int nrays, int nangles, int nslices, int offset);

    void Tomo360To180block(int* gpus, int ngpus, float* data, int nrays, int nangles, int nslices, int offset);

	int ComputeTomo360Offset16(int gpu, uint16_t* frames, uint16_t* cflat, uint16_t* cdark, int sizex, int sizey, int sizez, int numflats);

}


#endif