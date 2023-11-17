#ifndef RAFT_FD_H
#define RAFT_FD_H

#include <string.h>
#include "../common/structs.h"


extern "C"{

        void flatdark_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int is_log, int totalslices);

        void flatdark_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int is_log);

        __global__ void RemoveMeanKernel(float *in, float *mean, int sizex, int sizey, int sizez);

        __global__ void ComputeMeanKernel(float *in, float *mean, int sizex, int sizey, int sizez);

	void remove_mean_flat_dark(float *flat, float *dark, int sizex, int sizez, int numflats);

        void getFlatDarkCorrection(float* frames, float* flat, float* dark, int sizex, int sizey, int sizez, int numflats, int islog, dim3 BT, dim3 Grd);

}

#endif