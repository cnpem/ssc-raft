#ifndef RAFT_FD_H
#define RAFT_FD_H

#include <string.h>
#include "../common/structs.h"


extern "C"{

        void flatdark_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int Totalframes, int Initframe, int is_log, int totalslices);

        void flatdark_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int Totalframes, int Initframe, int is_log);

        __global__ void remove_meanKernel(float *in, float *mean, int sizex, int sizey, int sizez);

        __global__ void meanKernel(float *in, float *mean, int sizex, int sizey, int sizez);

	void remove_mean_flat_dark(float *flat, float *dark, int nrays, int nslices, int numflats);

}

#endif