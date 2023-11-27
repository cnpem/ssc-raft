#ifndef RAFT_FD_H
#define RAFT_FD_H

#include <string.h>
#include "../common/structs.h"


extern "C"{

	void getFlatDarkMultiGPU(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nangles, int nslices, int numflats, int is_log);

        void getFlatDarkGPU(GPU gpus, int gpu, float* frames, float* flat, float* dark, dim3 size, int numflats, int is_log);

	void getFlatDarkCorrection(float* frames, float* flat, float* dark, dim3 size, int numflats, GPU gpus);

	void getLog(float *tomogram, dim3 size, int numflats, GPU gpus);

}

#endif