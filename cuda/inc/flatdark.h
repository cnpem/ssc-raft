#ifndef RAFT_FD_H
#define RAFT_FD_H

#include <string.h>
#include "common/structs.h"


extern "C"{

        void flatdark_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int Totalframes, int Initframe, int is_log);

        void flatdark_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats, int Totalframes, int Initframe, int is_log);
}

#endif