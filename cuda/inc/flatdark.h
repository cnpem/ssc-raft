#ifndef RAFT_FD_H
#define RAFT_FD_H

#include <string.h>
#include "common/structs.h"


extern "C"{
        
        void flatdarkcpu(float* out, float* frames, float* cflat, float* cdark, 
		size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats, int tidx, int nthreads);

        void flatdarkcpu_block(float* out, float* frames, float* cflat, float* cdark, 
	        size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats);

        void flatdarktranspose_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats);

        void flatdarktranspose_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats);

        void flatdarkcpu_log(float* out, float* frames, float* cflat, float* cdark, 
		size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats, int tidx, int nthreads);

        void flatdarkcpu_log_block(float* out, float* frames, float* cflat, float* cdark, 
	        size_t sizex, size_t sizey, size_t sizez, size_t block, int numflats);

        void flatdarktranspose_log_gpu(int gpu, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats);

        void flatdarktranspose_log_block(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nslices, int nangles, int numflats);

}

#endif