#ifndef RAFT_TOMOF360_H
#define RAFT_TOMOF360_H

#include "common/complex.hpp"
#include "common/logerror.hpp"

/* Centersino - Find offset for 180 degrees parallel tomogram */

extern "C"{
    int Centersino(uint16_t* frame0, uint16_t* frame1, uint16_t* dark, uint16_t* flat, size_t sizex, size_t sizey);

    int find_centersino16(uint16_t* frame0, uint16_t* frame1, uint16_t* dark, uint16_t* flat, int sizex, int sizey);

    __global__ void KCrossCorrelation(complex* F, const complex* G, size_t sizex);

    __global__ void KCrossFrame(complex* f, complex* g, const uint16_t* frame0, const uint16_t* frame1, const uint16_t* dark, const uint16_t* flat, size_t sizex);
}

/* 360 offset - Find offset for 360 degrees parallel tomogram in a "panoramic" acquisition */
extern "C"{
	int ComputeTomo360Offset(int gpu, float* cpusinograms, int sizex, int sizey, int sizez);
	
	void Tomo360To180(int gpu, float* cpusinograms, int sizex, int sizey, int sizez, int offset);
}


#endif