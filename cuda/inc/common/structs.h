#ifndef COMMON_H
#define COMMON_H

/**
@file structs.h
@author Paola Ferraz (paola.ferraz@lnls.br)
@brief Header file for scr files.
@version 0.1
@date 2021-06-12

@copyright Copyright (c) 2021

 */

#include <stdio.h>
#include <string.h>

#define MEGA 1048576UL
#define GIGA 1073741824UL
#define SQUARE(x) ((x)*(x))


/**
@typedef PAR
@brief parameters for image.

@var PAR::Nx
Nx is the number of voxels in x direction.
@var PAR::Ny
Ny is the number of voxels in y direction.
@var PAR::Nz
Nx is the number of voxels in z direction.
@var PAR::zblock
Block size in cuda z direction.
@var PAR::slice
slice size, i.e., Nx*Ny.
@var PAR::devices
Array containing the range {0, 1, 2, …, ndev - 1}. 
@var PAR::mplan
Cuda FFT plan.
@var PAR::mplan2
Cuda FFT plan.
 */
typedef struct Parameters
{	/* General Parameters: Dimensions and Cuda*/
        size_t nangles, nrays, nslices; /* Reconstruction Problem Dimensions */
        size_t Nx, Ny, Nz; /* Thread dimensions */
	    size_t sizeofblock, blocksize, subvolume; 
        size_t slice, slicePadded, frame, framePadded; /* Thread dimensions */

	    int ngpus, *gpus;
	    cufftHandle mplan, mplan2;
	    cudaStream_t *streams;
        int num_streams;  
	    dim3 BT, Grd;  

	/* General parameters */
        float pixelDetx, pixelDety; 
        float z1x, z1y, z2x, z2y;  /* Parameter: distances */
        float magnx, magny;  /* Parameter: magnitude for z's */
        float effx_pixel, effy_pixel; /* Parameter: effective pixels */

    /* General Rebinning parameters */
        float ct, rt, st;
        float cr, rr, sr;
        float Lt, Lr;
        float Dt, Dr;
        float d1x, d1y, d2x, d2y; /* Rebinning parameter: distances */
        float mx, my; /* Rebinning parameter: magnitude for d's */
        float effa_pixel, effb_pixel; /* Rebinning parameter: effective pixels */

    /* General GC (Conical) parameters */

    
}PAR;

typedef struct GCData
{	/* GPU */
	float *volume; // projection volume (*volume)
	float *volumePadded; 
}GC;


typedef struct Profiling
{

}PROF;

#endif