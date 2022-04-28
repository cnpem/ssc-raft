#ifndef COMMON_H
#define COMMON_H

#include "../../inc/include.h"

#define MEGA 1048576UL
#define GIGA 1073741824UL
#define SQUARE(x) ((x)*(x))

typedef struct Parameters
{	/* General Parameters: Dimensions and Cuda*/
        size_t nangles, nrays; /* Reconstruction Problem Dimensions */
	    size_t Ninx, Niny, Ninz; /* Input dimensions */
        size_t Noutx, Nouty, Noutz; /* Output dimensions */
        size_t Nx, Ny, Nz, Nt, nx, ny, nz, nt; /* Thread dimensions */
	    size_t sizeofblock, blocksize, slice, subvolume; /* Thread dimensions */

	    int ngpus, *gpus;
	    cufftHandle mplan, mplan2;
	    cudaStream_t *streams;
        int num_streams;  
	    dim3 BT, Grd;  

	/* General parameters */
	    float Fx, Fy, ratio, z1, z2, zt;

    /* General Rebinning parameters */
        float efft_pixel, effr_pixel;
        float ct, rt, st;
        float cr, rr, sr;
        float Lt, Lr;
        float Dt, Dr;
    
}PAR;

typedef struct FANData
{
    /* General FAN parameters */
        float SD, effx_pixel, effy_pixel;
        float gammaM;
    /* CPU */
        float *data, *recon, *flat, *dark;
    
    /* GPU */
        cufftComplex *aux_data, *kernel;
        float *ddata, *sintable, *costable, *weight, *drecon, *dflat, *ddark; 
}FDAT;

typedef struct CONEData
{
    /* General CONE parameters */
        float SD, effx_pixel, effy_pixel;
        float gammaM;
    /* CPU */
        float *data, *recon, *flat, *dark;
    
    /* GPU */
        cufftComplex *aux_data, *kernel;
        float *ddata, *sintable, *costable, *weight, *drecon, *dflat, *ddark; 
}CDAT;

typedef struct REBData
{
    /* CPU */
        float *conetomo, *tomo;
    
    /* GPU */
        float *dctomo, *dtomo; 
}RDAT;

typedef struct Profiling
{

}PROF;

#endif