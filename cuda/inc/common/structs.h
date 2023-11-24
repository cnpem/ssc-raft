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

	/* General parameters */
    float alpha;
    size_t Npadx, Npady;

    int filter;
    
    /* General Rebinning parameters */
    float ct, rt, st;
    float cr, rr, sr;
    float Lt, Lr;
    float Dt, Dr;
    size_t Nx, Ny, Nz, Nt, nt; /* Thread dimensions */
    size_t sizeofblock, blocksize, slice, subvolume; /* Thread dimensions */
    float d1x, d1y, d2x, d2y;
    float mx, my;
    float effa_pixel, effb_pixel;

    /* Recon variables */
    int nx, ny, nz;
    float x, y, z;
    float dx, dy, dz;
    
    /* Tomogram (or detector) variables */
    int nrays, nslices, nangles;
    float Lrays, Lslices, Langles;
    float drays, dslices, dangles;

    /* Padding */
    int padx, pady, nprays, npslices;

    /* General reconstruction variables*/
    float pixelDetx, pixelDety;
    float energy, lambda, wave;

    /* Conical general variables*/
    float z1, z2, z12;
    float z1x, z1y, z2x, z2y;
    float magnx, magny;

    /* GPU variables */
    int ngpus, *gpus;
    cufftHandle mplan;
    cudaStream_t *streams;
    int num_streams;  
    dim3 BT, Grd;  

}PAR;

typedef struct config
{   
    /* Geometry */
    int geometry;

    /* Reconstruction variables */
    int   nx, ny, nz;
    float  x,  y,  z;
    float dx, dy, dz;
    
    /* Tomogram (or detector) variables (h for horizontal (nrays) and v for vertical (nslices)) */
    int   nrays, nslices, nangles;
    float h, v;
    float dh, dv, dangles;

    /* Padding */
    int padx, pady, padz, pad_nrays, pad_nangles, pad_nslices;

    /* General reconstruction variables*/
    float pixelDetx, pixelDety;
    float detector_pixel_x, detector_pixel_y;
    float energy, lambda, wave;
    float z1, z2, z12;
    float z1x, z1y, z2x, z2y;
    float magnx, magny;
    float magnitude_x, magnitude_y;

    /* GPU variables */
    int ngpus, *gpus;
    cufftHandle mplan;
    cudaStream_t *streams;
    int num_streams;  
    dim3 BT, Grd;  

    /* General variables */

    /* Bool variables - Pipeline */
    int do_flat_dark_correction, do_phase_filter;
    int do_rings, do_rotation_offset,do_alignment;
    int do_reconstruction;

    /* Flat/Dark Correction */
    int flat_dark_do_log;
    int numflats;

    /* Phase Filter */
    int phase_filter_type; /* Phase Filter type */
    float phase_filter_regularization; /* Phase Filter regularization parameter */

    /* Rings */
    int rings_block;
    float rings_lambda;
    float rings_computed_lambda;

    /* Rotation Axis Correction */
    int rotation_axis_offset;

    /* Reconstruction method variables */
    int reconstruction_method;
    int reconstruction_filter_type; /* Reconstruction Filter type */
    float reconstruction_filter_regularization; /* Reconstruction Filter regularization parameter */
    float reconstruction_regularization; /* General regularization parameter */

    /* Slices */
    // int slice_recon_start, slice_recon_end; // Slices: start slice = slice_recon_start, end slice = slice_recon_end
    // int slice_tomo_start, slice_tomo_end; // Slices: start slice = slice_tomo_start, end slice = slice_tomo_end
    int reconstruction_start_slice, reconstruction_end_slice; // Slices: start slice = reconstruction_start_slice, end slice = reconstruction_end_slice
    int tomogram_start_slice, tomogram_end_slice; // Slices: start slice = tomogram_start_slice, end slice = tomogram_end_slice

    /* Paralell */

    /* FBP */

    /* BST */

    /* EM RT */
    int em_iterations;

    /* EM FST */

    /* Conical */

    /* FDK */

    /* EM Conical */
    
}CFG;

typedef struct workspace
{	/* GPU */
	float *tomo, *flat, dark, *recon, *angles; 
	float *tomoPadded; 
}WKP;

typedef struct { 
    /* GPU */ 
    int index, index_gpu;

    /* Processes*/
    int batch_size_tomo, batch_size_recon, batch_index;

    /* Tomogram (or detector) and reconstruction filter (v of vertical) */
    int indv, indv_filter;
    long long int ind_tomo, n_tomo, ind_filter, n_filter;

    /* Reconstruction */
    long long int ind_recon, n_recon;
    float z, z_det;

} Process;


typedef struct Profiling
{

}PROF;

// typedef struct {  
//     float x,y,z;
//     float dx, dy, dz;
//     int nx, ny, nz;
//     float h,v;
//     float dh, dv;
//     int nh, nv;
//     float D, Dsd;
//     float beta_max;
//     float dbeta;
//     int nbeta;
//     int fourier;
//     int filter_type; // Filter Types
//     float reg; // Filter regularization
//     int is_slice; // (bool) Reconstruct a block of slices or not
//     int slice_recon_start, slice_recon_end; // Slices: start slice = slice_recon_start, end slice = slice_recon_end
//     int slice_tomo_start, slice_tomo_end; // Slices: start slice = slice_tomo_start, end slice = slice_tomo_end
//     int nph, padh;
//     float energy;

//     /* Filter Types definitions
//     enum EType
// 	{
//         none      = 0,
//         gaussian  = 1,
//         lorentz   = 2,
//         cosine    = 3,
//         rectangle = 4,
//         hann      = 5,
//         hamming   = 6,
//         ramp      = 7
// 	};
//     */

// } Lab;




#endif