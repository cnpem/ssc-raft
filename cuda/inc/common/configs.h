#ifndef RAFT_CONFIGS_H
#define RAFT_CONFIGS_H

#define IND(I,J,K,NX,NY) ( (I) + (J) * (NX) + (K) * (NX) * (NY) )

# define vc 299792458           /* Velocity of Light [m/s] */ 
# define plank 4.135667662E-15  /* Plank constant [ev*s] */


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

typedef struct Devices
{   
    /* GPU variables */
    int ngpus, *gpus;
    cufftHandle mplan;
    cudaStream_t *streams;
    int num_streams;  
    dim3 BT, Grd;  
    
}GPU;

extern "C"{

    void setReconstructionParameters(CFG *configs, float *parameters_float, int *parameters_int, int *flags);

	void setGPUParameters(GPU *gpus_parameters, dim3 size_pad, int ngpus, int *gpus);


}

#endif 
