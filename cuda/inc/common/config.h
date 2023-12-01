#ifndef RAFT_CONFIGS_H
#define RAFT_CONFIGS_H

#define IND(I,J,K,NX,NY) ( (I) + (J) * (NX) + (K) * (NX) * (NY) )

# define vc 299792458           /* Velocity of Light [m/s] */ 
# define plank 4.135667662E-15  /* Plank constant [ev*s] */


typedef struct dimension
{   
    dim3 size;
    float  x,  y,  z;
    float dx, dy, dz;
    dim3  pad;
    dim3  npad;
    int start_slice, end_slice; /* Slices: start and end slice */
    int nrays, nangles, nslices;

}DIM;

typedef struct geometry
{   
    /* Geometry */
    int geometry;

    /* General reconstruction variables*/
    float detector_pixel_x, detector_pixel_y;
    float energy, lambda, wave;
    float z1x, z1y, z2x, z2y;
    float magnitude_x, magnitude_y;  
    
}GEO;

typedef struct flags
{   
    /* Bool variables - Pipeline */
    int do_flat_dark_correction, do_flat_dark_log;
    int do_phase_filter;
    int do_rings, do_rotation_offset, do_alignment;
    int do_reconstruction;
    
}FLAG;

typedef struct config
{   
    /* Pipeline variables */

    GEO geometry;

    FLAG flags;

    /* Reconstruction variables */
    DIM recon;
    
    /* Tomogram variables */
    DIM tomo; 

    /* Flat/Dark Correction */
    int numflats;

    /* Phase Filter */
    int phase_filter_type;             /* Phase Filter type */
    float phase_filter_reg; /* Phase Filter regularization parameter */

    /* Rings */
    int rings_block;
    float rings_lambda;

    /* Rotation Axis Correction */
    int rotation_axis_offset;

    /* Reconstruction method variables */
    int reconstruction_method;
    int reconstruction_filter_type;   /* Reconstruction Filter type */
    float reconstruction_paganin_reg; /* Reconstruction Paganin Filter regularization parameter */
    float reconstruction_reg;         /* General regularization parameter */

    int datatype;
    float threshold;          

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
    cufftHandle mplan1dC2C, mplan1dR2C, mplan1dC2R;
    cudaStream_t *streams;
    int num_streams;  
    dim3 BT, Grd;  
    
}GPU;

extern "C" {

	inline void getDeviceProperties()
	{	/* Get Device Properties */
		int gpudevices; 
		cudaDeviceProp prop; 
		cudaGetDeviceCount(&gpudevices); /* Total Number of GPUs */ 
		printf("GPUs number: %d\n",gpudevices); 
		cudaGetDeviceProperties(&prop,0); /* Name of GPU */ 
		printf("Device name: %s\n",prop.name);	
	};

}

extern "C"{

    void setReconstructionParameters(CFG *configs, float *parameters_float, int *parameters_int, int *flags);

	void setGPUParameters(GPU *gpus_parameters, dim3 size_pad, int ngpus, int *gpus);

	void setPhaseFilterParameters(GEO *geometry, DIM *tomo, float *parameters_float, int *parameters_int);

}

#endif 
