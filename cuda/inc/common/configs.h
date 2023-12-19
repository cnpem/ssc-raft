#ifndef RAFT_CONFIGS_H
#define RAFT_CONFIGS_H

#define IND(I,J,K,NX,NY) ( (I) + (J) * (NX) + (K) * (NX) * (NY) )

#define vc 299792458           /* Velocity of Light [m/s] */ 
#define plank 4.135667662E-15  /* Plank constant [ev*s] */
#define PI 3.141592653589793238462643383279502884

#define TPBX 16
#define TPBY 16
#define TPBZ 4
#define TPBE 256

#define MEGA 1048576UL
#define GIGA 1073741824UL

#define SQR(x) ((x)*(x))
#define SIGN(x) ((x > 0) ? 1 : ((x < 0) ? -1 : 0))
#define APPROXINVX(x,e) ((SIGN(x))/(sqrtf( SQR(e) + SQR(x) )))

#include "include.h"

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
    int do_rings;
    int do_rotation, do_rotation_auto_offset, do_rotation_correction;
    int do_alignment;
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
    int numflats, numdarks;

    /* Phase Filter */
    int phase_filter_type;  /* Phase Filter type */
    float phase_filter_reg; /* Phase Filter regularization parameter */

    /* Rings */
    int rings_block;
    float rings_lambda, rings_lambda_computed;

    /* Rotation Axis Correction */
    int rotation_axis_offset, rotation_axis_method, rotation_axis_offset_computed;

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

    /* EM */
    int em_iterations;

    /* Conical */

    /* FDK */

    /* EM Conical */
    
}CFG;


typedef struct Devices
{   
    /* GPU variables */
    int ngpus, *gpus;
    cudaStream_t *streams;
    int num_streams;  
    dim3 BT, Grd;  

    /* Fourier Transforms */
    /* Plan FFTs*/
    cufftHandle mplan;
    cufftHandle mplan2dC2C, mplan2dR2C, mplan2dC2R, mplan2dR2R;
    cufftHandle mplan1dC2C, mplan1dR2C, mplan1dC2R, mplan1dR2R;

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

typedef struct workspace
{	/* GPU */
	float *tomo, *recon;
	float *flat, *dark, *angles; 
}WKP;

typedef struct Processes{ 
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

    int i, i_gpu, zi, z_filter, z_filter_pad;
    long long int n_proj, n_filter_pad;
    long long int idx_proj, idx_proj_max, idx_recon, idx_filter, idx_filter_pad;
    float z_ph;

}Process;


/* Commom parameters */
extern "C"{

    void setReconstructionParameters(CFG *configs, float *parameters_float, int *parameters_int, int *flags);

	void setGPUParameters(GPU *gpus_parameters, dim3 size_pad, int ngpus, int *gpus);

	void setPhaseFilterParameters(GEO *geometry, DIM *tomo, float *parameters_float, int *parameters_int);

    void setFBPParameters(CFG *configs, float *parameters_float, int *parameters_int);

	void setEMParameters(CFG *configs, float *parameters_float, int *parameters_int);

}

/* Processes - parallelization */
extern "C"{

	Process *setProcesses(CFG configs, GPU gpus, int total_number_of_processes);

    void setProcessParallel(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);

    void setProcessConebeam(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);

    void setProcessFrames(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);
    
    int getTotalProcesses(CFG configs, GPU gpus);

}

/* Workspace - GPU pointers */
extern "C"{

	WKP *allocateWorkspace(CFG configs, Process process);

	void freeWorkspace(WKP *workspace, CFG configs);

}

// FFT SHIFT -> Crop/Place -> FFT ISHIFT
// Warning: Fills empty space with non-zeros
// -> Blame EM
template<typename Type>
__global__ void KCopyShiftX(Type* out, Type* in, size_t outsizex, size_t insizex, size_t nangles, int csino, float filler)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t idy = blockIdx.y;
	
	size_t index = idy * insizex + (idx+insizex-csino)%insizex;
	size_t shift = idy * outsizex + (idx + outsizex - insizex/2) % outsizex;

	if(idx < insizex)
		out[shift] = in[index];
	else if(idx < outsizex)
		out[shift] = filler;
}

template<typename Type>
__global__ void KSetToOne(Type* vec, size_t size)
{
	size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < size)
		vec[idx] = Type(1);
}


#endif 
