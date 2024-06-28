#ifndef RAFT_CONFIGS_H
#define RAFT_CONFIGS_H

#define PARALLEL 0
#define CONEBEAM 1
#define FANBEAM  2

#define FFT_RANK_2D 2 
#define FFT_RANK_1D 1 

#define IND(I,J,K,NX,NY) (long long int)( (I) + (J) * (NX) + (K) * (NX) * (NY) )

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

#define BYTES_TO_GB (1.0/(1024.0*1024.0*1024.0))
#define A100_MEM 39.5 // A100 40GB device RAM memory, in GB.

#include "cufft.h"
#include <stdio.h>
#include <cuda_runtime_api.h>

// following includes not directly used here,
// should we move to specific files?
#include <iostream>
#include <future>

typedef struct dimension
{   
    dim3 size; /* Dimension values */
    float  posx,  posy,  posz; /* points values */
    float    dx,    dy,    dz; /* spacing values */
    float    Lx,    Ly,    Lz; /* Dimension length valuess */

    size_t xyz = size.x * size.y * size.x; /* Total number of points */
    size_t xy  = size.x * size.y; /* Total number of points on xy-plane */

    int xslice0, xslice1; /* Slices X: start (0)  and end slice (1)*/
    int yslice0, yslice1; /* Slices Y: start (0)  and end slice (1)*/
    int zslice0, zslice1; /* Slices Z: start (0)  and end slice (1)*/

    dim3  padsize; /* Padded dimensions: (size + 2 * pad) */
    dim3  pad;     /* Pad value */

    dim3 batchsize;
    dim3 padbatchsize;

    float width_memory_bytes;
    float lenght_memory_bytes;
    float slice_memory_bytes;
    float slice_padd_memory_bytes;
    float frame_memory_bytes;
    float frame_padd_memory_bytes;

}DIM; /* Data dimensions */

typedef struct geometry
{   
    /* Geometry: 
        0 -> parallel (PARALLEL)
        1 -> conebeam (CONEBEAM)
        2 -> fanbeam  (FANBEAM)
    */
    int geometry;

    /* General reconstruction variables*/
    float detector_pixel_x, detector_pixel_y;
    float obj_pixel_x, obj_pixel_y;
    float energy, wavelenght, wavenumber;
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
    float total_required_mem_per_slice_bytes;
    float total_required_mem_per_frame_bytes;
    int blocksize;

    GEO geometry;

    FLAG flags;

    /* Reconstruction variables */
    DIM obj;
    
    /* Tomogram variables */
    DIM tomo; 

    /* Flat/Dark Correction */
    int numflats, numdarks;

    /* Phase Retrieval */
    int  phase_type;  /* Phase type */
    float phase_reg; /* Phase regularization parameter */
    float beta_delta; /* beta/delta parameter */

    /* Rings */
    int rings_block;
    float rings_lambda, rings_lambda_computed;

    /* Rotation Axis Correction */
    int rotation_axis_offset, rotation_axis_method, rotation_axis_offset_computed;

    /* Reconstruction method variables */
    int reconstruction_method;
    int reconstruction_filter_type;   /* Reconstruction Filter type */
    float reconstruction_paganin;     /* Reconstruction Paganin regularization parameter */
    float reconstruction_reg;         /* General regularization parameter */
    float reconstruction_tv;          /* Total variation regularization parameter */

    int datatype;
    float threshold; 
    int interpolation; /* interpolation type */         

    /* Paralell */

    /* FBP */

    /* BST */

    /* EM Parallel */
    int em_iterations;

    /* Conical */

    /* FDK */

    /* EM Conical */

}CFG;


struct GPU
{
    /* GPU variables */
    int ngpus, *gpus;
    cudaStream_t *streams;
    int num_streams;
    dim3 BT, Grd;

    /* Fourier Transforms */
    /* Plan FFTs*/
    cufftHandle mplan;
    cufftHandle mplanI;
};


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
	float *tomo, *obj;
	float *flat, *dark, *angles; 
}WKP;

struct Process{
    size_t total_req_size_mem;
     
    /* Process variables to parallelize the data z-axis by independent blocks */

    /* GPU */ 
    int index, index_gpu;

    /* Processes*/
    int tomobatch_size, objbatch_size, batch_index;

    /* Tomogram (or detector) and reconstruction filter */
    int tomo_index_z, filter_index_z;
    long long int tomoptr_index, tomoptr_size, filterptr_index, filterptr_size;
    float tomo_posz;

    /* Object - Reconstruction */
    long long int objptr_index, objptr_size;
    float obj_posz;

    long long int n_recon, n_tomo, n_filter; 
    int i, i_gpu, zi, z_filter, z_filter_pad;
    long long int n_proj, n_filter_pad;
    long long int idx_proj, idx_proj_max, idx_recon, idx_filter, idx_filter_pad;
    float z_ph, z_det;
};


/* Commom parameters */
extern "C"{

    void setReconstructionParameters(CFG *configs, float *parameters_float, int *parameters_int, int *flags);
    void printGPUParameters(GPU *gpus_parameters);
	void setGPUParameters(GPU *gpus_parameters, dim3 size, int ngpus, int *gpus);

}

/* Processes - parallelization */
extern "C"{

	Process *setProcesses(CFG configs, GPU gpus, int total_number_of_processes);

    void setProcessParallel(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);

    void setProcessConebeam(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);
    
    int getTotalProcesses(CFG configs, float GPU_MEMORY, int sizeZ, bool using_fft);

    int compute_GPU_blocksize(int nslices, float total_required_mem_per_slice,
    bool using_fft, float GPU_MEMORY); 

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
