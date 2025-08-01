#ifndef RAFT_CONFIGS_H
#define RAFT_CONFIGS_H

#include <complex.h>
#include <driver_types.h>
#include <vector_types.h>
#include <cstddef>

#define FFT_RANK_2D 2
#define FFT_RANK_1D 1

#define IND(I,J,K,NX,NY) (long long int)( (I) + (J) * (NX) + (K) * (NX) * (NY) )

#define PADS(X, P) (int)( ( X * P) / 100 )

#define PDIM(X,P) (int)( X + 2 * ((int)( (X * P) / 100 )) )

#define vc 299792458           /* Velocity of Light [m/s] */ 
#define plank 4.135667696E-15  /* Plank constant [ev*s] */
#define PI 3.141592653589793238462643383279502884

#define BOLTZMANN_CONSTANT 1.3806488e-16  /* [erg/k] */ 
#define SPEED_OF_LIGHT 299792458e+2       /* [cm/s]  */ 
#define PLANCK_CONSTANT 6.58211928e-19    /* [keV*s] */

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

enum ReconstructionMethod
{
    none    = 0,
    fbpRT   = 1,
    fbpBST  = 2,
    eEMRT   = 3,
    tEMRT   = 4,
    tEMFQ   = 5,
    fdk     = 6
};

typedef struct coordinates
{
    float x; /* Coordinates values on x-direction */
    float y; /* Coordinates values on y-direction */
} coord; /* Coordinates values */

typedef struct dimension
{
    dim3 size; /* Dimension values */
    dim3  pad;     /* Pad value */
    int blocksize;
    int padding_mode;
} DIM; /* Data dimensions */

inline float calcSliceMemoryBytes(DIM dimension) {
   return dimension.size.x * dimension.size.y * sizeof(float);
}

inline float calcPaddedSliceMemoryBytes(DIM dimension) {
    return ((dimension.size.x * ( 1 + dimension.pad.x )) + (dimension.size.y * ( 1 + dimension.pad.y) ))* sizeof(float);
}

inline float calcWidthMemoryBytes(DIM dimension) {
    return dimension.size.y * sizeof(float);
}

inline float calcLengthMemoryBytes(DIM dimension) {
    return dimension.size.x * sizeof(float);
}

typedef struct geometry
{
    /* General reconstruction variables*/
    coord detector_pixel;
    coord obj_pixel;
    coord z1, z2;
    coord magnitude;
    float energy, wavelength;
}GEO;

typedef struct flags
{
    /* Bool variables - Pipeline */
    int do_flat_dark_correction, do_flat_dark_log;
    int do_paganin_filter;
    int do_rings;
    int do_rotation, do_rotation_axis_offset, do_rotation_correction;
    int do_alignment;
    int do_reconstruction;
    int do_eccentric;

}FLAG;

typedef struct ContrastEnhancementFilter
{
    /* Paganin Filter */
    int method; /* Contrast Enhancement methods. Options: paganin, paganin_slices*/
    float beta_delta; /* beta/delta parameter */
    float regularization; /* regularization parameter */
    int post_process; /* Function applied after usual convolution: ex. aplly -log() after Paganin kernel */

}CEF;

typedef struct RingsFilter
{
    /* Paganin Filter */
    int method; /* Rings methods. Options: titarenko*/
    int rings_block;    /* Titarenko's parameter */
    float rings_lambda; /* Titarenko's regularization parameter */

}RF;

typedef struct Reconstruction
{
    /* Paganin Filter */
    int method;                  /* Reconstruction methods. Options: FBP*/
    int filter;                  /* Filter. Options: ramp, hamming, hann, ... */
    float filter_reg;            /* General regularization parameter for filter */
    float paganin_slices = 0.0f; /* Paganin regularization parameter for slices method */
    int iterations;              /* Iterations for iterative methods */
    float rotation_axis_offset;  /* Rotation axis offset */
    float total_variation;       /* Total variation regularization parameter */
    int interpolation;           /* Interpolation Type. Options: 'bilinear' and 'nearest' */
}REC;

typedef struct config
{
    int nflats;
    
    GEO geometry;

    /* Reconstruction variables */
    DIM obj;

    /* Tomogram variables */
    DIM tomo; 

    FLAG flags;

    CEF ContrastParam;

    RF RingsParam;

    REC ReconParam;

} CFG;

inline void printDim(dim3 d) {
    printf("dim3 {%d, %d, %d}\n", d.x, d.y, d.z);
}

inline float calcTotalRequiredMemoryBytes(DIM tomo, DIM obj) {
    return (
            calcSliceMemoryBytes(tomo) +
            calcSliceMemoryBytes(obj) +
            calcPaddedSliceMemoryBytes(tomo) +
            calcLengthMemoryBytes(tomo) +
            calcWidthMemoryBytes(tomo)
           );
}

inline bool isParallelOrFanbeamGeometry(GEO geometry) {
    return geometry.magnitude.y == 1;
}

inline bool isConeGeometry(GEO geometry) {
    return geometry.magnitude.y != 1;
}

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
	float *tomo, *obj, *tomoPadd, *objPadd;
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

    long long int n_recon, n_recon_pad, n_tomo, n_filter; 
    int i, i_gpu, zi, z_filter, z_filter_pad, z_proj, z_recon;
    long long int n_proj, n_proj_pad, n_filter_pad;
    long long int idx_proj, idx_proj_pad, idx_proj_max, idx_recon, idx_recon_pad, idx_filter, idx_filter_pad;
    float z_ph, z_det;
};


inline size_t getTotalDeviceMemory(int device = 0) {
    size_t total_mem, free_mem;

    cudaSetDevice(device);
    cudaMemGetInfo(&free_mem, &total_mem);

    return total_mem;
}

/* Processes - parallelization */

extern "C" {

    Process *setProcesses(CFG configs, GPU gpus, int total_number_of_processes);

    void setProcessParallel(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);

    void setProcessConebeam(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);

    int getTotalProcesses(CFG configs, float GPU_MEMORY, int sizeZ, bool using_fft);

    int compute_GPU_blocksize(int nslices, float total_required_mem_per_slice,
            bool using_fft, float GPU_MEMORY);

}

/* Workspace - GPU pointers */
extern "C"{

	WKP *Initialize_workspace(CFG configs, size_t tomo_batch_size, size_t obj_batch_size);

	void freeWorkspace(WKP *workspace);

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
