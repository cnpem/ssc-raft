#ifndef RAFT_PGN_H
#define RAFT_PGN_H

#include <string.h>
#include "../common/structs.h"


extern "C"{

	void phase_filters(float *projections, float *paramf, size_t *parami, 
				int nrays, int nangles, int nslices,
				int *gpus, int ngpus);

	void _phase_filters_threads(PAR param, float *projections, size_t nrays, size_t nangles, size_t nslices, int ngpu);
	void set_phase_filters_parameters(PAR *param, float *paramf, size_t *parami, size_t sizex, size_t sizey, size_t sizez);
    void apply_filter(PAR param, float *data, float *kernel, float *ans, cufftComplex *dataPadded, size_t sizex, size_t sizey, size_t sizez);

    float find_matrix_max(float *matrix, size_t sizex, size_t sizey);
	void print_matrix(float *matrix, size_t sizex, size_t sizey);

 	__global__ void paganinKernel(float *kernel, PAR param, size_t sizex, size_t sizey, size_t sizez);
    __global__ void paganinReturn(float *data, PAR param, size_t sizex, size_t sizey, size_t sizez);
 	__global__ void bornKernel(float *kernel, PAR param, size_t sizex, size_t sizey, size_t sizez);
    __global__ void bornPrep(float *data, PAR param, size_t sizex, size_t sizey, size_t sizez);
 	__global__ void rytovKernel(float *kernel, PAR param, size_t sizex, size_t sizey, size_t sizez);
    __global__ void rytovPrep(float *data, PAR param, size_t sizex, size_t sizey, size_t sizez);
 	__global__ void bronnikovKernel(float *kernel, PAR param, size_t sizex, size_t sizey, size_t sizez);
    __global__ void bronnikovPrep(float *data, PAR param, size_t sizex, size_t sizey, size_t sizez);
    
    __global__ void CConvolve(cufftComplex *a, float *b, cufftComplex *ans, size_t sizex, size_t sizey, size_t sizez);
    __global__ void fftNormalize(cufftComplex *c, size_t sizex, size_t sizey, size_t sizez);
    __global__ void fftshiftKernel(float *c, size_t sizex, size_t sizey, size_t sizez);
	__global__ void Normalize(float *a, float b, size_t sizex, size_t sizey);

	void _paganin_gpu(PAR param, float *projections, float *d_kernel, size_t nrays, size_t nangles, size_t nslices);
	void _born_gpu(PAR param, float *projections, float *d_kernel, size_t nrays, size_t nangles, size_t nslices);
	void _rytov_gpu(PAR param, float *projections, float *d_kernel, size_t nrays, size_t nangles, size_t nslices);
	void _bronnikov_gpu(PAR param, float *projections, float *d_kernel, size_t nrays, size_t nangles, size_t nslices);

	void _paganin_gpu2(PAR param, float *out, float *projections, float *d_kernel, size_t nrays, size_t nangles, size_t nslices);

	void phase_filters2(float *out, float *projections, float *paramf, size_t *parami, 
				int nrays, int nangles, int nslices,
				int *gpus, int ngpus);
	
	void _phase_filters_threads2(PAR param, float *out, float *projections, size_t nrays, size_t nangles, size_t nslices, int ngpu);
	__global__ void KCopy(cufftComplex *in, float *out, size_t sizex, size_t sizey, size_t sizez);

}

#endif