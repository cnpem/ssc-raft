#include <stddef.h>
#include <stdio.h>  // fprintf (function).
#include <stdlib.h> // stderr (poiter to error file).
#include <string.h> // strstr.
#include "emc_manager_data_types.cuh"
#include "emc_cone_data_types.cuh"


#define DEBUG 0
#define DEBUG_CUDA_CHECK()                      \
do {                                            \
    if(DEBUG) {                                 \
        CHECK_CUDA( cudaDeviceSynchronize() );  \
    }                                           \
} while(0)                                      \

#define CHECK_CUDA(call)                                                                       \
do {                                                                                           \
    bool is_cudaGetLastError = (strstr(#call, "cudaGetLastError()") != NULL);                  \
    cudaError_t cudaStatus = call;                                                             \
    check_cuda(cudaStatus, is_cudaGetLastError, __LINE__, __FILE__);                           \
} while(0)

void check_cuda(cudaError_t cudaStatus, bool is_cudaGetLastError, int line, const char *file) {
    if (cudaSuccess != cudaStatus) {
        fprintf(stderr, "ERROR: CUDA call in line %d of file '%s' failed with '%s' (cudaStatus=%d).\n",
            line, file, cudaGetErrorString(cudaStatus), cudaStatus);
        if (is_cudaGetLastError) {
            if (cudaGetLastError() == cudaStatus) {
                throw CUDA_GOT_CONTEXT_CORRUPTOR_ERROR;
            } else {
                throw CUDA_ASYNC_ERROR;
            }
        } else {
            if (cudaGetLastError() != cudaSuccess) {
                if (cudaGetLastError() != cudaSuccess) {
                    throw CUDA_GOT_CONTEXT_CORRUPTOR_ERROR;
                } else {
                    throw CUDA_CALL_ERROR;
                }
            }
        }
    }
}


bool all_positive(float *arr, size_t size) {
    for(size_t i = 0; i < size - 1; ++i) {
        if(arr[i] * arr[i+1] < 0) {
            printf("Negative value encountered at index %d.\n", i);
            return false;
        }
        if(arr[i] == 0.0) {
            printf("Zero value encountered at index %d.\n", i);
            return false;
        }
    }
    return true;
}


// Main function (unmangled for python calls):
extern "C" {
int conebeam_tEM_gpu(
    struct Lab lab,
    float *flat, float *px, float *py, float *pz, float *beta,
    float *recon, float *tomo,
    int ngpus, int niter, 
    float tv, float max_val);
}

// Transmission Expectation optimization for Poisson distribution.
enum exeStatus tEM_poisson(
    struct Lab lab,
    float *flat, float *px, float *py, float *pz, float *beta,
    float *recon, float *counts,
    int ngpus, int niter,
    float tv, float max_val); // tv stands for total variation.

// Initial GPU stuff:
void initial_mallocs_and_copies_to_gpus(
    float *flat, float *px, float *py, float *pz, float *beta,
    float **flat_cu, float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    float **new_recon_cu, float **recon_cu,
    size_t size_chunk_recon,
    struct Lab lab,
    cudaStream_t *stream,
    int ngpus);
// To clean remaining GPU stuff (cudaFree, streamDestroy, etc):
void free_gpu_memory(
    float **new_recon_cu, float **recon_cu,
    float **flat_cu, float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    cudaStream_t *stream,
    int ngpus);
// denominator calculator kernel manager:
bool calc_counts_backproj(
    float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    float *counts, float *counts_backproj,
    struct Lab *labs,
    size_t size_recon_bulk,
    int num_of_batches, int ngpus);
// Every iteration kernel's manager.
double update_recon(
    float **flat_cu, float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    float **new_recon_cu, float **recon_cu,
    struct Lab *labs, struct BulkBoundary *boundaries,
    size_t size_recon_bulk_bytes,
    int num_of_batches, int ngpus,
    float *recon, float *backproj, 
    float tv, float max_val,
    cudaStream_t *stream);
// Copy backcounts to GPU:
void copy_backproj_to_gpu(
    float *backproj, float **recon_cu,
    cudaStream_t stream,
    size_t size_recon_bulk_bytes,
    int ibat, int ngpus);
// Called by update_recon:
int launch_batch_of_kernels(
    float **flat_cu, float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    float **new_recon_cu, float **recon_cu,
    float *backproj,
    struct Lab *labs, struct BulkBoundary *boundary,
    cudaStream_t *stream,
    size_t size_recon_bulk,
    int ibat, int ngpus,
    float tv, float max_val);
int copy_recon_to_cpu(
    float *recon,
    float **recon_cu,
    cudaStream_t *stream,
    size_t size_recon_bulk_bytes,
    int ibat, int ngpus);
int copy_recon_to_gpus(
    float *recon,
    float **recon_cu,
    cudaStream_t *stream,
    size_t size_recon_bulk_bytes,
    int ibat, int ngpus);
bool is_valid_data(struct Lab lab, int ngpus);
size_t get_size_recon(struct Lab lab);
int calc_number_of_recon_bulks(int ngpus, float mem_gpu, struct Lab lab);
size_t calc_num_bytes_recon_bulk(int num_of_recon_bulks, struct Lab lab);
void distribute_labs(struct Lab *labs, struct Lab lab, int ndiv);
void distribute_boundaries(
    struct BulkBoundary *boundaries, 
    struct Lab *labs, 
    struct Lab lab, 
    int num_of_recon_bulks);
size_t available_memory_per_gpu();
float get_mem_gpu(int ngpus);
bool check_gpus(int ngpus);