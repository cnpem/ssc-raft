#include <iostream>
#include <new>
#include <chrono>
#include <thread>
#include <cmath>
#include <vector>
#include <algorithm> // std::min_element.
#include <stdlib.h>  // size_t and malloc for backcounts.
#include "geometries/conebeam/emc_manager.cuh"
#include "geometries/conebeam/emc_kernel.cuh"


#define MAX_GPUS 64
#define A100_MEM 39.5

// pode ir pro pacote de utils/cuda_utils pq não é especifico pro tEM:
#define BYTES_TO_GB (1/((double) (1024*1024*1024)))

#define MAX_NITER 1000

#define MAX_NUM_OF_RECON_BULKS 2048

#define NUM_THREADS_UNARY_OPERATION 128
#define NUM_THREADS_BINARY_OPERATION 128
#define NUM_THREADS_PER_BLOCK 128


extern "C" {
int conebeam_tEM_gpu(
    struct Lab lab,
    float *flat,
    float *px,
    float *py,
    float *pz,
    float *beta,
    float *recon,
    float *tomo, // ou int counts?
    int ngpus,
    int niter, 
    float tv,
    float max_val)
{
    enum exeStatus execution_status = SUCCESS;
    auto start = std::chrono::high_resolution_clock::now();
    bool data_ok, gpus_ok;

    gpus_ok = check_gpus(ngpus);
    data_ok = is_valid_data(lab, ngpus);
    if ( data_ok && gpus_ok ) {
        execution_status = tEM_poisson(
            lab, 
            flat, px, py, pz, beta, 
            recon, tomo, 
            ngpus, niter, 
            tv, max_val);
    } else {
        if (!data_ok && gpus_ok ) { 
            execution_status = INVALID_INPUT;
        } else if (!gpus_ok && data_ok) {
            execution_status = GPUS_CHECK_ERROR;
        } else {
            execution_status = INVALID_INPUT_AND_GPUS_CHECK_ERROR;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cout << "Duração (s) da função tEM: " << duration.count()/1000.0 << std::endl;
    std::cout << "Status da função tEM: " << static_cast<int>(execution_status) << std::endl;
    std::cout << "\t(SUCCESS=0; GPU_CHECK_ERROR=-2, INVALID_INPUT=-1; CUDA_ERROR>0)" << std::endl;

    return static_cast<int>(execution_status);
}
}

enum exeStatus tEM_poisson(
    struct Lab lab,
    float *flat, float *px, float *py, float *pz, float *beta,
    float *recon, float *counts,
    int ngpus, int niter,
    float tv, float max_val)
{
    enum exeStatus exe_status = SUCCESS;
    // Loglikelyhood history variable:
    float loglike[MAX_NITER];
    // backcounts:
    float *backproj = static_cast<float*>(malloc(sizeof(float) * get_size_recon(lab)));
    std::cout << "get_size_recon(lab) = " << get_size_recon(lab) << std::endl;
    std::cout << "backproj = " << backproj << std::endl;
    // GPUs streams and pointers:
    float *new_recon_cu[MAX_GPUS], *recon_cu[MAX_GPUS];
    float *flat_cu[MAX_GPUS], *beta_cu[MAX_GPUS], *px_cu[MAX_GPUS], *py_cu[MAX_GPUS], *pz_cu[MAX_GPUS];
    cudaStream_t stream[MAX_GPUS];
    // GPU data management and distribution variables:
    int num_of_recon_bulks = calc_number_of_recon_bulks(ngpus, get_mem_gpu(ngpus), lab);
    int num_of_batches = num_of_recon_bulks/ngpus; // division with no remainder by design.
    size_t size_recon_bulk_bytes = calc_num_bytes_recon_bulk(num_of_recon_bulks, lab);
    size_t size_recon_bulk = size_recon_bulk_bytes / sizeof(float);
    std::cout << "size_recon_bulk = " << size_recon_bulk << std::endl;
    struct Lab *labs = new struct Lab[num_of_recon_bulks]; // corresponding lab for each recon bulk.
    struct BulkBoundary *boundaries = new struct BulkBoundary[num_of_recon_bulks]; // recon bulks boundaries.

    if (backproj == NULL || labs == NULL || boundaries == NULL) {
        exe_status = MALLOC_ERROR;
        std::cout << "backproj memory allocation error." << std::endl;
        return exe_status;
    }

    distribute_labs(labs, lab, num_of_recon_bulks);
    distribute_boundaries(boundaries, labs, lab, num_of_recon_bulks);

    try {
        // (iteration persistent) Memory allocations and copies from CPU to GPUs:
        initial_mallocs_and_copies_to_gpus(
            flat, px, py, pz, beta,
            flat_cu, px_cu, py_cu, pz_cu, beta_cu,
            new_recon_cu, recon_cu,
            size_recon_bulk_bytes, lab, stream, ngpus);
        // Counts backproj:
        calc_counts_backproj(
            px_cu, py_cu, pz_cu, beta_cu, 
            counts, backproj, 
            labs,
            size_recon_bulk,
            num_of_batches, ngpus);
        std::cout << "Starting all positive check for backproj." << std::endl;
        all_positive(backproj, static_cast<size_t>(lab.nx)*lab.ny*lab.nz);
        std::cout << "Finished all positive check for backproj.\n" << std::endl;
        // Para podermos usar mem copy async (verificar se é mesmo necessário):
        CHECK_CUDA( cudaHostRegister(recon, sizeof(float)*lab.nx*lab.ny*lab.nz, cudaHostRegisterDefault) );
        // EM fixed-point iterations:
        std::cout << "Iteration \t Log-likelihood" << std::endl;
        for (int i = 0; i < niter; ++i) {
            loglike[i] = update_recon(
                flat_cu, px_cu, py_cu, pz_cu, beta_cu,
                new_recon_cu, recon_cu,
                labs, boundaries,
                size_recon_bulk_bytes, 
                num_of_batches, ngpus,
                recon, backproj, 
                tv, max_val,
                stream);
            std::cout << i << " \t " << loglike[i] << std::endl;

            if (DEBUG) {
                std::cout << "Starting all positive check for recon." << std::endl;
                all_positive(recon, static_cast<size_t>(lab.nx)*lab.ny*lab.nz);
                std::cout << "Finished all positive check for recon.\n" << std::endl;
            }

            if (0 < i && loglike[i] < loglike[i-1]) {
                std::cout << "Loglikelyhood decreased at iteration:\t  " << i << std::endl;
                // break;
                //   A possible explanation: there are several fixed-points and we just swiched 
                // from one fixed-point sequence to another?
                //   na verdade isso apenas mudaria a taxa de convergencia. não vejo motivo para 
                // a alterancia entre sequencias de ponto-fixo CONVERGENTEs oscilarem dessa forma.
            }
        }

        CHECK_CUDA( cudaHostUnregister(recon) );
        free_gpu_memory(
            new_recon_cu, recon_cu, 
            flat_cu, px_cu, py_cu, pz_cu, beta_cu, 
            stream, ngpus);
        CHECK_CUDA(cudaGetLastError());
        delete[] labs;
        delete[] boundaries;

        std::cout << "Function 'tEM_poisson' executed successfully." << std::endl;

    } catch (enum exeStatus cuda_final_status) {
        exe_status = cuda_final_status;
        // DESALOCAR AS COISAS DE CUDA.
        std::cout << "CUDA ERROR FOUND." << std::endl;
    }

    free(backproj);

    return exe_status;
}

void initial_mallocs_and_copies_to_gpus(
    float *flat, float *px, float *py, float *pz, float *beta,
    float **flat_cu, float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    float **new_recon_cu, float **recon_cu,
    size_t size_recon_bulk_bytes,
    struct Lab lab,
    cudaStream_t *stream,
    int ngpus) 
{
    // Create cudaStreams:
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaStreamCreate(&stream[i]) );
    }
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaSetDevice(i) );
        // Memory allocations:
        CHECK_CUDA( cudaMalloc(&new_recon_cu[i], size_recon_bulk_bytes) );
        CHECK_CUDA( cudaMalloc(&recon_cu[i], size_recon_bulk_bytes) );
        CHECK_CUDA( cudaMalloc(&flat_cu[i], lab.ndetc * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&px_cu[i], lab.ndetc * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&py_cu[i], lab.ndetc * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&pz_cu[i], lab.ndetc * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&beta_cu[i], lab.nbeta * sizeof(float)) );
        // Copies from CPU to GPU:
        CHECK_CUDA( cudaMemcpy(flat_cu[i], flat, lab.ndetc * sizeof(float), cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(px_cu[i], px, lab.ndetc * sizeof(float), cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(py_cu[i], py, lab.ndetc * sizeof(float), cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(pz_cu[i], pz, lab.ndetc * sizeof(float), cudaMemcpyHostToDevice) );
        CHECK_CUDA( cudaMemcpy(beta_cu[i], beta, lab.nbeta * sizeof(float), cudaMemcpyHostToDevice) );
    }
    // No need to cudaGetLastError since all calls in this function are sync.
}


// To clean remaining GPU stuff (cudaFree, streamDestroy, etc):
void free_gpu_memory(
    float **new_recon_cu, float **recon_cu,
    float **flat_cu, 
    float **px_cu, float **py_cu, float **pz_cu, 
    float **beta_cu,
    cudaStream_t *stream,
    int ngpus) 
{
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaStreamDestroy(stream[i]) );
        CHECK_CUDA( cudaFree(new_recon_cu[i]) );
        CHECK_CUDA( cudaFree(recon_cu[i]) );
        CHECK_CUDA( cudaFree(flat_cu[i]) );
        CHECK_CUDA( cudaFree(px_cu[i]) );
        CHECK_CUDA( cudaFree(py_cu[i]) );
        CHECK_CUDA( cudaFree(pz_cu[i]) );
        CHECK_CUDA( cudaFree(beta_cu[i]) );
    }
}



bool calc_counts_backproj(
    float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    float *counts, float *counts_backproj,
    struct Lab *labs,
    size_t size_recon_bulk,
    int num_of_batches, int ngpus)
{
    bool ok = true;
    size_t size_tomo = static_cast<size_t>(labs[0].ndetc) * labs[0].nbeta;
    size_t num_blocks = size_tomo / NUM_THREADS_PER_BLOCK;
    // size_t num_blocks_reciprocal = size_recon_bulk / NUM_THREADS_UNARY_OPERATION;
    size_t num_blocks_reciprocal = size_recon_bulk % NUM_THREADS_UNARY_OPERATION == 0 ?
        size_recon_bulk/NUM_THREADS_UNARY_OPERATION : (size_recon_bulk/NUM_THREADS_UNARY_OPERATION) + 1;
    float *counts_cu[MAX_GPUS];
    float *backproj_cu[MAX_GPUS];

    // Mallocs:
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaMalloc(&counts_cu[i], size_tomo * sizeof(float)) );
        CHECK_CUDA( cudaMalloc(&backproj_cu[i], size_recon_bulk * sizeof(float)) );
    }
    // Copies and memset:
    for (int ibat = 0; ibat < num_of_batches; ++ibat) {
        for (int i = 0; i < ngpus; ++i) {
            CHECK_CUDA( cudaSetDevice(i) );
            CHECK_CUDA( cudaMemcpy(
                counts_cu[i],
                counts,
                size_tomo * sizeof(float), cudaMemcpyHostToDevice) );
            CHECK_CUDA( cudaMemset(backproj_cu[i], 0, size_recon_bulk * sizeof(float)) );
                // https://forums.developer.nvidia.com/t/can-i-set-a-floats-to-zero-with-cudamemset/153706
        }
        for (int i = 0; i < ngpus; ++i) {
            CHECK_CUDA( cudaSetDevice(i) );
            backpropagation<<<num_blocks, NUM_THREADS_PER_BLOCK>>>(
                px_cu[i], py_cu[i], pz_cu[i],
                beta_cu[i],
                counts_cu[i],
                backproj_cu[i],
                labs[i + ibat*ngpus]);
            CHECK_CUDA( cudaPeekAtLastError() );
            DEBUG_CUDA_CHECK();
            reciprocal<<<num_blocks_reciprocal, NUM_THREADS_UNARY_OPERATION>>>(
                backproj_cu[i],
                size_recon_bulk);
            CHECK_CUDA( cudaPeekAtLastError() );
            DEBUG_CUDA_CHECK();
        }
        CHECK_CUDA( cudaDeviceSynchronize() ); // catch kernel runtime error.

        for (int i = 0; i < ngpus; ++i) {
            CHECK_CUDA( cudaSetDevice(i) );
            std::cout << "(i + ibat*ngpus)*size_recon_bulk = " << (i + ibat*ngpus)*size_recon_bulk << std::endl;
            std::cout << "counts_backproj[(i + ibat*ngpus)*size_recon_bulk] = " << counts_backproj[(i + ibat*ngpus)*size_recon_bulk] << std::endl;
            
            CHECK_CUDA( cudaMemcpy(
                &counts_backproj[(i + ibat*ngpus)*size_recon_bulk], 
                backproj_cu[i],
                size_recon_bulk * sizeof(float), cudaMemcpyDeviceToHost) );
        }
    }

    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaSetDevice(i) );
        CHECK_CUDA( cudaFree(backproj_cu[i]) );
        CHECK_CUDA( cudaFree(counts_cu[i]) );
    }

    return ok;
}


double update_recon(
    float **flat_cu, float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    float **new_recon_cu, float **recon_cu,
    struct Lab *labs, struct BulkBoundary *boundaries,
    size_t size_recon_bulk_bytes,
    int num_of_batches, int ngpus,
    float *recon, float *backproj,
    float tv, float max_val,
    cudaStream_t *stream)
{
    double log_likelihood = 0;
    std::cout << "number of batches: " << num_of_batches << std::endl;
    for (int ibat = 0; ibat < num_of_batches; ++ibat) {
        copy_recon_to_gpus(recon, recon_cu, stream, size_recon_bulk_bytes, ibat, ngpus);
        launch_batch_of_kernels(
            flat_cu, px_cu, py_cu, pz_cu, beta_cu,
            new_recon_cu, recon_cu,
            backproj,
            labs, boundaries,
            stream, 
            size_recon_bulk_bytes/sizeof(float), // arrumar (trocar bulk_bytes por bulk apenas)
            ibat, ngpus, 
            tv, max_val);
        copy_recon_to_cpu(recon, new_recon_cu, stream, size_recon_bulk_bytes, ibat, ngpus);
    }
    
    return log_likelihood;
}


void copy_backproj_to_gpu(
    float *backproj, float **recon_cu, 
    cudaStream_t *stream, 
    size_t size_recon_bulk_bytes,
    int ibat, int ngpus)
{
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaMemcpyAsync(
            recon_cu[i+ibat*ngpus], 
            &backproj[(i+ibat*ngpus)*size_recon_bulk_bytes], 
            size_recon_bulk_bytes, 
            cudaMemcpyHostToDevice, stream[i]) );
    }
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaStreamSynchronize(stream[i]) );
    }
}


int launch_batch_of_kernels(
    float **flat_cu, float **px_cu, float **py_cu, float **pz_cu, float **beta_cu,
    float **new_recon_cu, float **recon_cu,
    float *backproj,
    struct Lab *labs, struct BulkBoundary *boundary,
    cudaStream_t *stream,
    size_t size_recon_bulk,
    int ibat, int ngpus, 
    float tv, float max_val)
{
    size_t num_blocks = (static_cast<size_t>(labs[0].ndetc) * labs[0].nbeta) / NUM_THREADS_PER_BLOCK;
    size_t num_blocks_mult = size_recon_bulk % NUM_THREADS_BINARY_OPERATION == 0 ? 
        size_recon_bulk/NUM_THREADS_BINARY_OPERATION : (size_recon_bulk/NUM_THREADS_BINARY_OPERATION) + 1;
    size_t num_blocks_tv = size_recon_bulk % NUM_THREADS_UNARY_OPERATION == 0 ? 
        size_recon_bulk/NUM_THREADS_UNARY_OPERATION : (size_recon_bulk/NUM_THREADS_UNARY_OPERATION) + 1;
    size_t num_blocks_limits = size_recon_bulk % NUM_THREADS_UNARY_OPERATION == 0 ? 
        size_recon_bulk/NUM_THREADS_UNARY_OPERATION : (size_recon_bulk/NUM_THREADS_UNARY_OPERATION) + 1;
    float *backproj_of_counts_cu;

std::cout << "num_blocks_limits = " << num_blocks_limits << std::endl;
std::cout << "labs[0].nx = " << labs[0].nx << std::endl;
std::cout << "labs[0].ny = " << labs[0].ny << std::endl;
std::cout << "labs[0].nz = " << labs[0].nz << std::endl;

    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaSetDevice(i) );
        // Set the new_recon_cu array to zero:
        CHECK_CUDA( cudaMemsetAsync(
            new_recon_cu[i], // pointer to GPU memory.
            0, // value to set each byte of memory in the region.
            size_recon_bulk*sizeof(float), // size of the memory region (in bytes).
            stream[i]) ); // stream.
        // Launch the backprojection of radon kernel:
        backproj_of_radon_2<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream[i]>>>(
            flat_cu[i],
            px_cu[i], py_cu[i], pz_cu[i],
            beta_cu[i],
            recon_cu[i],
            new_recon_cu[i],
            labs[i + ibat*ngpus],
            boundary[i + ibat*ngpus]);
        CHECK_CUDA( cudaPeekAtLastError() );
        DEBUG_CUDA_CHECK();
    }

    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaSetDevice(i) );
        // f(n+1) = f(n) * backproj_radon:
        multiply<<<num_blocks_mult, NUM_THREADS_BINARY_OPERATION, 0, stream[i]>>>(
            new_recon_cu[i],
            recon_cu[i],
            size_recon_bulk);
        CHECK_CUDA( cudaPeekAtLastError() );
        DEBUG_CUDA_CHECK();
        // regularization (kind of tv -total variation):
        if (tv > 0.0) {
            total_variation_2d<<<num_blocks_tv, NUM_THREADS_UNARY_OPERATION, 0, stream[i]>>>(
                new_recon_cu[i],
                size_recon_bulk,
                labs[i + ibat*ngpus],
                tv);
            CHECK_CUDA( cudaPeekAtLastError() );
            DEBUG_CUDA_CHECK();
        }
        // Copy the (reciprocal of the) backprojection of counts to GPU:
        backproj_of_counts_cu = recon_cu[i]; // using recon_cu empty shell.
        CHECK_CUDA( cudaMemcpyAsync(
            backproj_of_counts_cu, 
            &backproj[(i+ibat*ngpus)*size_recon_bulk], 
            size_recon_bulk*sizeof(float), 
            cudaMemcpyHostToDevice, stream[i]) );
        // Divide (multiply) the new_recon_cu array by the (reciprocal of) backprojection of counts:
        multiply<<<num_blocks_mult, NUM_THREADS_BINARY_OPERATION, 0, stream[i]>>>(
            new_recon_cu[i],
            backproj_of_counts_cu, // *backproj_counts.
            size_recon_bulk);
        CHECK_CUDA( cudaPeekAtLastError() );
        DEBUG_CUDA_CHECK();
        // max value constraint:
        if (max_val > 0.0) {
            apply_limits<<<num_blocks_limits, NUM_THREADS_UNARY_OPERATION, 0, stream[i]>>>(
                new_recon_cu[i],
                max_val,
                size_recon_bulk);
            CHECK_CUDA( cudaPeekAtLastError() );
            DEBUG_CUDA_CHECK();
        }
    }
    
    std::cout << "tv = " << tv << std::endl;
    std::cout << "max_val = " << max_val << std::endl;
    CHECK_CUDA( cudaDeviceSynchronize() );

    return 0;
}

// atualiza a recon na RAM da CPU:
int copy_recon_to_cpu(
    float *recon, 
    float **recon_cu,
    cudaStream_t *stream,
    size_t size_recon_bulk_bytes,
    int ibat,
    int ngpus)     
{
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaMemcpyAsync(
            &recon[(i+ibat*ngpus)*size_recon_bulk_bytes/sizeof(float)], // arrumar 
            recon_cu[i],
            size_recon_bulk_bytes,
            cudaMemcpyDeviceToHost, stream[i]) );
    }
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaStreamSynchronize(stream[i]) );
    }
    return 0;
}

// atualiza a recon nas GPUs:
int copy_recon_to_gpus(
    float *recon, 
    float **recon_cu,
    cudaStream_t *stream,
    size_t size_recon_bulk_bytes,
    int ibat,
    int ngpus)
{
    std::cout << "ibat: " << ibat << std::endl;
    std::cout << "ngpus: " << ngpus << std::endl;
    std::cout << "size_recon_bulk_bytes: " << size_recon_bulk_bytes << std::endl;
    for (int i = 0; i < ngpus; ++i) {
        std::cout << "recon_cu[i]: " << recon_cu[i] << std::endl;
        std::cout << "recon[(i+ibat*ngpus)*size_recon_bulk_bytes/sizeof(float)]: " << recon[(i+ibat*ngpus)*size_recon_bulk_bytes/sizeof(float)] << std::endl;
        CHECK_CUDA( cudaMemcpyAsync(
            recon_cu[i],
            &recon[(i+ibat*ngpus)*size_recon_bulk_bytes/sizeof(float)],
            size_recon_bulk_bytes,
            cudaMemcpyHostToDevice, stream[i]) );
    }
    for (int i = 0; i < ngpus; ++i) {
        CHECK_CUDA( cudaStreamSynchronize(stream[i]) );
    }
    return 0;
}


bool is_valid_data(struct Lab lab, int ngpus) {
    bool ok = false;
    // Check se não tem nada absurdo:
    if (   lab.nz % ngpus != 0 // Por enquanto vamos admitir apenas blocos de recon que podem ser divididos igualmente entre as GPUs.
        || ngpus > MAX_GPUS
        || lab.nz > 2048 // um passo de cada vez.
        || lab.nz < ngpus
    ) {
        std::cout << "lab.nz: " << lab.nz << std::endl;
        std::cout << "ngpus: " << ngpus << std::endl;
        std::cout << "parametros invalidos.";
    } else {
        ok = true;
    }
    return ok;
}


size_t get_size_recon(struct Lab lab) {
    return static_cast<size_t>(lab.nx)*static_cast<size_t>(lab.ny)*static_cast<size_t>(lab.nz);
}


int calc_number_of_recon_bulks(int ngpus, float mem_gpu, struct Lab lab) {
    int num_recon_bulks;
    float mem_available;
    float perc_max_occupancy = 80.0/100.0; // the rest is for the localthread mem for paths.
    float mem_recon = BYTES_TO_GB * static_cast<float>(sizeof(int)) * static_cast<float>(lab.nx) * lab.ny * lab.nz;
    float mem_required = 3*mem_recon; // rever essa conta; quanto menos memoria alocar, melhor.
      // mem_req = fk central bulk + fk upper bulk + fk lower bulk 
      //         + fk+1.

    for (int i = 1; i < MAX_NUM_OF_RECON_BULKS; ++i) {
        num_recon_bulks = i*ngpus;
        mem_available = perc_max_occupancy * static_cast<float>(num_recon_bulks) * mem_gpu;
        if (mem_available > mem_required) {
            break;
        }
    }
    std::cout << std::endl;
    std::cout << "num recon bulks = " << num_recon_bulks << std::endl;
    std::cout << std::endl;
    return num_recon_bulks;
}


size_t calc_num_bytes_recon_bulk(int num_of_recon_bulks, struct Lab lab) {
    return sizeof(float) * static_cast<size_t>(lab.nx)*lab.ny*lab.nz / num_of_recon_bulks;
}

size_t calc_size_recon_bulk(int num_of_recon_bulks, struct Lab lab) {
    if (lab.nz % num_of_recon_bulks != 0) {
        return 0;
    }
    return sizeof(float) * static_cast<size_t>(lab.nx)*lab.ny*lab.nz / num_of_recon_bulks;
}


void distribute_labs(struct Lab *labs, struct Lab lab, int ndiv) {

    if (lab.nz % ndiv != 0) {
        std::cout << "número de divisões da recon tem resto.";
        return;  // rever esse lixinho aqui.
    }

    std::cout << "Criando labs:" << std::endl;
    for (int i = 0; i < ndiv; ++i) {
        std::cout << "\t  índice do lab: " << i << std::endl;
        labs[ndiv-i-1] = lab;
        // Update variables:
        labs[ndiv-i-1].Lz = lab.Lz/ndiv;
        labs[ndiv-i-1].nz = lab.nz/ndiv;
        labs[ndiv-i-1].z0 = lab.z0 + (ndiv - 1)*labs[ndiv-i-1].Lz - i*2*labs[ndiv-i-1].Lz;
        std::cout << "\t  lab.z0: " << labs[ndiv-i-1].z0 << std::endl;
    }
}

void distribute_boundaries(
    struct BulkBoundary *boundaries, 
    struct Lab *labs, 
    struct Lab lab, 
    int num_of_recon_bulks) 
{
    // int nz_up, nz_lw;
    double delta_up, delta_lw;
    double tan_up, tan_lw;
    double Dsr; // distance: source <-> rotation axis.
    double Lxy; // distance: diagonal of a recon slice z = cte.
    double dz = 2*lab.Lz / static_cast<double>(lab.nz);

    Lxy = std::sqrt(lab.Lx*lab.Lx + lab.Ly*lab.Ly);
    Dsr = std::sqrt(lab.sx*lab.sx + lab.sy*lab.sy + lab.sz*lab.sz);

    for (int i = 0; i < num_of_recon_bulks; ++i) {
        tan_up = (labs[i].z0 + labs[i].Lz) / (Dsr - Lxy);
        tan_lw = (labs[i].z0 - labs[i].Lz) / (Dsr + Lxy);
        delta_up = 2*Lxy*tan_up;
        delta_lw = 2*Lxy*tan_lw;
        boundaries[i].nz_upper_width = static_cast<int>(std::ceil(std::abs(delta_up/dz)));
        boundaries[i].nz_lower_width = static_cast<int>(std::ceil(std::abs(delta_lw/dz)));
        if (i == 0 || i == -1 + num_of_recon_bulks) {
            boundaries[i].outter_boundary = true;
        } else {
            boundaries[i].outter_boundary = false;
        }
    }
}


float get_mem_gpu(int ngpus) {
    return A100_MEM;
}


bool check_gpus(int ngpus) {
    bool ok = true;
    float mem_gpu;
    std::vector<size_t> global_mem(ngpus);
    std::vector<size_t> shared_mem(ngpus);
    int number_of_gpus_available;
    cudaDeviceProp props;

    try {
        CHECK_CUDA( cudaGetDeviceCount(&number_of_gpus_available) );
        if (number_of_gpus_available < ngpus) {
            ok = false;
            std::cout << "There aren't as many GPUs available as requested:" << std::endl;
            std::cout << "Requested: " << ngpus << std::endl;
            std::cout << "Available: " << number_of_gpus_available << std::endl;
            ngpus = number_of_gpus_available; // to check the devices properties before exiting.
        }

        for (int i = 0; i < ngpus; ++i) {
            CHECK_CUDA( cudaSetDevice(i) );
            CHECK_CUDA( cudaGetDeviceProperties(&props, i) );
            global_mem[i] = props.totalGlobalMem;
            std::cout << "\nDevice " << i << std::endl;
            std::cout << "\t  Name: " << props.name << std::endl;
            std::cout << "\t  Total global memory: " 
                      << global_mem[i]*BYTES_TO_GB 
                      << " GB" << std::endl;
            std::cout << "\t  Maximum shared memory per block: " 
                      << props.sharedMemPerBlock/1024 
                      << " KB" << std::endl;

            // checa se as GPUs são todas do mesmo modelo.
        }
    }
    catch (enum exeStatus error) { // CUDA error raised from CHECK_CUDA.
        ok = false;
    }

    mem_gpu = static_cast<float>(*std::min_element(global_mem.begin(), global_mem.end()))
             * BYTES_TO_GB;
    if (mem_gpu < A100_MEM) {
        ok = false;
        std::cout << "ERROR: found GPU with insufficient memory: " 
                  << mem_gpu << " GB."
                  << std::endl;
    }
    return ok;
}