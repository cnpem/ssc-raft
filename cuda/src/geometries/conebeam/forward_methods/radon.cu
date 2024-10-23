#include <stddef.h>
#include <math.h>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <future>
#include <thread>
#include "common/configs.hpp"
#include "geometries/conebeam/radon.cuh"


extern "C" {
    int cbradon(
            struct Lab lab,
            float *px,
            float *py,
            float *pz,
            float *beta,
            float *sample,
            float *tomo,
            int gpu_id) {
        int status_code = -1;
        enum Optimize optimize_path = OPT_ON; // vamos receber de python na verdade.
        auto start = std::chrono::high_resolution_clock::now();

        // usar try catch na ray_integral:
        status_code = ray_integral(lab, px, py, pz, beta, sample, tomo, optimize_path, gpu_id);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Duração da função ray_integral para a GPU " << gpu_id << ": " << duration.count()/1000.0 << std::endl;

        return status_code;
    }

    int calc_nchunks(Lab lab, float gpu_mem_gb) {
        const float tomo_size_gb = float(lab.nbeta) * float(lab.ny * lab.nx) * sizeof(float) / 1e9f;
        const float phantom_size_gb = float(lab.nz) * float(lab.ny * lab.nx) * sizeof(float) / 1e9f;
        printf("sizes = %f %f\n", tomo_size_gb, phantom_size_gb);
        const int n_chunks = ceil(tomo_size_gb / (0.95f * gpu_mem_gb - phantom_size_gb));
        printf("chunks = %d\n", n_chunks);
        return n_chunks;
    }

    int cbradon_MultiGPU(
            int* gpus, int ngpus,
            struct Lab lab,
            float *px, float *py, float *pz,
            float *beta,
            float *sample,
            float *tomo) {

        const float total_gpu_mem_gb = getTotalDeviceMemory() / 1e9f;
        const int n_chunks = calc_nchunks(lab, total_gpu_mem_gb);
        const size_t beta_chunk_size = lab.nbeta / n_chunks;
        const size_t tomo_chunk_size = beta_chunk_size * size_t(lab.nx * lab.ny);

        int status_code = 0;
        enum Optimize optimize_path = OPT_ON; // vamos receber de python na verdade.
        auto start = std::chrono::high_resolution_clock::now();

        std::future<int> threads[ngpus];

        for (int chunk = 0; chunk < n_chunks; ++chunk) {
            const int gpu = chunk % ngpus;
            if (threads[gpu].valid())
                status_code |= threads[gpu].get();

            Lab chunkLab = lab;
            chunkLab.nbeta = min(beta_chunk_size, lab.nbeta - beta_chunk_size * chunk);

            threads[gpu] = std::async(ray_integral, chunkLab, px, py, pz,
                    beta + beta_chunk_size * chunk,
                    sample, tomo + tomo_chunk_size * chunk,
                    optimize_path, gpu);
        }

        for(int g = 0; g < ngpus; ++g) {
            if (threads[g].valid())
                status_code |= threads[g].get();
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        return status_code;
    }
}
