#include <iostream>
#include <chrono>
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
}