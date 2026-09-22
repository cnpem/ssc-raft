#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include "geometries/conebeam/fdk.hpp"
#include <fstream>
#include <future>
#include <thread>
#include <time.h>
#include <vector>
#include <iostream>

#include "common/logerror.hpp"

using namespace std;

int main(int argc, char *argv[])
{
    return 0;
}

extern "C"
{
    void gpu_fdk(Lab lab, float *recon, float *proj, float *angles,
                 int *gpus, int ndevs, double *time, 
                 float* error_x, float* error_y, float* error_z)
    {
        int i, n_process;

        // int block    = ( lab.slice_recon_end - lab.slice_recon_start );
        int blockgpu = (lab.nv + ndevs - 1) / ndevs;

        if ( ( blockgpu <= ndevs) ) ndevs = 1;
        // printf("Rot axis = %d \n", lab.rotation_axis_offset);
        n_process = memory(lab, ndevs);

        Process *process = (Process *)calloc(n_process, sizeof(Process));

        for (i = 0; i < n_process; i++)
            set_process(lab, i, &process[i], n_process, gpus, ndevs);

        // printf("Filter:\n");
        // clock_t f_begin = clock();

        set_filtering_fft(lab, proj, n_process, ndevs, process);

        // clock_t f_end = clock();
        // time[0] = double(f_end - f_begin) / CLOCKS_PER_SEC;

        HANDLE_ERROR(cudaDeviceSynchronize());

        // printf("Backprojection:\n");
        // clock_t b_begin = clock();

        set_backprojection(lab, recon, proj, angles, n_process, ndevs, process, error_x, error_y, error_z);

        // clock_t b_end = clock();
        // time[1] = double(b_end - b_begin) / CLOCKS_PER_SEC;

        HANDLE_ERROR(cudaDeviceSynchronize());

        free(process);
    }
}

extern "C"
{

    void process_backprojection_worker(Lab lab, float *recon, float* angles, float *proj,
            Process process, float* error_x, float* error_y, float* error_z)
    {
        float *c_proj, *c_recon, *c_beta;
        copy_to_gpu_back(lab, proj, recon, angles, &c_proj, &c_recon, &c_beta, process);
        backprojection(lab, c_recon, c_proj, c_beta, process, error_x, error_y, error_z);
        copy_to_cpu_back(lab, recon, c_proj, c_recon, c_beta, process);
    }

    void set_backprojection(Lab lab, float *recon, float *proj, float *angles,
            int n_process, int ndevs, Process *process, float* error_x, float* error_y, float* error_z)
    {
        thread threads[ndevs];

        for (int k = 0; k < n_process; ++k) {
            const int g = k % ndevs;
            if (threads[g].joinable())
                threads[g].join();
            threads[g] = thread(process_backprojection_worker, lab, recon, angles, proj, process[k], error_x, error_y, error_z);
        }

        for (int g = 0; g < ndevs; ++g) {
            if (threads[g].joinable())
                threads[g].join();
        }
    }
}

extern "C"
{

    void process_filtering_ftt_worker(Lab lab, float* proj, Process process)
    {
        float *c_filter, *c_W;
        cufftComplex *c_signal;
        copy_gpu_filter_fft(lab, proj, &c_filter, &c_signal, &c_W, process);
        fft(lab, c_filter, c_signal, c_W, process);
        copy_cpu_filter_fft(lab, proj, c_filter, c_signal, c_W, process);
    }

    void set_filtering_fft(Lab lab, float *proj, int n_process, int ndevs, Process *process)
    {
        thread threads[ndevs];

        for (int k = 0; k < n_process; ++k) {
            const int g = k % ndevs;
            if (threads[g].joinable())
                threads[g].join();
            threads[g] = thread(process_filtering_ftt_worker, lab, proj, process[k]);
        }

        for (int g = 0; g < ndevs; ++g) {
            if (threads[g].joinable())
                threads[g].join();
        }
    }
}

/* Uncompatible version as of 21/03/2025 - To be removed */
// extern "C"
// {
//     void set_filtering_conv(Lab lab, float *proj, int n_process, int ndevs, Process *process)
//     {

//         int i, k = 0;
//         std::vector<thread> threads_filt;

//         float *c_filter[ndevs];
//         float *c_Q[ndevs];

//         while (k < n_process)
//         {

//             if (k % ndevs == 0)
//             {
//                 for (i = 0; i < ndevs; i++)
//                     copy_gpu_filter_conv(lab, proj, &c_filter[i], &c_Q[i], process[k + i]);
//                 cudaDeviceSynchronize();
//             }

//             threads_filt.emplace_back(thread(filtering_conv, lab, c_filter[k % ndevs], c_Q[k % ndevs], process[k]));
//             k = k + 1;

//             if (k % ndevs == 0)
//             {
//                 for (i = 0; i < ndevs; i++)
//                     threads_filt[i].join();
//                 threads_filt.clear();
//                 cudaDeviceSynchronize();

//                 for (i = 0; i < ndevs; i++)
//                     copy_cpu_filter_conv(proj, c_filter[i], c_Q[i], process[k - ndevs + i]);
//             }
//         }
//     }
// }
