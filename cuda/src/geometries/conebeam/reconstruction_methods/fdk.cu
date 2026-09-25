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
                 int *gpus, int ndevs)
    {
        int i, n_process;

        int blockgpu = (lab.nv + ndevs - 1) / ndevs;

        if ( ( blockgpu <= ndevs) ) ndevs = 1;
        // printf("Rot axis = %d \n", lab.rotation_axis_offset);
        n_process = memory(lab, ndevs);

        Process *process = (Process *)calloc(n_process, sizeof(Process));

        for (i = 0; i < n_process; i++)
            set_process(lab, i, &process[i], n_process, gpus, ndevs);

        HANDLE_ERROR(cudaDeviceSynchronize());

        set_backprojection(lab, recon, proj, angles, n_process, ndevs, process);

        HANDLE_ERROR(cudaDeviceSynchronize());

        free(process);
    }
}

extern "C"
{

    void process_backprojection_worker(Lab lab, float *recon, float* angles, float *proj,
            Process process)
    {
        float *c_proj, *c_recon, *c_beta;
        copy_to_gpu_back(lab, proj, recon, angles, &c_proj, &c_recon, &c_beta, process);
        backprojection(lab, c_recon, c_proj, c_beta, process);
        copy_to_cpu_back(lab, recon, c_proj, c_recon, c_beta, process);
    }

    void set_backprojection(Lab lab, float *recon, float *proj, float *angles,
            int n_process, int ndevs, Process *process)
    {
        thread threads[ndevs];

        for (int k = 0; k < n_process; ++k) {
            const int g = k % ndevs;
            if (threads[g].joinable())
                threads[g].join();
            threads[g] = thread(process_backprojection_worker, lab, recon, angles, proj, process[k]);
        }

        for (int g = 0; g < ndevs; ++g) {
            if (threads[g].joinable())
                threads[g].join();
        }
    }
}


