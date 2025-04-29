#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include "geometries/conebeam/fdk.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <thread>
#include <time.h>
#include <vector>
#include <iostream>


extern "C" {
void set_process(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs) {   

    // Variables for the reconstruction volume
    int nz_gpu_recon, zi_min_recon, zi_max_recon, zi_recon;
    long long int n_recon, idx_recon;
    long long int n_recon_pad, idx_recon_pad;
    float z_min, z_max, L, Lx;

    float block = 0.0f;

    // Variables for the projection volume
    float Z_min, Z_max;
    int Zi_min, Zi_max, zi_proj;
    long long int n_proj, idx_proj;
    long long int n_proj_pad, idx_proj_pad;

    // Variables for the filter volumes
    int nv_gpu_filter, zi_min_filter, zi_max_filter;
    long long int n_filter, idx_filter, n_filter_pad, idx_filter_pad;
    int zi_filter, zi_filter_pad;

    // --- Divide the reconstruction volume among processes ---
    // Paola version:
    // nz_gpu_recon = (int) ceil((float) lab.nz / n_process);
    // // nz_gpu_recon = (lab.nz + n_process - 1) / n_process;
    // zi_min_recon = i * nz_gpu_recon;
    // block        = std::min(nz_gpu_recon, lab.nz - zi_min_recon);
    // zi_max_recon = zi_min_recon + block;

    // Original version:
    nz_gpu_recon = (int) ceil((float) lab.nz / n_process);
    zi_min_recon = i * nz_gpu_recon;
    zi_max_recon = std::min((i + 1) * nz_gpu_recon, lab.nz);

    n_recon       = (long long int) (zi_max_recon - zi_min_recon) * lab.nx * lab.ny;
    idx_recon     = (long long int) zi_min_recon * lab.nx * lab.ny;
    n_recon_pad   = (long long int) (zi_max_recon - zi_min_recon) * lab.nph * lab.nph;
    idx_recon_pad = (long long int) zi_min_recon * lab.nph * lab.nph;

    zi_recon      = (zi_max_recon - zi_min_recon);

    // printf("process: %d \n", i);
    // printf("zi_recon: %d \n", zi_recon);

    z_min = -lab.z + zi_min_recon * lab.dz;
    z_max = -lab.z + zi_max_recon * lab.dz;

    // --- Calculate the required range in the projection volume based on z_min and z_max ---
    // L = sqrt(lab.x * lab.x + lab.y * lab.y);

    // --- Calculate the required range in the PADDED projection volume based on z_min and z_max ---
    float Rxpad = lab.dx * (float)lab.nph / 2.0f;
    float Rypad = lab.dy * (float)lab.nph / 2.0f;

    L = sqrt(Rxpad * Rxpad + Rypad * Rypad);

    Z_min = std::max(-lab.v, std::min(
        lab.Dsd * z_min / (lab.D - L),
        lab.Dsd * z_min / (lab.D + L)
    ));
    Z_max = std::min(+lab.v, std::max(
        lab.Dsd * z_max / (lab.D + L),
        lab.Dsd * z_max / (lab.D - L)
    ));

    // Change ceil and floor here?
    Zi_min = std::max(0, (int) floor((Z_min + lab.v) / lab.dv));
    Zi_max = std::min(lab.nv, (int) ceil((Z_max + lab.v) / lab.dv));

    idx_proj = (long long int) Zi_min * lab.nbeta * lab.nh;
    n_proj   = (long long int) (Zi_max - Zi_min) * lab.nbeta * lab.nh;

    idx_proj_pad = (long long int) Zi_min * lab.nbeta * lab.nph;
    n_proj_pad   = (long long int) (Zi_max - Zi_min) * lab.nbeta * lab.nph;
    zi_proj      = (Zi_max - Zi_min);

    // printf("zi_proj: %d \n", zi_proj);

    // --- Calculate filter volumes based on the projection indices ---
    // Original version:
    nv_gpu_filter = (int) ceil((float) lab.nv / n_process);
    zi_min_filter = i * nv_gpu_filter;
    zi_max_filter = std::min((i + 1) * nv_gpu_filter, lab.nv);

    // Paola version:
    // block         = 0.0f;
    // nv_gpu_filter = (int) ceil((float) lab.nv / n_process);
    // // nv_gpu_filter = (lab.nv + n_process - 1) / n_process;
    // zi_min_filter = i * nv_gpu_filter;
    // block         = std::min(nv_gpu_filter, lab.nv - zi_min_filter);
    // zi_max_filter = zi_min_filter + block;

    n_filter   = (long long int) (zi_max_filter - zi_min_filter)*lab.nbeta*lab.nh;
    idx_filter = (long long int) zi_min_filter*lab.nbeta*lab.nh;
    zi_filter  = zi_max_filter - zi_min_filter;

    n_filter_pad   = (long long int) (zi_max_filter - zi_min_filter) * lab.nbeta * lab.nph;
    idx_filter_pad = (long long int) zi_min_filter * lab.nbeta * lab.nph;
    zi_filter_pad  = zi_max_filter - zi_min_filter;

    // --- Populate the process structure ---
    (*process).i = i;
    (*process).i_gpu = gpus[i % ndevs];  

    (*process).zi = Zi_min;
    (*process).z_proj = zi_proj;
    (*process).idx_proj = idx_proj;
    (*process).n_proj = n_proj;
    (*process).idx_proj_pad = idx_proj_pad;
    (*process).n_proj_pad = n_proj_pad;

    (*process).n_filter = n_filter;
    (*process).idx_filter = idx_filter;
    (*process).z_filter = zi_filter; 

    (*process).n_filter_pad = n_filter_pad;
    (*process).idx_filter_pad = idx_filter_pad;
    (*process).z_filter_pad = zi_filter_pad; 

    (*process).n_recon = n_recon;
    (*process).z_recon = zi_recon;
    (*process).idx_recon = idx_recon;
    (*process).z_ph = z_min;
    (*process).z_det = -lab.v + Zi_min * lab.dv;
    (*process).n_recon_pad = n_recon_pad;
    (*process).idx_recon_pad = idx_recon_pad;
}
}

extern "C" {
    int memory(Lab lab, int ndev) {

        int blockgpu = (int) ceil((float) lab.nz / ndev); //(lab.nv + ndev - 1) / ndev;

        size_t total_required_mem_per_slice_bytes = (
        5 * static_cast<float>(sizeof(float)) * lab.nh  * lab.nbeta + // Tomo slic e
            static_cast<float>(sizeof(float)) * lab.nh  * lab.nh    + // Reconstructed object slice
            static_cast<float>(sizeof(float)) * lab.nph * lab.nph   + // Reconstructed object padded slice
        5 * static_cast<float>(sizeof(float)) * lab.nph * lab.nbeta + // Tomo padded slice + filter kernel
            static_cast<float>(sizeof(float)) * lab.nbeta             // angles
        ); 

        int blocksize = lab.blocksize;

        if ( blocksize == 0 ){
            int blocksize_aux  = compute_GPU_blocksize( blockgpu, 
                                                        total_required_mem_per_slice_bytes, 
                                                        true, 
                                                        BYTES_TO_GB * getTotalDeviceMemory());

            blocksize          = min(blockgpu, blocksize_aux);
        }

        int n_process = (int)ceil( (float) blockgpu / blocksize ) * ndev;

        // printf("Blocksize: %d \n", blocksize);
        // printf("n_process: %d \n", n_process);
        // printf("blockgpu: %d \n", blockgpu);

        return n_process;
    }
}