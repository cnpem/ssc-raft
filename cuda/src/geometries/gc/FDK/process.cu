#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include "../../../../inc/gc/fdk.h"
#include <fstream>
#include <future>
#include <thread>
#include <time.h>
#include <vector>
#include <iostream>

extern "C"{
void set_process(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs){   

    long long int  n_filter, idx_filter, n_recon, idx_recon, n_proj, idx_proj;
    float z_min, z_max, Z_min, Z_max, L;  
    int nz_gpu,  zi_min, zi_max, Zi_max, Zi_min, zi_filter; 

    nz_gpu = (int) (lab.nv/n_process);
    zi_min = i*nz_gpu;
    zi_max = (int) std::min((i+1)*nz_gpu, lab.nv);

    n_filter = (long long int) (zi_max - zi_min)*lab.nbeta*lab.nh;
    idx_filter = (long long int) zi_min*lab.nbeta*lab.nh;
    zi_filter = (int) (n_filter/(lab.nbeta*lab.nh));

    printf("Process = %d: n_filter = %lld, idx_filter = %lld, z_filter = %d\n",   i, n_filter, idx_filter, zi_filter);
    
    nz_gpu = (int) (lab.nz/n_process);
    zi_min = i*nz_gpu;
    zi_max = (int) std::min((i+1)*nz_gpu, lab.nz);

    n_recon = (long long int) (zi_max - zi_min)*lab.nx*lab.ny;
    idx_recon = (long long int) zi_min*lab.nx*lab.ny;

    z_min = - lab.z + zi_min*lab.dz;
    z_max = - lab.z + zi_max*lab.dz;
    
    printf("Process = %d:  n_recon  = %lld,  idx_recon = %lld, z_ph = %f \n",  i, n_recon, idx_recon, z_min);
    
    L = sqrt(lab.x*lab.x + lab.y*lab.y);
    //variáveis de9tector
   	Z_min = std::max(-lab.v, std::min(  lab.Dsd*z_min/(lab.D - L),  
                                        lab.Dsd*z_min/(lab.D + L) ));
    Z_max = std::min(+lab.v, std::max(  lab.Dsd*z_max/(lab.D + L),
                                        lab.Dsd*z_max/(lab.D - L))); 

    Zi_min = std::max(0, (int) floor((Z_min + lab.v)/lab.dv));
    Zi_max = std::min(lab.nv, (int) ceil((Z_max + lab.v)/lab.dv));

    idx_proj = (long long int) (Zi_min)*(lab.nbeta*lab.nh);
    n_proj = (long long int) (Zi_max+1-Zi_min)*(lab.nbeta*lab.nh);

    (*process).i = i;
    (*process).i_gpu = (int) gpus[i%ndevs];  
    (*process).zi = Zi_min;
    (*process).idx_proj = idx_proj;
    (*process).n_proj = n_proj;
    (*process).n_filter = n_filter;
    (*process).idx_filter = idx_filter;
    (*process).z_filter = zi_filter; 
    (*process).n_recon = n_recon;
    (*process).idx_recon = idx_recon;
    (*process).z_ph = z_min;
    (*process).z_det = - lab.v + Zi_min*lab.dv;

    printf("Process = %lld:  n_proj = %lld,   idx_proj = %lld, Z_det = %f, z_det = %f, zi = %d\n\n",   (*process).i, (*process).n_proj, (*process).idx_proj, Z_min, (*process).z_det, (*process).zi);
    
}}

extern "C"{
int memory(Lab lab, int ndev){
    long double mem_gpu, mem_recon, mem_proj;
    int n_process;
    long long int n_proj, n_recon;
    // int flag, count, i;
    // float z_min, z_max, Z_min, Z_max;
    // int Zi_min, Zi_max, nz_gpu, zi_min, zi_max;

    
    n_proj = (long long int)(lab.nv)*(long long int)(lab.nbeta)*(long long int)(lab.nh);
    n_recon = (long long int)(lab.nx)*(long long int)(lab.ny)*(long long int)(lab.nz);

    mem_gpu = 40;
    mem_proj = 32*n_proj*1.16*(pow(10,-10));
    mem_recon = 32*n_recon*1.16*(pow(10,-10));

    n_process = (int) std::ceil((mem_proj + mem_recon)/mem_gpu);

    // divisão de processos 
    if(n_process < ndev) n_process = ndev;

    //verificar se divisão está ok
    // n_process = 2*n_process;
    if(lab.nx == 2048 && lab.nbeta == 2048) n_process = 8;

    printf("\n \n \n   N_PROCESS =  %d   MEM_PROJ = %Lf \n \n \n ", n_process, mem_proj);

    return n_process;
}}
