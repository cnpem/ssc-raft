#include <stdio.h>
#include <cufft.h>
#include <stdlib.h>
#include "../../../../inc/geometries/gc/fdk.h"
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
    long long int  n_filter_pad, idx_filter_pad;
    int zi_filter_pad; 

    nz_gpu = (int) ceil( (float)lab.nv/n_process);
    zi_min = i*nz_gpu;
    zi_max = (int) std::min((i+1)*nz_gpu, lab.nv);

    n_filter = (long long int) (zi_max - zi_min)*lab.nbeta*lab.nh;
    idx_filter = (long long int) zi_min*lab.nbeta*lab.nh;
    zi_filter = (int) (n_filter/(lab.nbeta*lab.nh));

    n_filter_pad = (long long int) (zi_max - zi_min)*lab.nbeta*lab.nph;
    idx_filter_pad = (long long int) zi_min*lab.nbeta*lab.nph;
    zi_filter_pad = (int) (n_filter_pad/(lab.nbeta*lab.nph));

    printf("n_process = %d, nz = %d, %d \n",n_process,lab.nz,nz_gpu);
    printf("S0 = %d, S1 = %d, zi_max = %d , zi_min = %d \n",lab.slice_recon_start,lab.slice_recon_end,zi_max,zi_min);

    printf("n_process = %d, nph = %d, padh %d \n",n_process,lab.nph,lab.padh);
    printf("Process = %d: n_filter = %lld, idx_filter = %lld, z_filter = %d, nbeta = %d, hbeta = %f \n", i, n_filter, idx_filter, zi_filter, lab.nbeta, lab.dbeta);
    printf("Process = %d: n_filter_pad = %lld, idx_filter_pad = %lld, z_filter_pad = %d \n", i, n_filter_pad, idx_filter_pad, zi_filter_pad);
    
    nz_gpu = (int) ceil( (float)lab.nz/n_process);
    zi_min = i*nz_gpu;
    zi_max = (int) std::min((i+1)*nz_gpu, lab.nz);

    n_recon = (long long int) (zi_max - zi_min)*lab.nx*lab.ny;
    idx_recon = (long long int) zi_min*lab.nx*lab.ny;

    z_min = - lab.z + zi_min*lab.dz;
    z_max = - lab.z + zi_max*lab.dz;
    
    printf("Process = %d:  n_recon  = %lld,  idx_recon = %lld, z_ph = %f \n",  i, n_recon, idx_recon, z_min);
    
    L = sqrt(lab.x*lab.x + lab.y*lab.y);
    // L = std::max(lab.x, lab.y);
    // L = 3 * L;
    
    //variáveis de9tector
   	Z_min = std::max(-lab.v, std::min(  lab.Dsd*z_min/(lab.D - L),  
                                        lab.Dsd*z_min/(lab.D + L) ));
    Z_max = std::min(+lab.v, std::max(  lab.Dsd*z_max/(lab.D + L),
                                        lab.Dsd*z_max/(lab.D - L))); 

    Zi_min = std::max(0, (int) floor((Z_min + lab.v)/lab.dv));
    Zi_max = std::min(lab.nv, (int) ceil((Z_max + lab.v)/lab.dv));

    idx_proj = (long long int) (Zi_min)*(lab.nbeta*lab.nh);
    n_proj = (long long int) (Zi_max-Zi_min)*(lab.nbeta*lab.nh);

    printf("Zimax = %d, %d, %e, Zimin = %d, %d, %e \n",Zi_max,zi_max,z_max,Zi_min,zi_min,z_min);

    (*process).i = i;
    (*process).i_gpu = (int) gpus[i%ndevs];  
    (*process).zi = Zi_min;
    (*process).idx_proj = idx_proj;
    (*process).n_proj = n_proj;

    (*process).n_filter = n_filter;
    (*process).idx_filter = idx_filter;
    (*process).z_filter = zi_filter; 

    (*process).n_filter_pad = n_filter_pad;
    (*process).idx_filter_pad = idx_filter_pad;
    (*process).z_filter_pad = zi_filter_pad; 

    (*process).n_recon = n_recon;
    (*process).idx_recon = idx_recon;
    (*process).z_ph = z_min;
    (*process).z_det = - lab.v + Zi_min*lab.dv;

    printf("Process = %lld:  n_proj = %lld,   idx_proj = %lld, Z_det = %f, z_det = %f, Zimin = %d, Zimax = %d\n\n",   (*process).i, (*process).n_proj, (*process).idx_proj, Z_min, (*process).z_det, (*process).zi, Zi_max);

}}

extern "C"{
void set_process_slices(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs){   

    long long int  n_filter, idx_filter, n_recon, idx_recon, n_proj, idx_proj;
    float z_min, z_max, Z_min, Z_max, L;  
    int nz_gpu,  zi_min, zi_max, Zi_max, Zi_min, zi_filter, zi_filter_pad; 
    int zi_min0, zi_max0, Zi_max0, Zi_min0; 
    float z_min0, z_max0, Z_max0, Z_min0; 
    int zzi_min, zzi_max;
    long long int  n_filter_pad, idx_filter_pad;

    // Reconstructed object 
    nz_gpu = (int) ceil( (float) ( lab.slice_recon_end - lab.slice_recon_start )/n_process); 
    zi_min = lab.slice_recon_start + i*nz_gpu;
    zi_max = (int) std::min(lab.slice_recon_start + (i+1)*nz_gpu, lab.slice_recon_end); 

    printf("n_process = %d, nz = %d, %d \n",n_process,lab.nz,nz_gpu);
    printf("S0 = %d, S1 = %d, zi_max = %d , zi_min = %d \n",lab.slice_recon_start,lab.slice_recon_end,zi_max,zi_min);

    n_recon = (long long int) (zi_max - zi_min)*lab.nx*lab.ny;
    idx_recon = (long long int) ( zi_min - lab.slice_recon_start ) * lab.nx * lab.ny;

    z_min = - lab.z + zi_min*lab.dz;
    z_max = - lab.z + zi_max*lab.dz;
    
    printf("Process = %d:  n_recon  = %lld,  idx_recon = %lld, z_ph = %f \n",  i, n_recon, idx_recon, z_min);
    
    L = sqrt(lab.x*lab.x + lab.y*lab.y);
    
    //variáveis dectector

   	Z_min = std::max(-lab.v, std::min(  lab.Dsd*z_min/(lab.D - L),  
                                        lab.Dsd*z_min/(lab.D + L) ));
    Z_max = std::min(+lab.v, std::max(  lab.Dsd*z_max/(lab.D + L),
                                        lab.Dsd*z_max/(lab.D - L))); 

    Zi_min = std::max(0, (int) floor((Z_min + lab.v)/lab.dv));
    Zi_max = std::min(lab.nv, (int) ceil((Z_max + lab.v)/lab.dv));

    printf("Z0,Zimax = %d, %d; Z1,Zimin = %d, %d \n",lab.slice_tomo_end,Zi_max,lab.slice_tomo_start,Zi_min);

    idx_proj = (long long int) (Zi_min - lab.slice_tomo_start)*(lab.nbeta*lab.nh);
    n_proj   = (long long int) (Zi_max-Zi_min)*(lab.nbeta*lab.nh);

    // Filter indexes:
    nz_gpu = (int) ceil( (float)(lab.nv)/n_process); 
    zzi_min = i*nz_gpu;
    zzi_max = (int) std::min((i+1)*nz_gpu, lab.nv); 

    n_filter       = (long long int) (zzi_max - zzi_min)*lab.nbeta*lab.nh;
    n_filter_pad   = (long long int) (zzi_max - zzi_min)*lab.nbeta*lab.nph;
    idx_filter     = (long long int) zzi_min*lab.nbeta*lab.nh;
    idx_filter_pad = (long long int) zzi_min*lab.nbeta*lab.nph;
    zi_filter      = (int) (n_filter/(lab.nbeta*lab.nh ));
    zi_filter_pad  = (int) (n_filter_pad/(lab.nbeta*lab.nph));

    printf("n_process = %d, nv = %d, %d \n",n_process,lab.nv,nz_gpu);
    printf("Process = %d: n_filter = %lld, idx_filter = %lld, z_filter = %d, nbeta = %d, hbeta = %f \n", i, n_filter, idx_filter, zi_filter, lab.nbeta, lab.dbeta);
    
    (*process).i = i;
    (*process).i_gpu = (int) gpus[i%ndevs];  
    (*process).zi = Zi_min;
    (*process).idx_proj = idx_proj;
    (*process).n_proj = n_proj;

    (*process).n_filter = n_filter;
    (*process).idx_filter = idx_filter;
    (*process).z_filter = zi_filter; 
    (*process).n_filter_pad = n_filter_pad;
    (*process).idx_filter_pad = idx_filter_pad;
    (*process).z_filter_pad = zi_filter_pad; 

    (*process).n_recon = n_recon;
    (*process).idx_recon = idx_recon;
    (*process).z_ph = z_min;
    (*process).z_det = - lab.v + Zi_min*lab.dv;

    printf("Process = %lld:  n_proj = %lld,   idx_proj = %lld, Z_det = %f, z_det = %f, zi = %d\n\n",   (*process).i, (*process).n_proj, (*process).idx_proj, Z_min, (*process).z_det, (*process).zi);
    
}}

extern "C"{
void set_process_slices_2(Lab lab, int i, Process* process, int n_process, int* gpus, int ndevs){   

    long long int  n_filter, idx_filter, n_recon, idx_recon, n_proj, idx_proj, idx_proj_max;
    float z_min, z_max, Z_min, Z_max, L;  
    int nz_gpu,  zi_min, zi_max, Zi_max, Zi_min, zi_filter; 
    int zi_min0, zi_max0, Zi_max0, Zi_min0; 
    float z_min0, z_max0, Z_max0, Z_min0; 
    int zzi_min, zzi_max;
    long long int  n_filter_pad, idx_filter_pad;
    int zi_filter_pad;

    // Reconstructed object 
    printf("block = %d, n_z = %e, vzz = %d \n",( lab.slice_recon_end - lab.slice_recon_start ), (float)( lab.slice_recon_end - lab.slice_recon_start ) / n_process, (int)ceil( (float)( lab.slice_recon_end - lab.slice_recon_start ) / n_process ) );

    nz_gpu = (int) ceil( (float)( lab.slice_recon_end - lab.slice_recon_start ) / n_process ); 

    zi_min = lab.slice_recon_start + i*nz_gpu;
    zi_max = (int) std::min(lab.slice_recon_start + (i+1)*nz_gpu, lab.slice_recon_end); 

    printf("n_process = %d, nv = %d, %d \n",n_process,lab.nv,nz_gpu);
    printf("S0 = %d, S1 = %d, zi_max = %d , zi_min = %d \n",lab.slice_recon_start,lab.slice_recon_end,zi_max,zi_min);
    printf("zimax - zimin = %d, nx = %d, ny = %d , multi = %d \n",(zi_max - zi_min),lab.nx,lab.ny,(zi_max - zi_min)*lab.nx*lab.ny);

    n_recon = (long long int) (zi_max - zi_min)*lab.nx*lab.ny;
    idx_recon = (long long int) ( zi_min - lab.slice_recon_start ) * lab.nx * lab.ny;

    z_min = - lab.z + zi_min*lab.dz;
    z_max = - lab.z + zi_max*lab.dz;
    
    printf("Process = %d:  n_recon  = %lld,  idx_recon = %lld, %d, z_ph = %f \n",  i, n_recon, idx_recon, ( zi_min - lab.slice_recon_start ), z_min);
    
    L = sqrt(lab.x*lab.x + lab.y*lab.y);
    // L = std::max(lab.x, lab.y);
    
    //variáveis dectector
    // For slices ---- Paola 11 set 2023:
    z_min0 = - lab.z + lab.slice_recon_start*lab.dz;
    z_max0 = - lab.z + lab.slice_recon_end*lab.dz;

    Z_min0 = std::max(-lab.v, std::min( lab.Dsd*z_min0/(lab.D - L),  
                                        lab.Dsd*z_min0/(lab.D + L)));
    Z_max0 = std::min(+lab.v, std::max( lab.Dsd*z_max0/(lab.D + L),
                                        lab.Dsd*z_max0/(lab.D - L))); 

    Zi_min0 = std::max(0, (int) floor((Z_min0 + lab.v)/lab.dv));
    Zi_max0 = std::min(lab.nv, (int) ceil((Z_max0 + lab.v)/lab.dv));

    printf("Zimax0 = %d, %e, Zimin0 = %d, %e \n",Zi_max0,z_max0,Zi_min0,z_min0);
    // -------------------------------

   	Z_min = std::max(-lab.v, std::min(  lab.Dsd*z_min/(lab.D - L),  
                                        lab.Dsd*z_min/(lab.D + L) ));
    Z_max = std::min(+lab.v, std::max(  lab.Dsd*z_max/(lab.D + L),
                                        lab.Dsd*z_max/(lab.D - L))); 

    Zi_min = std::max(0, (int) floor((Z_min + lab.v)/lab.dv));
    Zi_max = std::min(lab.nv, (int) ceil((Z_max + lab.v)/lab.dv));

    printf("Zimax = %d, %d, %e, Zimin = %d, %d, %e \n",Zi_max,zi_max,z_max,Zi_min,zi_min,z_min);

    idx_proj = (long long int) (Zi_min)*(lab.nbeta*lab.nh);
    idx_proj_max = (long long int) (Zi_max)*(lab.nbeta*lab.nh);
    n_proj = (long long int) (Zi_max-Zi_min)*(lab.nbeta*lab.nh);

    // Filter indexes:
    nz_gpu = (int) ceil( (float)(Zi_max0 - Zi_min0) / n_process ); 
    zzi_min = Zi_min0 + i*nz_gpu;
    zzi_max = (int) std::min(Zi_min0 + (i+1)*nz_gpu, Zi_max0); // lab.blockv

    n_filter       = (long long int) (zzi_max - zzi_min)*lab.nbeta*lab.nh;
    idx_filter     = (long long int) zzi_min*lab.nbeta*lab.nh;
    zi_filter      = (int) (n_filter/(lab.nbeta*lab.nh ));

    n_filter_pad   = (long long int) (zzi_max - zzi_min)*lab.nbeta*lab.nph;
    idx_filter_pad = (long long int) zzi_min*lab.nbeta*lab.nph;
    zi_filter_pad  = (int) (n_filter_pad/(lab.nbeta*lab.nph));

    printf("n_process = %d, nph = %d, padh %d \n",n_process,lab.nph,lab.padh);
    printf("Process = %d: n_filter = %lld, idx_filter = %lld, z_filter = %d, nbeta = %d, hbeta = %f \n", i, n_filter, idx_filter, zi_filter, lab.nbeta, lab.dbeta);
    printf("Process = %d: n_filter_pad = %lld, idx_filter_pad = %lld, z_filter_pad = %d \n", i, n_filter_pad, idx_filter_pad, zi_filter_pad);

    (*process).i = i;
    (*process).i_gpu = (int) gpus[i%ndevs];  
    (*process).zi = Zi_min;
    (*process).idx_proj_max = (Zi_max-Zi_min);
    (*process).idx_proj = idx_proj;
    (*process).n_proj = n_proj;

    (*process).n_filter = n_filter;
    (*process).idx_filter = idx_filter;
    (*process).z_filter = zi_filter; 

    (*process).n_filter_pad = n_filter_pad;
    (*process).idx_filter_pad = idx_filter_pad;
    (*process).z_filter_pad = zi_filter_pad; 

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
    
    int block = ( lab.slice_recon_end - lab.slice_recon_start );

    // n_proj = (long long int)(lab.nv)*(long long int)(lab.nbeta)*(long long int)(lab.nh); 
    n_proj = (long long int)(block)*(long long int)(lab.nbeta)*(long long int)(lab.nph);

    n_recon = (long long int)(lab.nx)*(long long int)(lab.ny)*(long long int)(lab.nz);

    mem_gpu = 40;
    mem_proj = 32*n_proj*1.16*(pow(10,-10));
    mem_recon = 32*n_recon*1.16*(pow(10,-10));

    if ( block <= ndev ){

        n_process = ndev;

    }else{

        n_process = (int) std::ceil((mem_proj + mem_recon)/mem_gpu);

        printf("Number of procs = %d, block = %d \n",n_process,block);

        if ( n_process > block ){

            float blocksize = (float)n_process / block;

            if ( blocksize <= 0 )

                n_process = 1;

        }else{

            if ( block > 16 ){

                // n_process = 2*n_process;
                if(lab.nx > 1024 && lab.nbeta > 1024) n_process = 8;
                if(lab.nx > 2048 && lab.nbeta > 2048) n_process = 16;
                if(lab.nph >= 4096) n_process = 16;

                //if(lab.nbeta >= 1900) n_process = 16;
                // if (lab.nbeta >= 3900) n_process = 32;
                if (lab.nh >= 4096 || lab.nbeta >= 4096) n_process = 32;
                if (lab.nh >= 8000 || lab.nbeta >= 8000) n_process = 64;

            
            } 
        }
    }

    printf("\n \n \n   N_PROCESS =  %d   MEM_PROJ = %Lf \n \n \n ", n_process, mem_proj);


    return n_process;
}}
