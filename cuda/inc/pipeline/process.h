#ifndef RAFT_PROC_H
#define RAFT_PROC_H

#include "../common/config.h"

typedef struct Processes{ 
    /* GPU */ 
    int index, index_gpu;

    /* Processes*/
    int batch_size_tomo, batch_size_recon, batch_index;

    /* Tomogram (or detector) and reconstruction filter (v of vertical) */
    int indv, indv_filter;
    long long int ind_tomo, n_tomo, ind_filter, n_filter;

    /* Reconstruction */
    long long int ind_recon, n_recon;
    float z, z_det;

    int i, i_gpu, zi, z_filter, z_filter_pad;
    long long int n_proj, n_filter_pad;
    long long int idx_proj, idx_proj_max, idx_recon, idx_filter, idx_filter_pad;
    float z_ph;

}Process;

extern "C"{

	Process *setProcesses(CFG configs, GPU gpus, int total_number_of_processes);

    void setProcessParallel(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);

    void setProcessConebeam(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);

    void setProcessFrames(CFG configs, Process* process, GPU gpus, int index, int n_total_processes);
    
    int getTotalProcesses(CFG configs, GPU gpus);

}

#endif 