#ifndef RAFT_PIPE_H
#define RAFT_PIPE_H

#include "./common/configs.h"

extern "C"{

    void ReconstructionPipeline(float *recon, float *data, float *flats, float *darks, float *angles,
        float *parameters_float, int *parameters_int, int *flags,
        int *gpus, int ngpus);

    void _setReconstructionPipeline(CFG configs, Process *process, GPU gpus,
        float *recon, float *data, 
        float *flats, float *darks, float *angles,
        int total_number_of_processes);

	void _ReconstructionProcessPipeline(CFG configs, Process process, GPU gpus, 
    float *recon, float *frames, float *flats, float *darks, float *angles);

    void _ReconstructionPipeline(CFG configs, WKP *workspace, Process process, GPU gpus);

    void getReconstructionParallel(CFG configs, Process process, GPU gpus,  WKP *workspace);
    void getReconstructionConebeam(CFG configs, Process process, GPU gpus,  WKP *workspace);
    // void getReconstructionFanbeam(CFG configs, Process process, GPU gpus,  WKP *workspace);

}

#endif 