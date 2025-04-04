#ifndef RAFT_PIPE_H
#define RAFT_PIPE_H

#include "common/configs.hpp"

extern "C"{

    void ReconstructionPipeline(float *recon, float *data, float *flats, float *darks, float *angles,
        float *parameters_float, int *parameters_int, int *flags,
        int *gpus, int ngpus);

    void _setReconstructionPipeline(CFG *configs, Process *process, GPU gpus,
        float *recon, float *data, 
        float *flats, float *darks, float *angles,
        int total_number_of_processes);

	void _ReconstructionProcessPipeline(CFG configs, Process process, GPU gpus, 
    float *recon, float *frames, float *flats, float *darks, float *angles);

    void _ReconstructionPipeline(CFG configs, WKP *workspace, GPU gpus);

    void getReconstructionParallel(CFG configs, GPU gpus,  WKP *workspace);
    void getReconstructionConebeam(CFG configs, GPU gpus,  WKP *workspace);
    void getReconstructionMethods(CFG configs, GPU gpus, WKP *workspace);

}

#endif 
