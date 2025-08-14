#ifndef RAFT_PIPE_H
#define RAFT_PIPE_H

#include "common/configs.hpp"
#include "processing/processing.hpp"


extern "C"{

    void ReconstructionPipelineMultiGPU(CFG configs, int *gpus, int ngpus,
    float *object, float *data, float *flats, float *darks, float *angles);

    void ReconstructionPipelineProcessMultiGPU(CFG configs, int *gpus, int ngpus,
        float *object, float *data, float *flats, float *darks, float *angles);

}

#endif 
