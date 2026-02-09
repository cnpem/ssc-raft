#ifndef RAFT_PIPE_H
#define RAFT_PIPE_H

#include "common/configs.hpp"
#include "processing/processing.hpp"


extern "C"{

    void ReconstructionPipelineMultiGPU(CFG configs, int *gpus, int ngpus,
    float *object, float *data, float *flats, float *darks, float *angles);

    // void ReconstructionPipelineProcessMultiGPU(CFG configs, int *gpus, int ngpus,
    //     float *object, float *data, float *flats, float *darks, float *angles);

    void getReconstructionMethods(CFG configs, WKP *workspace, int tomoblock, int objblock, cudaStream_t nstream);

    void getRingsMethods(CFG configs, WKP *workspace, int tomoblock, cudaStream_t nstream);

    void getAlignMethods(CFG configs, WKP *workspace, int tomoblock, cudaStream_t nstream);

    void getBackgroundCorrectionMethods(CFG configs, WKP *workspace, int tomoblock, cudaStream_t nstream);

}

#endif 
