#ifndef RAFT_PIPE_H
#define RAFT_PIPE_H

#include "common/configs.hpp"
#include "processing/processing.hpp"


extern "C"{

    void ReconstructionPipeline_GPU(CFG configs,
        float *object, float *data, float *flats, float *darks, float *angles, 
        int gpu_device);

}

#endif 
