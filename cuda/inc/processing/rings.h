#ifndef RAFT_RINGS_H
#define RAFT_RINGS_H

#define PADDING 32


extern "C"{
    
    void getRingsMultiGPU(int *gpus, int ngpus, float *data, float *lambda_computed, int nrays, int nangles, int nslices, float lambda_rings, int ring_blocks);

    void getRingsGPU(GPU gpus, int gpu, float *data, float *lambda_computed, dim3 size, float lambda_rings, int ring_blocks);

    float getRings(float *tomogram, dim3 size, float lambda_rings, int ring_blocks, GPU gpus);

}


#endif