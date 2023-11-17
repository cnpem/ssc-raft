#ifndef RAFT_RINGS_H
#define RAFT_RINGS_H

#define PADDING 32


extern "C"{
    
    float Rings(float* volume, int vsizex, int vsizey, int vsizez, float lambda, size_t slicesize);

    void ringsblock(int* gpus, int ngpus, float* data, int nrays, int nangles, int nslices, float lambda_rings, int ringblocks);

    void getRings(float* tomogram, int nrays, int nangles, int nslices, float lambda_rings, int ringblocks);
}


#endif