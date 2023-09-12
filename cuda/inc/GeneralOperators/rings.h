#ifndef RAFT_RINGS_H
#define RAFT_RINGS_H

#define PADDING 32


extern "C"{
    void RingsEM(float* sinogram, float* ptrflat, size_t sizex, size_t sizey, size_t sizez);

    void Rings(float* volume, int vsizex, int vsizey, int vsizez, float lambda, size_t slicesize);

    void applyringsEM(int gpu, float* sinogram, float* ptrflat, int sizex, int sizey, int sizez);

    void ringsblock(int* gpus, int ngpus, float* data, int nrays, int nangles, int nslices, float lambda_rings, int ringblocks);

}


#endif