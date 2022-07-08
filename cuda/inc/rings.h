#ifndef RAFT_RINGS_H
#define RAFT_RINGS_H

#define PADDING 32


extern "C"{
    void applyrings(int gpu, float* volume, int sizex, int sizey, int sizez, float lambda);

    void RingsEM(float* sinogram, float* ptrflat, size_t sizex, size_t sizey, size_t sizez);

    void applyringsEM(int gpu, float* sinogram, float* ptrflat, int sizex, int sizey, int sizez);

    void ringsblock(int* gpus, int ngpus, float* data, int nrays, int nangles, int nslices, float lambda_rings, int ringblocks);

}


#endif