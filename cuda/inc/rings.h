#ifndef RAFT_RINGS_H
#define RAFT_RINGS_H

#define PADDING 32


extern "C"{
    void RingsEM(float* sinogram, float* ptrflat, size_t sizex, size_t sizey, size_t sizez);
}


#endif