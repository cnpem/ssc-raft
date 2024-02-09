#ifndef RAFT_BST_PAR_H
#define RAFT_BST_PAR_H

#include "common/configs.hpp"

extern "C" {

    void BST(float *blockRecon, float *wholesinoblock, int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0);

    void BST_core(
        float *blockRecon, float *wholesinoblock,
        cImage& cartesianblock, cImage& polarblock, cImage& realpolar,
        cufftHandle plan1d, cufftHandle plan2d,
        int Nrays, int Nangles, int trueblocksize, int blocksize, int sizeimage, int pad0);

}
#endif
