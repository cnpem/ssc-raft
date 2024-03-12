#ifndef RAFT_BST_PAR_H
#define RAFT_BST_PAR_H

#include "common/configs.hpp"
#include "common/operations.hpp"
#include "common/types.hpp"

extern "C" {

    void setBSTParameters(CFG *configs, float *parameters_float, int *parameters_int);
    void printBSTParameters(CFG *configs);

    void EMFQ_BST(float* blockRecon, float *wholesinoblock, float *angles,
    int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0);

    void EMFQ_BST_ITER(
	float* blockRecon, float *wholesinoblock, float *angles,
	cImage& cartesianblock, cImage& polarblock, cImage& realpolar,
	cufftHandle plan1d, cufftHandle plan2d,
	int Nrays, int Nangles, int trueblocksize, int blocksize, int sizeimage, 
    int pad0);

}
#endif
