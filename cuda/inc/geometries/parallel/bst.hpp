#ifndef RAFT_BST_PAR_H
#define RAFT_BST_PAR_H

#include "common/configs.hpp"
#include "common/operations.hpp"
#include "common/types.hpp"

typedef struct workspaceBST
{	
	cufftHandle plan1d, plan2d, filterplan;
	cImage *filtersino, *cartesianblock, *polarblock, *realpolar;
}WBST;

extern "C" {

    WBST *InitializeBST_workspace(dim3 tomo_size, dim3 obj_size, int bst_padd, int blocksize_bst);
    void freeBSTWorkspace(WBST *workspace);

    void getBST(float* blockRecon, float* wholesinoblock, float* angles, 
        int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0, 
        float reg, float paganin, int filter_type, float offset, float pixel,
        WBST *bst_workspace, cudaStream_t stream); 

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
