// in case it is compiled with host compiler instead of nvcc:

#ifndef BST_H
#define BST_H


extern "C" {
    void BST(float* blockRecon, float *wholesinoblock, int Nrays, int Nangles, int trueblocksize, int sizeimage, int pad0);
}

#endif