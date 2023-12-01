#ifndef RAFT_RADON_PAR_H
#define RAFT_RADON_PAR_H


extern "C"{

	void GRadon(int device, float* _frames, float* _image, int nrays, int nangles, int blocksize);

    __global__ void KRadon_RT(float* restrict frames, const float* image, int nrays, int nangles);


}


#endif