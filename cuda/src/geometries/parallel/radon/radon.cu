// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "../../../../inc/sscraft.h"

extern "C"{

    __global__ void KRadon_RT(float* restrict frames, const float* image, int nrays, int nangles)
    {
        int ray = (blockDim.x * blockIdx.x + threadIdx.x);
        int ang = (blockDim.y * blockIdx.y + threadIdx.y);
        const size_t sizef = nrays*nrays;
        
        if ( ray>=nrays || ang >= nangles )
            return;
        
        float btheta = float(M_PI)*float(ang + nangles/2)/nangles;
        
        float ct = nrays/2;
        float cos_t = cosf(btheta + 1E-5f);
        float sin_t = sinf(btheta + 1E-5f);

        float x = ct - 2.0f*ct*cos_t + (ray-nrays/2)*sin_t;
        float y = ct - 2.0f*ct*sin_t - (ray-nrays/2)*cos_t;

        float tx0 = -x/cos_t;
        float tx1 = (nrays-x)/cos_t;

        float ty0 = -y/sin_t;
        float ty1 = (nrays-y)/sin_t;

        float d1 = fmaxf(fminf(tx0,tx1), fminf(ty0,ty1));
        int d2 = int(ceil(fminf(fmaxf(tx0,tx1), fmaxf(ty0,ty1)) - d1)+0.1f);

        x += d1*cos_t;
        y += d1*sin_t;
        
        float radon = 0;
        for(int s=0; s<d2; s++)
        {
            radon += image[(int(y+0.5f)*nrays + int(x+0.5f))%sizef + blockIdx.z*sizef];

            x += cos_t;
            y += sin_t;
        }

        frames[blockIdx.z*size_t(nrays*nangles) + nrays*ang + ray] = radon;
    }

}

extern "C"
{
	void GRadon(int device, float* _frames, float* _image, int nrays, int nangles, int blocksize)
	{
		cudaSetDevice(device);

		rImage frames(nrays,nangles,blocksize);
		rImage image(_image,nrays,nrays,blocksize);

		KRadon_RT<<<dim3(nrays/64,nangles,blocksize),64>>>(frames.gpuptr, image.gpuptr, nrays, nangles);

		frames.CopyTo(_frames);
		cudaDeviceSynchronize();
	}

}