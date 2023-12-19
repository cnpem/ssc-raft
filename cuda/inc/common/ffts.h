#ifndef RAFT_FFT_H
#define RAFT_FFT_H

#include "./common/configs.h"
#include "./common/types.hpp"
#include "./common/operations.hpp"
#include "./common/complex.hpp"
#include "./common/logerror.hpp"

// Auto create 4 real <-> complex transforms sharing the same workarea
// Meant to be used with the FST versions of Radon and BackProjection

enum padType {
    C2C  = 0,
    R2C  = 1,
    C2R  = 2,
    R2R  = 3,
    none = 4
};

enum fftType {
    C2C_C2C = 0,
    C2R_R2C = 1,
    R2C_C2R = 2,
    R2R_R2R = 3,
    none    = 4
};

template<>
padType paddingType<TypeIn,TypeOut>();

template<>
padType paddingType<float,float>();

template<>
padType paddingType<cufftComplex,cufftComplex>();

    template<>
padType paddingType<cufftComplex,float>();

    template<>
padType paddingType<float,cufftComplex>();

struct Convolution
{
	Convolution() = default;
	explicit Convolution(int _dim, dim3 _size, dim3 _pad, float _padval, fftType _type) : 
    dim(_dim), size(_size), padd(_pad), padval(_padval), typefft(_type){};

	int dim          = 2;
	dim3 size        = dim3(1, 1, 1);
	dim3 padd        = dim3(1, 1, 1);

    float padval     = 0.0;

    size_t sizey     = (dim - 1) * padd.y - (dim - 2) * size.y;
    size_t batch     = (dim - 1) * size.z - (dim - 2) *(sizey * size.z);
    size_t npad      = (size_t)padd.x * sizey * size.z;

    dim3 pad         = dim3(padd.x, sizey, batch);

	fftType typefft  = fftType::C2C_C2C;
    
	/* Plan FFTs*/
	cufftHandle mplan, implan;

	Plan(cufftHandle mplan, cufftHandle implan, int dim, dim3 pad, fftType typefft);
    ~Plan(cufftHandle mplan, cufftHandle implan, fftType typefft);

    template<typename Type1, typename Type2, typename Type3>
	void convolve(GPU gpus, Type1 *gpuptr, Type2 *kernel);

    template<typename Type>
	void padd(GPU gpus, Type *in, void *padded, void *ipadded);

    template<typename TypeIn, typename TypeOut>
	void Recpadd(GPU gpus, TypeIn *inpadded, TypeOut *out);

    template<typename TypeIn, typename TypeOut>
    padType setPad();
};

#endif
