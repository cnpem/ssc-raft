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
    R2R_R2R = 1,
    none    = 2
};

struct Convolution
{
	Convolution() = default;
	explicit Convolution(int _dim, dim3 _size, dim3 _pad_size, float _pad_value, fftType _type) : 
    dim(_dim), size(_size), pad_size(_pad_size), pad_value(_pad_value), typefft(_type){};

	int dim          = 2;
	dim3 size        = dim3(1, 1, 1);
	dim3 pad_size    = dim3(1, 1, 1);

    float pad_value  = 0.0;

    size_t sizey     = (dim - 1) * pad_size.y - (dim - 2) * size.y;
    size_t batch     = (dim - 1) * size.z - (dim - 2) *(sizey * size.z);
    size_t npad      = (size_t)pad_size.x * sizey * size.z;

    dim3 final_pad_size = dim3(pad_size.x, sizey, batch);

	fftType typefft  = fftType::C2C_C2C;
    
	/* Plan FFTs*/
	cufftHandle mplan, implan;

	Plan(cufftHandle mplan, cufftHandle implan, int dim, dim3 final_pad_size, fftType typefft);
    ~Plan(cufftHandle mplan, cufftHandle implan, fftType typefft);

    template<typename Type1, typename Type2, typename Type3>
	void convolve(GPU gpus, Type1 *input, Type2 *kernel, Type3 *output);

    template<typename TypeIn, typename TypeOut>
	void padding(GPU gpus, TypeIn *in, TypeOut *padded, padType type);

    template<typename TypeIn, typename TypeOut>
	void remove_padding(GPU gpus, TypeIn *inpadded, TypeOut *out, padType type);

    template<typename Type>
    padType setPad(fftType type_fft);
};

#endif
