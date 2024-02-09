#include "common/opt.hpp"
#include "processing/filters.hpp"

extern "C"{
	
    __global__ void fbp_filter_kernel(Filter filter, cufftComplex *kernel, dim3 size)
	{
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        
		if( (i >= size.x)) return;

		float dt   =        2.0f / (float)size.x;
		float wMax =        1.0f / ( 2.0f * dt );
		float dw   = 2.0f * wMax / (float)size.x;

		/* Reciprocal grid */ 
        float w = - wMax + i * dw;

		w = filter.apply( w );
	
		kernel[i] = exp1j(- 2.0f * float(M_PI) * filter.axis_offset * w ) * w ;
	}

	void filterFBP(GPU gpus, Filter filter, 
    float *tomogram, cufftComplex *filter_kernel, 
    dim3 size, dim3 size_pad)
	{	
        /* int dim = { 1, 2 }
            1: if plan 1D multiples cuffts
            2: if plan 2D multiples cuffts */
        int dim = 1; 

        dim3 pad((int)((size_pad.x - size.x) / 2 ), 0, 0);
        dim3 size_kernel(size_pad.x, 1, 1);

        opt::MPlanFFT(gpus.mplan, dim, size_pad);

		fbp_filter_kernel<<<gpus.Grd.x,gpus.BT.x>>>(filter, filter_kernel, size_pad);

		convolution_R_C2C(  gpus, tomogram, filter_kernel, 
                            size, 
                            pad, 
                            size_kernel, 
                            0.0f, dim);

		cufftDestroy(gpus.mplan1);
	}
}

extern "C" {

	void convolution_Real_C2C(GPU gpus, 
    float *data, cufftComplex *kernel, 
    dim3 size, dim3 kernel_size, 
    dim3 pad, float pad_value, int dim)
	{
        size_t npad   = (size_t)(size.x + 2 * pad.x) * (size.y + (dim - 1) * 2 * pad.y) * size.z;

        cufftComplex *dataPadded = opt::allocGPU<cufftComplex>(npad);

        opt::paddR2C<<<gpus.Grd,gpus.BT>>>(data, dataPadded, size, pad, pad_value);
        
        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_FORWARD));
        
        opt::product_Complex_Complex<<<gpus.Grd,gpus.BT>>>(dataPadded, kernel, dataPadded, size, kernel_size, pad);	
        
        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_INVERSE));
        
        opt::remove_paddC2R<<<gpus.Grd,gpus.BT>>>(dataPadded, data, size, pad, dim);

		cudaFree(dataPadded);
	}

    __global__ void fftshiftKernel(float *c, dim3 size)
    {
        int shift;
        int N = ( (size.x * size.y) + size.x ) / 2 ;	
        int M = ( (size.x * size.y) - size.x ) / 2 ;	
        float temp;
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        int index; 

        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;
        
        if ( i < ( size.x / 2 ) ){	
            if ( j < ( size.y / 2 ) ){	
                index = size.x * (k*size.y + j)  + i;
                shift = index + N;
                temp 	 = c[index];	
                c[index] = c[shift];	
                c[shift] = temp;
            }
        }else{
            if ( j < ( size.y / 2 ) ){
                index = size.x * (k*size.y + j)  + i;
                shift = index + M;
                temp 	 = c[index];	
                c[index] = c[shift];	
                c[shift] = temp;
            }
        }
    }

    __global__ void Normalize(float *a, float b, dim3 size)
    {
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        
        size_t index = size.x * j + i;
        
        if ( (i >= size.x) || (j >= size.y) || (k >= 1) ) return;
        
        a[index] = a[index] / b; 
    }

	void SinoFilter(float* sino, size_t nrays, size_t nangles, size_t blocksize, int csino, bool bRampFilter, Filter reg, bool bShiftCenter, float* sintable)
	{	
		cImage fft(nrays/2+1,nangles);
		// cImage fft2(nrays/2+1,nangles);

		// printf("FILTER: %ld %ld %ld %ld \n",nrays,nangles,blocksize,nrays/2+1);

		cufftHandle plan_r2c, plan_c2r;
		cufftPlan1d(&plan_r2c, nrays, CUFFT_R2C, nangles);
		cufftPlan1d(&plan_c2r, nrays, CUFFT_C2R, nangles);
		
		dim3 blk = fft.ShapeBlock();
		dim3 thr = fft.ShapeThread();

		// printf("Enter sino filter \n ");

		for(int k=0; k<blocksize; k++)
		{
			HANDLE_FFTERROR( cufftExecR2C(plan_r2c, sino+k*nrays*nangles, fft.gpuptr) );

			if(bRampFilter)
				BandFilterReg<<<blk,thr>>>(fft.gpuptr, nrays/2+1, csino, bShiftCenter, sintable, reg);
			else
				std::cout << __FILE__ << " " << __LINE__ << " " << "Auto reg missing!" << std::endl;

			HANDLE_FFTERROR( cufftExecC2R(plan_c2r, fft.gpuptr, sino+k*nrays*nangles) );
		}
		
		cufftDestroy(plan_r2c);
		cufftDestroy(plan_c2r);
	}

	__global__ void BandFilterReg(complex* vec, size_t sizex, int icenter, bool bShiftCenter, float* sintable, Filter mfilter)
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

		float rampfilter = float(tx) / (float)sizex;
		rampfilter = mfilter.apply(rampfilter);
		// printf("band filter value: %lf \n ",rampfilter);

		float fcenter = 1.0f - (bShiftCenter ? (sintable[ty]) : 0);
		fcenter = -2*float(M_PI)/float(2*sizex-2) * fcenter * icenter;

		if(tx < sizex)
			vec[ty*sizex + tx] *= exp1j(fcenter * tx) * rampfilter;
	}

	void Highpass(rImage& x, float wid)
	{
		size_t sizex = x.sizex;
		size_t sizey = x.sizey;

		cImage fourier(sizex/2+1, sizey);
		cufftHandle planrc;
		cufftHandle plancr;

		// Optimize allocation
		cufftPlan1d(&planrc, sizex, CUFFT_R2C, sizey);
		cufftPlan1d(&plancr, sizex, CUFFT_C2R, sizey);

		for(size_t bz=0; bz<x.sizez; bz++)
		{
			cufftExecR2C(planrc, x.gpuptr + sizex*sizey*bz, fourier.gpuptr);
			KFilter<<<dim3((sizex/2+32)/32,sizey),32>>>(fourier.gpuptr, sizex, wid);
			cufftExecC2R(plancr, fourier.gpuptr, x.gpuptr + sizex*sizey*bz);
		}

		cufftDestroy(planrc);
		cufftDestroy(plancr);
	}

	__global__ void KFilter(complex* x, size_t sizex, float wid)
	{
		const size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

		if(idx > sizex/2)
			return;

		float xs = float(idx)*wid/sizex;
		x[blockIdx.y*(sizex/2+1) + idx] *= (1.0f - expf(-20.0f*xs*xs))/sizex;
	}

	

	__device__ complex DeltaFilter(complex* img, int sizeimage, float fx, float fy)
	{
		fx = fminf(fx, sizeimage/2-1E-4f);
		int ix = int(fx);
		int iy = int(fy);

		float a = fx-ix;
		float b = fy-iy;

		const int h2n = sizeimage/2+1;

		return  img[(iy%sizeimage)*h2n + ix]*(1-a)*(1-b) +
				img[((iy+1)%sizeimage)*h2n + ix]*(1-a)*b +
				img[(iy%sizeimage)*h2n + (ix+1)]*a*(1-b) +
				img[((iy+1)%sizeimage)*h2n + (ix+1)]*a*b;
	}
	

	__global__ void BandFilterC2C(complex* vec, size_t sizex, int center, Filter mfilter = Filter())
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

		float rampfilter = 2.0f*fminf(tx,sizex-tx)/(float)sizex;
		rampfilter = mfilter.apply(rampfilter);

		if(tx < sizex)
			vec[ty*sizex + tx] *= exp1j(-2*float(M_PI)/float(sizex) * center * tx) * rampfilter;
	}

}

__host__ __device__ inline float Filter::apply(float input)
{
	float param = 0.0f;

	if (type == EType::gaussian)
	{
		input *= exp(-0.693f * reg * input * input);
		input /= (1.0f + paganin * input * input);
	}
	else if (type == EType::lorentz)
	{
		input /= 1.0f + reg * input * input;
		input /= (1.0f + paganin * input * input);
	}
	else if (type == EType::cosine)
	{
		input *= cosf(float(M_PI) * 0.5f * input);
		input /= (1.0f + paganin * input * input);
	}
	else if (type == EType::rectangle)
	{
		param = fmaxf(input * reg * float(M_PI) * 0.5f, 1E-4f);
		input *= sinf(param) / param;
		input /= (1.0f + paganin * input * input);
	}
	else if (type == EType::hann)
	{
		input *= 0.5f + 0.5f * cosf(2.0f * float(M_PI) * input);
		input /= (1.0f + paganin * input * input);
	}
	else if (type == EType::hamming)
	{
		input *= (0.54f + 0.46f * cosf(2.0f * float(M_PI) * input));
		input /= (1.0f + paganin * input * input);
	}
	else if (type == EType::ramp)
	{
		input /= (1.0f + paganin * input * input);
	}

	return input;
}

