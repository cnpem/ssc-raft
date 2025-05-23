#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <cstddef>
#include <cstdio>
#include "common/opt.hpp"
#include "configs.hpp"
#include "processing/filters.hpp"
#include "common/complex.hpp"

extern "C"{
	
__global__ void fbp_filtering_C2C(Filter filter, 
    complex *kernel, dim3 size, float pixel)
	{
        size_t i  = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j  = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k  = blockIdx.z*blockDim.z + threadIdx.z;

        size_t index = IND(i,j,k,size.x,size.y);
        
        if ( i >= size.x || j >= size.y || k >= size.z) return;

        float w = fminf( i, size.x - i ) / (float)size.x;

        float expoent = 2.0f * float(M_PI)/(float)( 2 * size.x - 2) * filter.axis_offset * i;

        w = filter.apply( w );

        kernel[index] *= exp1j(- expoent ) * w;
        
	}

    __global__ void fbp_filtering_R2C2R(Filter filter,
    complex *kernel, dim3 size)
	{
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;

        size_t index = IND(i,j,k,size.x,size.y);
        
        if ( i >= size.x  || (j >= size.y) ) return;

        float w =  0.5f * fminf( i, size.x - i ) / ((float)size.x);

        float expoent = 2.0f * float(M_PI)/(float)( 2 * size.x - 2) * filter.axis_offset * i;

        w = filter.apply( w );

        kernel[index] *= exp1j(- expoent ) * w;
	}

    void convolution_Real_C2C_1D(GPU gpus, cufftComplex *data, 
    dim3 size, Filter filter, float pixel)
	{
        dim3 threadsPerBlock(TPBX, TPBY, 1);
        dim3 gridBlock( (int)ceil( size.x / threadsPerBlock.x ) + 1, 
                        (int)ceil( size.y / threadsPerBlock.y ) + 1, 
                        1);

        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, data, data, CUFFT_FORWARD));
                
        fbp_filtering_C2C<<<gridBlock,threadsPerBlock>>>(filter, (complex*)data, size, pixel);

        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, data, data, CUFFT_INVERSE));
	}

    void convolution_R2C_C2R_1D(GPU gpus, float *data, 
    dim3 size, Filter filter)
	{
        size_t nfft = opt::get_total_points(size);

        cufftComplex *fft = opt::allocGPU<cufftComplex>(nfft);

        dim3 threadsPerBlock(TPBX,TPBY,1);
        dim3 gridBlock( (int)ceil( size.x / threadsPerBlock.x ) + 1, 
                        (int)ceil( size.y / threadsPerBlock.y ) + 1, 
                        1);
              
        HANDLE_FFTERROR(cufftExecR2C(gpus.mplan, data, fft));
                
        fbp_filtering_R2C2R<<<gridBlock,threadsPerBlock>>>(filter, (complex*)fft, size);

        HANDLE_FFTERROR(cufftExecC2R(gpus.mplanI, fft, data));

        HANDLE_ERROR(cudaFree(fft));
	}

    void filterFBP_Complex(GPU gpus, Filter filter, 
    float *tomogram, dim3 size, dim3 size_pad, dim3 pad, float pixel)
	{	
        /* int dim = { 1, 2 }
            1: if plan 1D multiples cuffts
            2: if plan 2D multiples cuffts */
        int dim = 1;

        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock( (int)ceil( size_pad.x / TPBX ) + 1,
                        (int)ceil( size_pad.y / TPBY ) + 1,
                        (int)ceil( size_pad.z / TPBZ ) + 1);

        opt::MPlanFFT(&gpus.mplan, dim, size_pad, CUFFT_C2C);

        size_t npad = opt::get_total_points(size_pad);

        cufftComplex *dataPadded = opt::allocGPU<cufftComplex>(npad);

        opt::paddR2C<<<gridBlock,gpus.BT>>>(tomogram, dataPadded, size, pad);

		convolution_Real_C2C_1D(gpus, dataPadded, size_pad, filter, pixel);

        opt::remove_paddC2R<<<gridBlock,gpus.BT>>>(dataPadded, tomogram, size, pad);

        opt::scale<<<gridBlock,gpus.BT>>>(tomogram, size, (float)size_pad.x);

        HANDLE_ERROR(cudaFree(dataPadded));
		HANDLE_FFTERROR(cufftDestroy(gpus.mplan));
	}

	void filterFBPpad(GPU gpus, Filter filter, 
    float *tomogram, dim3 size, dim3 size_pad, dim3 pad)
	{	
        /* int dim = { 1, 2 }
            1: if plan 1D multiples cuffts
            2: if plan 2D multiples cuffts */
        // int dim = 1;

        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock( (int)ceil( size_pad.x / TPBX ) + 1,
                        (int)ceil( size_pad.y / TPBY ) + 1,
                        (int)ceil( size_pad.z / TPBZ ) + 1);

        dim3 fft_size = dim3( size_pad.x / 2 + 1, size.y, 1 );

        size_t npad = opt::get_total_points(size_pad);

		cufftPlan1d(&gpus.mplan , size_pad.x, CUFFT_R2C, size_pad.y);
		cufftPlan1d(&gpus.mplanI, size_pad.x, CUFFT_C2R, size_pad.y);

        float *dataPadded = opt::allocGPU<float>(npad);

        opt::paddR2R<<<gridBlock,threadsPerBlock>>>(tomogram, dataPadded, size, pad);

        size_t offset; 
        for( int k = 0; k < size.z; k++){  
            
            offset = (size_t)k * size_pad.x * size_pad.y;

            convolution_R2C_C2R_1D( gpus, dataPadded + offset, fft_size, filter);
        }
        
        opt::remove_paddR2R<<<gridBlock,threadsPerBlock>>>(dataPadded, tomogram, size, pad);

        float scale = (float)(size_pad.x) * filter.pixel;

        opt::scale<<<gridBlock,threadsPerBlock>>>(tomogram, size, scale);

        HANDLE_ERROR(cudaFree(dataPadded));
		HANDLE_FFTERROR(cufftDestroy(gpus.mplan));
        HANDLE_FFTERROR(cufftDestroy(gpus.mplanI));
	}

    void filterFBP(GPU gpus, Filter filter, 
    float *tomogram, dim3 size)
	{	
        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock( (int)ceil( size.x / TPBX ) + 1,
                        (int)ceil( size.y / TPBY ) + 1,
                        (int)ceil( size.z / TPBZ ) + 1);

        dim3 fft_size = dim3( size.x / 2 + 1, size.y, 1 );

		cufftPlan1d(&gpus.mplan , size.x, CUFFT_R2C, size.y);
		cufftPlan1d(&gpus.mplanI, size.x, CUFFT_C2R, size.y);

        size_t offset; 
        for( int k = 0; k < size.z; k++){  
            
            offset = (size_t)k * size.x * size.y;

            convolution_R2C_C2R_1D( gpus, tomogram + offset, fft_size, filter);
        }
        
        float scale = (float)(size.x) * filter.pixel;

        opt::scale<<<gridBlock,threadsPerBlock>>>(tomogram, size, scale);

		HANDLE_FFTERROR(cufftDestroy(gpus.mplan));
        HANDLE_FFTERROR(cufftDestroy(gpus.mplanI));
	}
}

extern "C" {

	void SinoFilter(float* sino, size_t nrays, size_t nangles, size_t blocksize, int csino, bool bRampFilter, Filter reg, bool bShiftCenter, float* sintable, float pixel)
	{	
		cImage fft(nrays/2+1,nangles);
		// cImage fft2(nrays/2+1,nangles);

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
				BandFilterReg<<<blk,thr>>>(fft.gpuptr, nrays/2+1, csino, bShiftCenter, sintable, reg, pixel);
			else
				std::cout << __FILE__ << " " << __LINE__ << " " << "Auto reg missing!" << std::endl;

			HANDLE_FFTERROR( cufftExecC2R(plan_c2r, fft.gpuptr, sino+k*nrays*nangles) );
		}
		
		cufftDestroy(plan_r2c);
		cufftDestroy(plan_c2r);
	}

	__global__ void BandFilterReg(complex* vec, size_t sizex, int icenter, bool bShiftCenter, float* sintable, Filter mfilter, float pixel)
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

		float rampfilter = float(tx) / (float)sizex;

		rampfilter = mfilter.apply(rampfilter);

		float fcenter = 1.0f - (bShiftCenter ? (sintable[ty]) : 0);
		fcenter = -2.0f*float(M_PI)/float(2*sizex-2) * fcenter * icenter;

		if(tx < sizex)
			vec[ty*sizex + tx] *= exp1j(fcenter * tx ) * rampfilter;
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
	

	__global__ void BandFilterC2C(complex* vec, size_t sizex, int center, float pixel, Filter mfilter = Filter())
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

        float rampfilter = 1.0f * fminf(tx, sizex - tx) / (float)sizex;

		rampfilter = mfilter.apply(rampfilter);

		if(tx < sizex)
            vec[ty*sizex + tx] *= exp1j(-2*float(M_PI)/float(sizex) * center * tx) * rampfilter;

	}

    __global__ void SetX(complex* out, float* in, int sizex)
    {
        /* Float to Complex (imaginary part zero)*/
        size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t ty = blockIdx.y + gridDim.y * blockIdx.z;
        
        if(tx < sizex)
        {
            out[ty*sizex + tx].x = in[ty*sizex + tx];
            out[ty*sizex + tx].y = 0;
        }
    }

    __global__ void GetX(float* out, complex* in, int sizex, float scale)
    {
        /* Complex (real part) to Float */
        size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t ty = blockIdx.y + gridDim.y * blockIdx.z;

        if(tx < sizex)
            out[ty*sizex + tx] = in[ty*sizex + tx].x / scale;

    }

    __global__ void GetXBST(void* out, complex* in, size_t sizex, float threshold, EType::TypeEnum raftDataType, int rollxy)
    {
        size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t ty = blockIdx.y + blockDim.y * blockIdx.z;
        
        if(tx >= sizex)
            return;
        
        float fpixel = (in[ty*sizex + tx].x)/float(sizex);
        BasicOps::set_pixel(out, fpixel, tx, ty, sizex, threshold, raftDataType);
    }

    void BSTFilter_stream(cufftHandle plan,
    complex* filtersino, float* sinoblock,
    size_t nrays, size_t nangles, int csino, Filter reg, float pixel, 
    cudaStream_t stream) 
    {

        dim3 filterblock((nrays+255)/256,nangles,1);
        dim3 filterthread(256,1,1);

        SetX<<<filterblock,filterthread, 0, stream>>>(filtersino, sinoblock, nrays);

        HANDLE_FFTERROR(cufftExecC2C(plan, filtersino, filtersino, CUFFT_FORWARD));

        BandFilterC2C<<<filterblock,filterthread, 0, stream>>>(filtersino, nrays, csino, pixel, reg);

        HANDLE_FFTERROR(cufftExecC2C(plan, filtersino, filtersino, CUFFT_INVERSE));

        float scale = 1.0f; //(float)nrays * pixel;

        GetX<<<filterblock,filterthread, 0, stream>>>(sinoblock, filtersino, nrays, scale);

        //cudaMemset(sinoblock, 0, nrays*nangles*4);
    }

    void BSTFilter(cufftHandle plan,
    complex* filtersino, float* sinoblock,
    size_t nrays, size_t nangles, int csino, 
    Filter reg, float pixel) 
    {

        dim3 filterblock((nrays+255)/256,nangles,1);
        dim3 filterthread(256,1,1);

        SetX<<<filterblock,filterthread>>>(filtersino, sinoblock, nrays);

        HANDLE_FFTERROR(cufftExecC2C(plan, filtersino, filtersino, CUFFT_FORWARD));

        BandFilterC2C<<<filterblock,filterthread>>>(filtersino, nrays, csino, pixel, reg);

        HANDLE_FFTERROR(cufftExecC2C(plan, filtersino, filtersino, CUFFT_INVERSE));

        float scale = 1.0f; 

        GetX<<<filterblock,filterthread>>>(sinoblock, filtersino, nrays, scale);

    }


}

__host__ __device__ inline float Filter::apply(float input)
{
	float param = 0.0f;

	if (type == EType::gaussian)
	{
		input *= exp(-0.693f * reg * input * input) * ( 1.0f / ( 1.0f + paganin * input * input ) );
	}
	else if (type == EType::lorentz)
	{
		input *= ( 1.0f / ( 1.0f + reg * input * input ) ) * ( 1.0f / ( 1.0f + paganin * input * input ) );
	}
	else if (type == EType::cosine)
	{
		input *= cosf(float(M_PI) * 0.5f * input) * ( 1.0f / ( 1.0f + paganin * input * input ) );
	}
	else if (type == EType::rectangle)
	{
		param = fmaxf(input * reg * float(M_PI) * 0.5f, 1E-4f);
		input *= ( sinf(param) / param ) * ( 1.0f / ( 1.0f + paganin * input * input ) );
	}
	else if (type == EType::hann)
	{
		input *= ( 0.5f + 0.5f * cosf(2.0f * float(M_PI) * input) ) * ( 1.0f / ( 1.0f + paganin * input * input ) );
	}
	else if (type == EType::hamming)
	{
		input *= ( 0.54f + 0.46f * cosf(2.0f * float(M_PI) * input) ) * ( 1.0f / ( 1.0f + paganin * input * input ) );
	}
	else if (type == EType::ramp)
	{
		input *= ( 1.0f / ( 1.0f + paganin * input * input ) );
	}
    else if (type == EType::differential)
	{
		input = 1.0f / ( 2.0f * float(M_PI) * SIGN(input) );
	}
    else if (type == EType::none)
	{
		input = 1.0f;
	}

	return input;
}




