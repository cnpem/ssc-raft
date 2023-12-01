#include "../../../inc/sscraft.h"

extern "C"{
	
	void filterFBP(GPU gpus, Filter filter, float *tomogram, cufftComplex *filter_kernel, dim3 size, dim3 size_pad)
	{	
		int n[] = {(int)size_pad.x};
		cufftPlanMany(&gpus.mplan1dC2C, 1, n, n, 1, size_pad.x, n, 1, size_pad.x, CUFFT_C2C, (long long int)size.y * size.z);

		_fbp_filter<<<gpus.Grd,gpus.BT>>>(filter, filter_kernel, size_pad);

		convolution_mplan1DR2R(gpus, tomogram, filter_kernel, 0.0f, size, size_pad);

		cufftDestroy(gpus.mplan1dC2C);
	}

	__global__ void _fbp_filter(Filter filter, cufftComplex *kernel, dim3 size)
	{
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = size.x * j + i;
		size_t index = size.y * k * size.x + ind;

		if( i > size.x ) return;

		float dt   =        2.0f / (float)size.x;
		float wMax =        1.0f / ( 2.0f * dt );
		float dw   = 2.0f * wMax / (float)size.x;

		/* Reciprocal grid */ 
        float w = - wMax + i * dw;

		w = filter.apply( w );
	
		kernel[index] = ( exp1j(- 2.0f * float(M_PI) * filter.axis_offset * w ) * w );
	}
}

extern "C" {

	void convolution_mplan2DR2R(GPU gpus, float *data, float *kernel, float pad_value, dim3 size, dim3 size_pad)
	{
        dim3 pad_size = dim3(size_pad.x, size_pad.y, size.z);
        size_t npad   = (size_t)size_pad.x * size_pad.y * size.z;

        cufftComplex *dataPadded;

		HANDLE_ERROR(cudaMalloc((void **)&dataPadded, sizeof(cufftComplex) * npad ));

        padding<<<gpus.Grd,gpus.BT>>>(data, dataPadded, pad_value, size, pad_size);
        
        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_FORWARD));
        
        ProdComplexFloat<<<gpus.Grd,gpus.BT>>>(dataPadded, kernel, dataPadded, pad_size);	
        
        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_INVERSE));
        
        fftNormalize2d<<<gpus.Grd,gpus.BT>>>(dataPadded, pad_size);

        recuperate_padding<<<gpus.Grd,gpus.BT>>>(dataPadded, data, size, pad_size);

		cudaFree(dataPadded);
	}

	void convolution_mplan1DR2R(GPU gpus, float *data, cufftComplex *kernel, float pad_value, dim3 size, dim3 size_pad)
	{
        dim3 pad_size = dim3(size_pad.x, size.y, size.z);
        size_t npad   = (size_t)size_pad.x * size.y * size.z;

        cufftComplex *dataPadded;

		HANDLE_ERROR(cudaMalloc((void **)&dataPadded, sizeof(cufftComplex) * npad ));

        padding<<<gpus.Grd,gpus.BT>>>(data, dataPadded, pad_value, size, pad_size);
        
        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_FORWARD));
        
        ProdComplexComplex<<<gpus.Grd,gpus.BT>>>(dataPadded, kernel, dataPadded, pad_size);	
        
        HANDLE_FFTERROR(cufftExecC2C(gpus.mplan, dataPadded, dataPadded, CUFFT_INVERSE));
        
        fftNormalize1d<<<gpus.Grd,gpus.BT>>>(dataPadded, pad_size);

        recuperate_padding<<<gpus.Grd,gpus.BT>>>(dataPadded, data, size, pad_size);

		cudaFree(dataPadded);
	}

    __global__ void ProdComplexFloat(cufftComplex *a, float *b, cufftComplex *ans, dim3 size)
    {
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = size.x * j + i;
        size_t index = size.y * k * size.x + ind;

        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;
        ans[index].x = a[index].x * b[ind];	
        ans[index].y = a[index].y * b[ind];	
    }

	__global__ void ProdComplexComplex(cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 size)
    {
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = size.x * j + i;
        size_t index = size.y * k * size.x + ind;

        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;
        ans[index] = ComplexMult(a[index],b[ind]);
		// ans[index] = a[index] * b[ind];
    }

    __global__ void fftNormalize2d(cufftComplex *c, dim3 size)
    {
        int N = ( size.x * size.y );	
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t index = size.x * ( k * size.y + j) + i;
        
        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;
        
        c[index].x /= (float)N; c[index].y /= (float)N; 
    }

	__global__ void fftNormalize1d(cufftComplex *c, dim3 size)
    {
        int N = ( size.x );	
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t index = size.x * ( k * size.y + j) + i;
        
        if ( (i >= size.x) || (j >= size.y) || (k >= size.z) ) return;
        
        c[index].x /= (float)N; c[index].y /= (float)N; 
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
	
	// inline __global__ void SetX(complex* out, float* in, int sizex)
	// {
	// 	size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
	// 	size_t ty = blockIdx.y + gridDim.y * blockIdx.z;
		
	// 	if(tx < sizex)
	// 	{
	// 		out[ty*sizex + tx].x = in[ty*sizex + tx];
	// 		out[ty*sizex + tx].y = 0;
	// 	}
	// }
	
	// inline __global__ void GetX(float* out, complex* in, int sizex)
	// {
	// 	size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
	// 	size_t ty = blockIdx.y + gridDim.y * blockIdx.z;
		
	// 	if(tx < sizex)
	// 		out[ty*sizex + tx] = (in[ty*sizex + tx].x)/sizex;
	// }
	
	// inline __global__ void GetXBST(void* out, complex* in, size_t sizex, float threshold, EType::TypeEnum raftDataType, int rollxy)
	// {
	// 	size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
	// 	size_t ty = blockIdx.y + blockDim.y * blockIdx.z;
		
	// 	if(tx >= sizex)
	// 		return;
		
	// 	float fpixel = (in[ty*sizex + tx].x)/float(sizex);
	// 	 BasicOps::set_pixel(out, fpixel, tx, ty, sizex, threshold, raftDataType);
	// }

	__global__ void BandFilterC2C(complex* vec, size_t sizex, int center, Filter mfilter = Filter())
	{
		int tx = blockIdx.x * blockDim.x + threadIdx.x;
		int ty = blockIdx.y * blockDim.y + threadIdx.y;

		float rampfilter = 2.0f*fminf(tx,sizex-tx)/(float)sizex;
		rampfilter = mfilter.apply(rampfilter);

		if(tx < sizex)
			vec[ty*sizex + tx] *= exp1j(-2*float(M_PI)/float(sizex) * center * tx) * rampfilter;
	}

	// void BSTFilter(cufftHandle plan, complex* filtersino, float* sinoblock, size_t nrays, size_t nangles, int csino, Filter reg)
	// {

	// 	dim3 filterblock((nrays+255)/256,nangles,1);
	// 	dim3 filterthread(256,1,1);

	// 	SetX<<<filterblock,filterthread>>>(filtersino, sinoblock, nrays);
			
	// 	HANDLE_FFTERROR(cufftExecC2C(plan, filtersino, filtersino, CUFFT_FORWARD));
			
	// 	BandFilterC2C<<<filterblock,filterthread>>>(filtersino, nrays, csino, reg);
			
	// 	HANDLE_FFTERROR(cufftExecC2C(plan, filtersino, filtersino, CUFFT_INVERSE));
		
	// 	GetX<<<filterblock,filterthread>>>(sinoblock, filtersino, nrays);

	// 	//cudaMemset(sinoblock, 0, nrays*nangles*4);
	// }

}