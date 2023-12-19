#include "../../inc/processing.h"

extern "C"{

   	static __global__ void KFlatDark(float* data, float* dark, float* flat, dim3 size, int numflats)
	{  
      // Supports 2 flats only
		long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
		long long int line;
		float ft, flat_before, flat_after, dk, T, Q, interp;

		if(idx < size.x && blockIdx.y < size.y && blockIdx.z < size.z){
			
			dk          = dark[size.x * blockIdx.z            + idx];
			flat_before = flat[size.x * numflats * blockIdx.z + idx];

			line        = size.x * size.y * blockIdx.z + size.x * blockIdx.y + idx;

         	if(numflats > 1){
				interp     = float( blockIdx.y ) / float( size.y ); 

				flat_after = flat[size.x * numflats * blockIdx.z + size.x + idx];

				ft         = flat_before * ( 1.0f - interp ) + interp * flat_after;
			}else{
				ft = flat_before;
			}

			T = data[line] - dk;
			Q = ft         - dk;

			data[line] = fmaxf(T, 0.5f) / fmaxf(Q,0.5f) ; 

		}
	}

	static __global__ void Klog(float* in, dim3 size)
	{  
		long long int idx = threadIdx.x + blockIdx.x*blockDim.x;
		long long int line;

		if(idx < size.x && blockIdx.y < size.y && blockIdx.z < size.z){
			
			line     = size.x * size.y * blockIdx.z + size.x * blockIdx.y + idx;

			in[line] = - logf( in[line] );

		}
	}

	void getLog(float *tomogram, dim3 size, GPU gpus)
	{
		Klog<<<gpus.Grd,gpus.BT>>>(tomogram, dim3(size.x,size.y,size.z));
	}

	void getFlatDarkCorrection(float* frames, float* flat, float* dark, dim3 size, int numflats, GPU gpus)
	{
		/* Do the dark subtraction and division by flat (with or without log) */
		KFlatDark<<<gpus.Grd,gpus.BT>>>(frames, dark, flat, dim3(size.x,size.y,size.z), numflats);

	}

	void getFlatDarkGPU(GPU gpus, int gpu, float* frames, float* flat, float* dark, dim3 size, int numflats, int is_log)
	{	
		// Supports 2 flats max
		cudaSetDevice(gpu);

		int i;
		int blocksize = min(size.z,32); // Herança do Giovanni -> Mudar

		int nblock = (int)ceil( (float) size.z / blocksize );
		int ptr = 0, subblock;

		// GPUs Pointers: declaration and allocation
		rImage data(size.x,size.y,blocksize); // Frames in GPU 

		Image2D<float> cflat(size.x, numflats, blocksize); // Flat in GPU
		Image2D<float> cdark(size.x,        1, blocksize); // Dark in GPU

		for(i = 0; i < nblock; i++){
			
			subblock = min(size.z - ptr, blocksize);

			data.CopyFrom (frames + (size_t)ptr * size.x * size.y  , 0, (size_t)subblock * size.x * size.y  );
			cflat.CopyFrom(flat   + (size_t)ptr * size.x * numflats, 0, (size_t)subblock * size.x * numflats);
			cdark.CopyFrom(dark   + (size_t)ptr * size.x           , 0, (size_t)subblock * size.x           );

			getFlatDarkCorrection(data.gpuptr, cflat.gpuptr, cdark.gpuptr, dim3(size.x, size.y, subblock), numflats, gpus);

			if (is_log == 1)
				getLog(data.gpuptr, dim3(size.x, size.y, subblock), gpus);

			data.CopyTo(frames + (size_t)ptr * size.x * size.y, 0, (size_t)subblock * size.x * size.y);
      
			/* Update pointer */
			ptr = ptr + subblock;
		}

		cudaDeviceSynchronize();
	}

	void getFlatDarkMultiGPU(int* gpus, int ngpus, float* frames, float* flat, float* dark, int nrays, int nangles, int nslices, int numflats, int is_log)
	{
		int i;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		int ptr = 0, subblock;
		
		GPU gpu_parameters;

		setGPUParameters(&gpu_parameters, dim3(nrays,nangles,nslices), ngpus, gpus);

		std::vector<std::future<void>> threads;
		
		if ( ngpus == 1 ){
			getFlatDarkGPU(gpu_parameters, gpus[0], frames, flat, dark, dim3(nrays, nangles, nslices), numflats, is_log);
		}else{
			for(i = 0; i < ngpus; i++){ 
				
				subblock = min(nslices - ptr, blockgpu);

				threads.push_back(std::async( std::launch::async, 
							getFlatDarkGPU, 
							gpu_parameters,
							gpus[i], 
							frames + (size_t)ptr*nrays*nangles, 
							flat   + (size_t)ptr*nrays*numflats, 
							dark   + (size_t)ptr*nrays, 
							dim3(nrays, nangles, subblock), 
							numflats, is_log
							));

				/* Update pointer */
				ptr = ptr + subblock;
			}

			for(auto& t : threads)
				t.get();

		}
	}
}

