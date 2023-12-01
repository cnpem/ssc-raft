#include "../../inc/sscraft.h"


extern "C" {
	/*---------------------------------
	Removing Nan's values of an input
	sinogram
	Eduardo X. Miqueles 
	Gilberto Martinez Jr.
	Fernando S. Furusato 
	-------------------------------*/

	void removeNan(float *sino, 
			int length)
	{
		int i;
	
		for (i = 0; i < length; i++){
			if ( sino[i] != sino[i] ){
				sino[i] = 0.0;
			}
		}
	}
	
	__global__ void removeNan_block(float *blockSino,
					int nviews,
					int nrays,
					int blockSize)
	{
		int tx = threadIdx.x + blockIdx.x*blockDim.x; 
		int ty = threadIdx.y + blockIdx.y*blockDim.y; 
		int tz = threadIdx.z + blockIdx.z*blockDim.z;
		int voxel;

		if( (tx < nrays) && (ty < nviews) && (tz<blockSize) ){
			voxel = tz*nrays*nviews + ty*nrays + tx;
			
			if (blockSino[ voxel ] != blockSino[ voxel ]){
				blockSino[ voxel ] = 0.0;
			}

		}

	}
}


extern "C" {

	__device__ float sinc(float x)
	{
		return x==0 ? 1.0f : sqrtf(sinf(x)/x);
	}


	__global__ void KDivide_transmission(GArray<float> ffk, const GArray<float> BP, GArray<float> BI)
	{
		size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
		size_t idy = blockIdx.y;
		size_t idz = blockIdx.z;

		if(idx < ffk.shape.x)
			ffk(idz,idy,idx) *= BP(idz,idy,idx)/fmaxf(1E-4f,BI(idz,idy,idx));
	}

	__global__ void KExponential(GArray<float> ffk, GArray<float> flat)
	{
		size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
		size_t idy = blockIdx.y;
		size_t idz = blockIdx.z;

		if(idx < ffk.shape.x)
			ffk(idz,idy,idx) = flat(idz,idx)*fminf(expf(-ffk(idz,idy,idx)),2);
	}



	__global__ void KShiftSwapAndFlip(float* image, size_t sizex, size_t sizey)
	{
		size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
		size_t idy = blockIdx.y*blockDim.y + threadIdx.y;
		size_t idz = blockIdx.z*blockDim.z + threadIdx.z;

		if(idx < sizex)
		{
			size_t index = (idz * sizex * sizey) + idy * sizex + idx;
			size_t shwsf = (idz * sizex * sizey) + (((sizex-idx-1) + sizex/2) % sizex)*sizex + ((sizex-idy-1)+sizex/2)%sizex;
			
			if(shwsf > index)
			{
				float temp = image[index];
				image[index] = image[shwsf];
				image[shwsf] = temp;
			}

		}
	}


	PlanRC::PlanRC(size_t sizeimage, size_t angles, size_t _batch): sizex(sizeimage), nangles(angles), batch(_batch)
	{
		size_t worksize;
		maxworksize = 0;

		cufftCreate(&plan1d_r2c); 
		cufftSetAutoAllocation(plan1d_r2c,0); 
		cufftMakePlan1d(plan1d_r2c, sizex, CUFFT_R2C, nangles*batch, &worksize);
		maxworksize = max(worksize, maxworksize);

		cufftCreate(&plan1d_c2r); 
		cufftSetAutoAllocation(plan1d_c2r,0); 
		cufftMakePlan1d(plan1d_c2r, sizex, CUFFT_C2R, nangles*batch, &worksize);
		maxworksize = max(worksize, maxworksize);

		int nstar[2] = {(int)sizex,(int)sizex};
		cufftCreate(&plan2d_r2c); 
		cufftSetAutoAllocation(plan2d_r2c,0); 
		//cufftMakePlan2d(plan2d_r2c, sizex, sizex, CUFFT_R2C, &worksize);
		cufftMakePlanMany(plan2d_r2c, 2, nstar, nullptr, 0, 0, nullptr, 0, 0, CUFFT_R2C, batch, &worksize);
		maxworksize = max(worksize, maxworksize);

		cufftCreate(&plan2d_c2r); 
		cufftSetAutoAllocation(plan2d_c2r,0); 
		//cufftMakePlan2d(plan2d_c2r, sizex, sizex, CUFFT_C2R, &worksize);
		cufftMakePlanMany(plan2d_c2r, 2, nstar, nullptr, 0, 0, nullptr, 0, 0, CUFFT_C2R, batch, &worksize);
		maxworksize = max(worksize, maxworksize);
		
		cudaMalloc(&WorkArea, maxworksize);
		cufftSetWorkArea(plan1d_r2c, WorkArea);
		cufftSetWorkArea(plan1d_c2r, WorkArea);
		cufftSetWorkArea(plan2d_r2c, WorkArea);
		cufftSetWorkArea(plan2d_c2r, WorkArea);
	};

	PlanRC::~PlanRC()
	{
		cufftDestroy(plan1d_r2c);
		cufftDestroy(plan1d_c2r);
		cufftDestroy(plan2d_r2c);
		cufftDestroy(plan2d_c2r);
		cudaFree(WorkArea);
		WorkArea = nullptr;
	};


	__global__ void KSwapZ(float* vec, size_t N)
	{
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		const int idy = threadIdx.y + blockIdx.y * blockDim.y;
		const int idz = threadIdx.z + blockIdx.z * blockDim.z;
		const size_t index1 = idz*N*N + idy*N + idx;
		const size_t index2 = idy*N*N + idz*N + idx;

		if(idx < N && idy < N && idz < N && idz < idy)
		{
			float temp = vec[index1];
			vec[index1] = vec[index2];
			vec[index2] = temp;
		}
	}


	__global__ void KSwapZ_4gpu(float* vec0, float* vec1, float* vec2, float* vec3, int gpu, size_t N)
	{
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		const int idy = threadIdx.y + blockIdx.y * blockDim.y;
		const int idz = threadIdx.z + blockIdx.z * blockDim.z + N/4*gpu;
		const size_t index1 = idz*N*N + idy*N + idx;
		const size_t index2 = idy*N*N + idz*N + idx;

		const size_t voldiv = N*N*N/4;
		
		float* ptr1 = nullptr;
		float* ptr2 = nullptr;

		if( index1 >= 4*voldiv)
			return;
		else if( index1 < 1*voldiv )
			ptr1 = vec0 + index1 - 0*voldiv;
		else if( index1 < 2*voldiv )
			ptr1 = vec1 + index1 - 1*voldiv;
		else if( index1 < 3*voldiv )
			ptr1 = vec2 + index1 - 2*voldiv;
		else
			ptr1 = vec3 + index1 - 3*voldiv;
		
		if( index2 >= 4*voldiv)
			return;
		else if( index2 < 1*voldiv )
			ptr2 = vec0 + index2 - 0*voldiv;
		else if( index2 < 2*voldiv )
			ptr2 = vec1 + index2 - 1*voldiv;
		else if( index2 < 3*voldiv )
			ptr2 = vec2 + index2 - 2*voldiv;
		else
			ptr2 = vec3 + index2 - 3*voldiv;


		__syncwarp();
		if(idx < N && idy < N && idz < idy)
		{
			float temp = *ptr1;
			*ptr1 = *ptr2;
			*ptr2 = temp;
		}
	}


	__global__ void KComputeRes(float* FDD, const float* restrict A, const float* restrict B, const float* restrict C, const float* sino, dim3 shape, float alpha)
	{
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t index = (blockIdx.z*shape.y + blockIdx.y)*shape.x + idx;
		size_t lineindex = idx + shape.x*blockIdx.z;

		if(idx < shape.x)
		{
			float a = A[lineindex];
			float b = B[lineindex];
			float c = C[lineindex];
			float fdd = FDD[index];

			//FDD[index] = alpha*(b+2*c*fdd)*(a + b*fdd + c*sq(fdd) - sino[index]);
			FDD[index] = alpha*(a + c*(float(blockIdx.y)-float(shape.y/2)) + b*fdd - sino[index]);
		}
	}

	__global__ void UpdateABC(float* restrict A, float* restrict B, float* restrict C, const float* restrict FDD, const float* restrict sino, dim3 shape, float stepb)
	{
		size_t localidx = blockIdx.x*blockDim.x + threadIdx.x;

		if(localidx >= shape.x)
			return;

		size_t lineidx = blockIdx.z*shape.x + localidx;
		size_t planeidx = blockIdx.z*shape.x*shape.y + localidx;

		float fres = 0;
		float Rmu1 = 0;
		float RFDD = 0;
		float GTheta = 0;

		float vala = A[lineidx];
		float valb = B[lineidx];
		float valc = C[lineidx];

		for(int idy=0; idy < shape.y; idy++)
		{
			float forw = FDD[planeidx + idy*shape.x];
			float lres = vala + valc*(float(idy)-float(shape.y/2)) + valb*forw + C[idy] - sino[planeidx + idy*shape.x];

			Rmu1 += forw;

			fres += lres;
			RFDD += forw*lres;
			GTheta += (float(idy)-float(shape.y/2))*lres;
		}

		Rmu1 /= float(shape.y);

		fres /= float(shape.y);
		RFDD /= float(shape.y);
		GTheta /= float(shape.y*shape.y*shape.y/6);

		float a_new = vala - fres;
		float b_new = valb - 0.5f*(RFDD-fres*Rmu1);
		float c_new = valc - stepb*GTheta;

		A[lineidx] = fminf(fmaxf(a_new,-0.2f),0.2f);
		B[lineidx] = fminf(fmaxf(b_new,0.9f),1.1f);
		//C[lineidx] = c_new;
	}

	__global__ void KSupport(float* recon, dim3 shape, float reg, float radius)
	{
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t idy = blockIdx.y;
		size_t index = (blockIdx.z*shape.y + idy)*shape.x + idx;

		if(idx < shape.x)
		{
			//float p2 = sq(float(idx)-float(shape.x/2)) + sq(float(idy)-float(shape.y/2));

			//if(p2 > sq(radius))
			if(recon[index] < 0)
				recon[index] /= 1.0f + reg;
		}
	}

	bool accessEnabled[8][8];

	void EnablePeerToPeer(const std::vector<int>& gpus)
	{
		static bool bP2P = false;
		if(bP2P == false)
		{
			bP2P = true;
			memset(accessEnabled, 0, 8*8);
		}

		for(int g : gpus)
		{
			cudaSetDevice(g);
			for(int g2 : gpus) if(g2 != g && accessEnabled[g][g2] == false)
			{			
				int canacess;
				HANDLE_ERROR( cudaDeviceCanAccessPeer(&canacess, g, g2) );
				
				if(canacess==false)
				{
					Log("Warning: GPU" << g << " cant access GPU" << g2);
				}
				else
				{
					cudaError_t cerror = cudaDeviceEnablePeerAccess(g2, 0);
					
					if( cerror == cudaSuccess )
					{
						accessEnabled[g][g2] = true;
						Log("P2P access enabled: " << g << " <-> " << g2);
					}
					else if(cerror == cudaErrorPeerAccessAlreadyEnabled)
					{
						Log("GPU" << g << " already has access to GPU" << g2);
					}
					else
					{
						HANDLE_ERROR( cerror );
					}
				}
			}
		}
	}

	void DisablePeerToPeer(const std::vector<int>& gpus)
	{
		for(int g : gpus)
		{
			cudaSetDevice(g);
			for(int g2 : gpus) if(g2 != g)
			{
				if(accessEnabled[g][g2] == true)
				{
					HANDLE_ERROR( cudaDeviceDisablePeerAccess(g2) );
					Log("P2P access disabled: " << g << " <-> " << g2);
					accessEnabled[g][g2] = false;
				}
			}
		}
	}
}