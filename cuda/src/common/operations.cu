#include "include.hpp"
#include "common/operations.hpp"
#include "common/types.hpp"

/*======================================================================*/
/* struct EType (in 'inc/commons/operations.hpp') functions definitions */

size_t EType::Size(EType::TypeEnum datatype)
{
	static const size_t datasizes[] = {0, 1, 2, 4, 2, 4, 8};
	return datasizes[static_cast<int>(datatype)];
}

std::string EType::String(EType::TypeEnum type)
{
	static const std::string datanames[] = {"INVALID",
											"UINT8",
											"UINT16",
											"INT32",
											"HALF",
											"FLOAT32",
											"DOUBLE"};

	return datanames[static_cast<int>(type)];
}

EType EType::Type(const std::string &nametype)
{
	EType etype;

	for (int i = 0; i < static_cast<int>(EType::TypeEnum::NUM_ENUMS); i++)
		if (nametype == EType::String(static_cast<EType::TypeEnum>(i)))
			etype.type = static_cast<EType::TypeEnum>(i);

	return etype;
}

/*============================================================================*/
/* namespace BasicOps (in 'inc/commons/operations.hpp') functions definitions */

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Add(Type1 *a, const Type2 *b, size_t size, size_t size2)
{
	const size_t index = BasicOps::GetIndex();

	if (index >= size)
		return;

	a[index] += b[index % size2];
}

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Sub(Type1 *a, const Type2 *b, size_t size, size_t size2)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] -= b[index % size2];
}

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Mul(Type1 *a, const Type2 *b, size_t size, size_t size2)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] *= b[index % size2];
}

template <typename Type1, typename Type2>
__global__ void BasicOps::KB_Div(Type1 *a, const Type2 *b, size_t size, size_t size2)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] /= b[index % size2];
}

template <typename Type1, typename Type2 = Type1>
__global__ void BasicOps::KB_Add(Type1 *a, Type2 n, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] += n;
}

template <typename Type1, typename Type2 = Type1>
__global__ void BasicOps::KB_Sub(Type1 *a, Type2 n, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] -= n;
}

template <typename Type1, typename Type2 = Type1>
__global__ void BasicOps::KB_Mul(Type1 *a, Type2 n, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] *= n;
}

template <typename Type1, typename Type2 = Type1>
__global__ void BasicOps::KB_Div(Type1 *a, Type2 n, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	if (index >= size)
		return;

	a[index] /= n;
}

template <typename Type>
__global__ void BasicOps::KB_Log(Type *a, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] = logf(fmaxf(a[index], 1E-10f));
}

template <typename Type>
__global__ void BasicOps::KB_Exp(Type *a, size_t size, bool bNeg)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	float var = a[index];
	a[index] = expf(bNeg ? (-var) : var);
}

template <typename Type>
__global__ void BasicOps::KB_Clamp(Type *a, const Type b, const Type c, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	a[index] = clamp(a[index], b, c);
}

template <typename Type>
__global__ void BasicOps::KB_log1j(float *out, const Type *in, size_t size) {}

template <typename Type>
__global__ void BasicOps::KB_exp1j(Type *out, const float *in, size_t size) {}

template <>
__global__ void BasicOps::KB_log1j<complex>(float *out, const complex *in, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = in[index].angle();
}

template <>
__global__ void BasicOps::KB_exp1j<complex>(complex *out, const float *in, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = complex::exp1j(in[index]);
}

template <typename Type>
__global__ void BasicOps::KB_Power(Type *a, float P, size_t size) {}

template <>
__global__ void BasicOps::KB_Power<float>(float *out, float P, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = expf(logf(fmaxf(fabsf(out[index]), 1E-25f)) * P);
}

template <>
__global__ void BasicOps::KB_Power<complex>(complex *out, float P, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = complex::exp1j(out[index].angle() * P) * expf(logf(fmaxf(out[index].abs(), 1E-25f)) * P);
}

template <typename Type>
__global__ void BasicOps::KB_ABS2(float *out, Type *a, size_t size) {}

template <>
__global__ void BasicOps::KB_ABS2<complex>(float *out, complex *in, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = in[index].abs2();
}

template <>
__global__ void BasicOps::KB_ABS2<float>(float *out, float *in, size_t size)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = in[index] * in[index];
}

#ifdef _USING_FP16
template <typename Type2, typename Type1>
__global__ void BasicOps::KConvert(Type2 *out, Type1 *in, size_t size, float threshold)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	float res = (float)in[index];

	if (std::is_same<Type2, __half>::value == true)
		res *= 1024.0f;
	else if (std::is_floating_point<Type2>::value == false)
		res = fminf(res / threshold, 1.0f) * ((1UL << (8 * (sizeof(Type2) & 0X7))) - 1);

	out[index] = Type2(res);
}

template <>
__global__ void BasicOps::KConvert<complex, complex16>(complex *out, complex16 *in, size_t size, float threshold)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = complex(in[index]);
}
template <>
__global__ void BasicOps::KConvert<complex16, complex>(complex16 *out, complex *in, size_t size, float threshold)
{
	const size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	out[index] = complex16(in[index]);
}
#endif

template <typename Type>
__global__ void BasicOps::KFFTshift1(Type *img, size_t sizex, size_t sizey)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
	size_t idz = blockIdx.z;

	if (idx < sizex / 2 && idy < sizey)
	{
		size_t index1 = idz * sizex * sizey + idy * sizex + idx;
		size_t index2 = idz * sizex * sizey + idy * sizex + idx + sizex / 2;

		Type temp = img[index1];
		img[index1] = img[index2];
		img[index2] = temp;
	}
}
template <typename Type>
__global__ void BasicOps::KFFTshift2(Type *img, size_t sizex, size_t sizey)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
	size_t idz = blockIdx.z;

	if (idx < sizex && idy < sizey / 2)
	{
		size_t index1 = idz * sizex * sizey + idy * sizex + idx;
		size_t index2 = idz * sizex * sizey + (idy + sizey / 2) * sizex + (idx + sizex / 2) % sizex;

		Type temp = img[index1];
		img[index1] = img[index2];
		img[index2] = temp;
	}
}
template <typename Type>
__global__ void BasicOps::KFFTshift3(Type *img, size_t sizex, size_t sizey, size_t sizez)
{
	size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
	size_t idy = threadIdx.y + blockIdx.y * blockDim.y;
	size_t idz = blockIdx.z;

	if (idx < sizex && idy < sizey / 2)
	{
		size_t index1 = idz * sizex * sizey + idy * sizex + idx;
		size_t index2 = ((idz + sizez / 2) % sizez) * sizex * sizey + (idy + sizey / 2) * sizex + (idx + sizex / 2) % sizex;

		Type temp = img[index1];
		img[index1] = img[index2];
		img[index2] = temp;
	}
}

/*=============================================================================*/
/* namespace Reduction (in 'inc/commons/operations.hpp') functions definitions */

__global__ void Reduction::KGlobalReduce(float *out, const float *in, size_t size)
{
	__shared__ float intermediate[32];
	if (threadIdx.x < 32)
		intermediate[threadIdx.x] = 0;
	__syncthreads();

	float mine = 0;

	for (size_t index = threadIdx.x + blockIdx.x * blockDim.x; index < size; index += blockDim.x * gridDim.x)
		mine += in[index];

	atomicAdd(intermediate + threadIdx.x % 32, mine);

	__syncthreads();

	Reduction::KSharedReduce32(intermediate);
	if (threadIdx.x == 0)
		out[blockIdx.x] = intermediate[0];
}

/*========================================================================*/
/* namespace Sync (in 'inc/commons/operations.hpp') functions definitions */

template <typename Type>
__global__ void Sync::KWeightedLerp(Type *val, const Type *acc, const float *div, size_t size, float lerp)
{
	size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	Type weighed = acc[index] / (div[index] + 1E-10f);
	val[index] = weighed * lerp + val[index] * (1.0f - lerp);
}

template <typename Type>
__global__ void Sync::KMaskedSum(Type *cval, const Type *acc, size_t size, const uint32_t *mask2)
{
	size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	uint32_t mask = mask2[index / 32];
	bool value = (mask >> (index & 0x1F)) & 0x1;

	if (value)
		cval[index] += acc[index];
}

template <typename Type>
__global__ void Sync::KMaskedBroadcast(Type *cval, const Type *acc, size_t size, const uint32_t *mask2)
{
	size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	uint32_t mask = mask2[index / 32];
	bool value = (mask >> (index & 0x1F)) & 0x1;

	if (value)
		cval[index] = acc[index];
}

template <typename Type>
__global__ void Sync::KSetMask(uint32_t *mask, const Type *value, size_t size, float thresh){};

template <>
__global__ void Sync::KSetMask<float>(uint32_t *mask, const float *fval, size_t size, float thresh)
{
	__shared__ uint32_t shvalue[1];
	if (threadIdx.x < 32)
		shvalue[threadIdx.x] = 0;

	__syncthreads();

	size_t index = BasicOps::GetIndex();
	if (index >= size)
		return;

	uint32_t value = (fval[index] > thresh) ? 1 : 0;
	value = value << threadIdx.x;

	atomicOr(shvalue, value);

	__syncthreads();
	if (threadIdx.x == 0)
		mask[index / 32] = shvalue[0];
}

extern "C"
{
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

		for (i = 0; i < length; i++)
		{
			if (sino[i] != sino[i])
			{
				sino[i] = 0.0;
			}
		}
	}

	__global__ void removeNan_block(float *blockSino,
									int nviews,
									int nrays,
									int blockSize)
	{
		int tx = threadIdx.x + blockIdx.x * blockDim.x;
		int ty = threadIdx.y + blockIdx.y * blockDim.y;
		int tz = threadIdx.z + blockIdx.z * blockDim.z;
		int voxel;

		if ((tx < nrays) && (ty < nviews) && (tz < blockSize))
		{
			voxel = tz * nrays * nviews + ty * nrays + tx;

			if (blockSino[voxel] != blockSino[voxel])
			{
				blockSino[voxel] = 0.0;
			}
		}
	}
}

extern "C"{

	__device__ float sinc(float x)
	{
		return x == 0 ? 1.0f : sqrtf(sinf(x) / x);
	}

	__global__ void KDivide_transmission(GArray<float> ffk, const GArray<float> BP, GArray<float> BI)
	{
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t idy = blockIdx.y;
		size_t idz = blockIdx.z;

		if (idx < ffk.shape.x)
			ffk(idz, idy, idx) *= BP(idz, idy, idx) / fmaxf(1E-4f, BI(idz, idy, idx));
	}

	__global__ void KExponential(GArray<float> ffk, GArray<float> flat)
	{
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t idy = blockIdx.y;
		size_t idz = blockIdx.z;

		if (idx < ffk.shape.x)
			ffk(idz, idy, idx) = flat(idz, idx) * fminf(expf(-ffk(idz, idy, idx)), 2);
	}

	__global__ void KShiftSwapAndFlip(float *image, size_t sizex, size_t sizey)
	{
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
		size_t idz = blockIdx.z * blockDim.z + threadIdx.z;

		if (idx < sizex)
		{
			size_t index = (idz * sizex * sizey) + idy * sizex + idx;
			size_t shwsf = (idz * sizex * sizey) + (((sizex - idx - 1) + sizex / 2) % sizex) * sizex + ((sizex - idy - 1) + sizex / 2) % sizex;

			if (shwsf > index)
			{
				float temp = image[index];
				image[index] = image[shwsf];
				image[shwsf] = temp;
			}
		}
	}

	__global__ void KSwapZ(float *vec, size_t N)
	{
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		const int idy = threadIdx.y + blockIdx.y * blockDim.y;
		const int idz = threadIdx.z + blockIdx.z * blockDim.z;
		const size_t index1 = idz * N * N + idy * N + idx;
		const size_t index2 = idy * N * N + idz * N + idx;

		if (idx < N && idy < N && idz < N && idz < idy)
		{
			float temp = vec[index1];
			vec[index1] = vec[index2];
			vec[index2] = temp;
		}
	}

	__global__ void KSwapZ_4gpu(float *vec0, float *vec1, float *vec2, float *vec3, int gpu, size_t N)
	{
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		const int idy = threadIdx.y + blockIdx.y * blockDim.y;
		const int idz = threadIdx.z + blockIdx.z * blockDim.z + N / 4 * gpu;
		const size_t index1 = idz * N * N + idy * N + idx;
		const size_t index2 = idy * N * N + idz * N + idx;

		const size_t voldiv = N * N * N / 4;

		float *ptr1 = nullptr;
		float *ptr2 = nullptr;

		if (index1 >= 4 * voldiv)
			return;
		else if (index1 < 1 * voldiv)
			ptr1 = vec0 + index1 - 0 * voldiv;
		else if (index1 < 2 * voldiv)
			ptr1 = vec1 + index1 - 1 * voldiv;
		else if (index1 < 3 * voldiv)
			ptr1 = vec2 + index1 - 2 * voldiv;
		else
			ptr1 = vec3 + index1 - 3 * voldiv;

		if (index2 >= 4 * voldiv)
			return;
		else if (index2 < 1 * voldiv)
			ptr2 = vec0 + index2 - 0 * voldiv;
		else if (index2 < 2 * voldiv)
			ptr2 = vec1 + index2 - 1 * voldiv;
		else if (index2 < 3 * voldiv)
			ptr2 = vec2 + index2 - 2 * voldiv;
		else
			ptr2 = vec3 + index2 - 3 * voldiv;

		__syncwarp();
		if (idx < N && idy < N && idz < idy)
		{
			float temp = *ptr1;
			*ptr1 = *ptr2;
			*ptr2 = temp;
		}
	}

	__global__ void KComputeRes(float *FDD, const float *restrict A, const float *restrict B, const float *restrict C, const float *sino, dim3 shape, float alpha)
	{
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t index = (blockIdx.z * shape.y + blockIdx.y) * shape.x + idx;
		size_t lineindex = idx + shape.x * blockIdx.z;

		if (idx < shape.x)
		{
			float a = A[lineindex];
			float b = B[lineindex];
			float c = C[lineindex];
			float fdd = FDD[index];

			// FDD[index] = alpha*(b+2*c*fdd)*(a + b*fdd + c*sq(fdd) - sino[index]);
			FDD[index] = alpha * (a + c * (float(blockIdx.y) - float(shape.y / 2)) + b * fdd - sino[index]);
		}
	}

	__global__ void UpdateABC(float *restrict A, float *restrict B, float *restrict C, const float *restrict FDD, const float *restrict sino, dim3 shape, float stepb)
	{
		size_t localidx = blockIdx.x * blockDim.x + threadIdx.x;

		if (localidx >= shape.x)
			return;

		size_t lineidx = blockIdx.z * shape.x + localidx;
		size_t planeidx = blockIdx.z * shape.x * shape.y + localidx;

		float fres = 0;
		float Rmu1 = 0;
		float RFDD = 0;
		float GTheta = 0;

		float vala = A[lineidx];
		float valb = B[lineidx];
		float valc = C[lineidx];

		for (int idy = 0; idy < shape.y; idy++)
		{
			float forw = FDD[planeidx + idy * shape.x];
			float lres = vala + valc * (float(idy) - float(shape.y / 2)) + valb * forw + C[idy] - sino[planeidx + idy * shape.x];

			Rmu1 += forw;

			fres += lres;
			RFDD += forw * lres;
			GTheta += (float(idy) - float(shape.y / 2)) * lres;
		}

		Rmu1 /= float(shape.y);

		fres /= float(shape.y);
		RFDD /= float(shape.y);
		GTheta /= float(shape.y * shape.y * shape.y / 6);

		float a_new = vala - fres;
		float b_new = valb - 0.5f * (RFDD - fres * Rmu1);
		// float c_new = valc - stepb * GTheta;

		A[lineidx] = fminf(fmaxf(a_new, -0.2f), 0.2f);
		B[lineidx] = fminf(fmaxf(b_new, 0.9f), 1.1f);
		// C[lineidx] = c_new;
	}

	__global__ void KSupport(float *recon, dim3 shape, float reg, float radius)
	{
		size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
		size_t idy = blockIdx.y;
		size_t index = (blockIdx.z * shape.y + idy) * shape.x + idx;

		if (idx < shape.x)
		{
			// float p2 = sq(float(idx)-float(shape.x/2)) + sq(float(idy)-float(shape.y/2));

			// if(p2 > sq(radius))
			if (recon[index] < 0)
				recon[index] /= 1.0f + reg;
		}
	}

	bool accessEnabled[8][8];

	void EnablePeerToPeer(const std::vector<int> &gpus)
	{
		static bool bP2P = false;
		if (bP2P == false)
		{
			bP2P = true;
			memset(accessEnabled, 0, 8 * 8);
		}

		for (int g : gpus)
		{
			cudaSetDevice(g);
			for (int g2 : gpus)
				if (g2 != g && accessEnabled[g][g2] == false)
				{
					int canacess;
					HANDLE_ERROR(cudaDeviceCanAccessPeer(&canacess, g, g2));

					if (canacess == false)
					{
						Log("Warning: GPU" << g << " cant access GPU" << g2);
					}
					else
					{
						cudaError_t cerror = cudaDeviceEnablePeerAccess(g2, 0);

						if (cerror == cudaSuccess)
						{
							accessEnabled[g][g2] = true;
							Log("P2P access enabled: " << g << " <-> " << g2);
						}
						else if (cerror == cudaErrorPeerAccessAlreadyEnabled)
						{
							Log("GPU" << g << " already has access to GPU" << g2);
						}
						else
						{
							HANDLE_ERROR(cerror);
						}
					}
				}
		}
	}

	void DisablePeerToPeer(const std::vector<int> &gpus)
	{
		for (int g : gpus)
		{
			cudaSetDevice(g);
			for (int g2 : gpus)
				if (g2 != g)
				{
					if (accessEnabled[g][g2] == true)
					{
						HANDLE_ERROR(cudaDeviceDisablePeerAccess(g2));
						Log("P2P access disabled: " << g << " <-> " << g2);
						accessEnabled[g][g2] = false;
					}
				}
		}
	}
}