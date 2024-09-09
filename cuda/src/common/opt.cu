#include <driver_types.h>

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

#include "common/complex.hpp"
#include "common/configs.hpp"
#include "common/logerror.hpp"
#include "common/opt.hpp"

/*============================================================================*/
/* namespace opt (in 'inc/commons/opt.hpp') functions definitions */


void opt::flip_x(float *data, int sizex, int sizey, int sizez) {
    const size_t sizexy = sizex * sizey;

    for (size_t k = 0; k < sizez; ++k) {
        for (size_t j = 0; j < sizey; ++j) {
            float* row = data + j * sizex + k * sizexy;
            for (size_t i = 0; i < (sizex/2); ++i) {
                std::swap(row[i], row[sizex - 1 - i]);
            }
        }
    }
}

//this implementation is still quite naive, but can be faster than numpy for big arrays
void opt::transpose_cpu(float *data, int sizex, int sizey, int sizez) {
    const size_t sizexy = sizex * sizey;
    const size_t sizexz = sizex * sizez;
    const size_t sizexyz = size_t(sizex) * size_t(sizey) * size_t(sizez);

    float *temp = (float *)aligned_alloc(64, sizeof(float) * sizexyz);

    for (size_t j = 0; j < sizey; ++j) {
        for (size_t k = 0; k < sizez; ++k) {
            memcpy(temp + j * sizexz + k * sizex, data + j * sizex + k * sizexy,
                    sizeof(float) * sizex);
        }
    }

    memcpy(data, temp, sizeof(float) * sizexyz);

    free(temp);
}

void opt::MPlanFFT(cufftHandle *mplan, int RANK, dim3 DATASIZE, cufftType FFT_TYPE) {
    /* rank:
    Dimensionality of the transform: 1D (1), 2D (2) or 3D (3) cufft */

    /* FFT_TYPE: The transform data type
        CUFFT_R2C, CUFFT_C2R, CUFFT_C2C
    */

    /* Array of size rank, describing the size of each dimension,
    n[0] being the size of the outermost and
    n[rank-1] innermost (contiguous) dimension of a transform. */
    int *n = (int *)malloc(RANK);
    n[0] = (int)DATASIZE.x;

    int idist = DATASIZE.x; /* Input data distance between batches */
    int odist = DATASIZE.x; /* Output data distance between batches */

    if (FFT_TYPE == CUFFT_C2R) idist = idist / 2 + 1;

    if (FFT_TYPE == CUFFT_R2C) odist = odist / 2 + 1;

    int batch = DATASIZE.y * DATASIZE.z; /* Number of batched executions */

    if (RANK >= 2) n[1] = (int)DATASIZE.y;
    batch = DATASIZE.z;
    idist *= DATASIZE.y;
    odist *= DATASIZE.y;

    if (RANK >= 3) {
        n[2] = (int)DATASIZE.z;
        idist *= DATASIZE.z;
        odist *= DATASIZE.z;
        batch = 1;
    }

    int *inembed = NULL, *onembed = NULL; /* Input/Output size with pitch (ignored for 1D transforms).
    If set to NULL all other advanced data layout parameters are ignored. */
    int istride = 1, ostride = 1;         /* Distance between two successive input/output elements. */

    HANDLE_FFTERROR(cufftPlanMany(mplan, RANK, n, inembed, istride, idist, onembed, ostride, odist, FFT_TYPE, batch));
}

__global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles) {
    size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    if ((k >= nangles)) return;

    sintable[k] = __sinf(angles[k]);
    costable[k] = __cosf(angles[k]);
}

void getLog(float *data, dim3 size, cudaStream_t stream) {
    dim3 threadsPerBlock(TPBX, TPBY, TPBZ);
    dim3 gridBlock((int)ceil(size.x / threadsPerBlock.x) + 1, (int)ceil(size.y / threadsPerBlock.y) + 1,
                   (int)ceil(size.z / threadsPerBlock.z) + 1);

    Klog<<<gridBlock, threadsPerBlock, 0, stream>>>(data, size);
}

static __global__ void Klog(float *data, dim3 size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if ((i >= size.x) || (j >= size.y) || (k >= size.z)) return;

    size_t index = IND(i, j, k, size.x, size.y);

    data[index] = -logf(data[index]);
}


dim3 opt::setGridBlock(dim3 size) {
    dim3 gridBlock((int)ceil(size.x / TPBX) + 1, (int)ceil(size.y / TPBY) + 1, (int)ceil(size.z / TPBZ) + 1);

    return gridBlock;
}

__global__ void opt::scale(cufftComplex *data, dim3 size, float scale) {

    size_t i = blockIdx.x*blockDim.x + threadIdx.x; 
    size_t j = blockIdx.y*blockDim.y + threadIdx.y; 
    size_t k = blockIdx.z*blockDim.z + threadIdx.z;

    size_t index = size.y * size.x * k + size.x * j + i; 

    if( (i >= size.x) || (j >= size.y) || (k >= size.z)) return;  

    data[index].x = data[index].x / scale;
    data[index].y = data[index].y / scale;
}

