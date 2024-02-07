#include "../../inc/common/configs.h"
#include "../../inc/common/logerror.hpp"


__global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles)
{
    size_t k = blockIdx.x*blockDim.x + threadIdx.x;

    if ( (k >= nangles) ) return;

    sintable[k] = asinf(angles[k]);
    costable[k] = acosf(angles[k]);
}