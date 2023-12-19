#include "../../../inc/filters.h"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/logerror.hpp"
#include "../../inc/common/complex.hpp"

/*============================================================================*/
/* namespace opt (in 'inc/commons/operations.hpp') functions definitions */

template<typename Type>
Type* Opt::allocGPU(size_t nsize)
{
    Type *ptr; HANDLE_ERROR(cudaMalloc((void **)&ptr, nsize * sizeof(Type) )); return ptr;
};

template<typename Type1, typename Type2>
void Opt::pointTopointProd(GPU gpus, Type1 *a, Type2 *b, Type1 *ans, dim3 sizea, dim3 sizeb)
{
    int type = Opt::assert_dimension_xyz(sizea, sizeb);

    ErrorAssert(type != 0, "Error: Arrays dimension do not match! \n");

    Opt::product<Type1,Type2>(gpus, a, b, ans, sizea, sizeb);
        
}

template<>
void Opt::product<float,float>(GPU gpus, float *a, float *b, float *ans, dim3 sizea, dim3 sizeb);
{
    Opt::product_Real_by_Real<<<gpus.Grd,gpus.BT>>>(a, b, ans, sizea, sizeb);
}

template<>
void Opt::product<float,cufftComplex>(GPU gpus, float *a, cufftComplex *b, float *ans, dim3 sizea, dim3 sizeb)
{
    Opt::product_Real_by_Complex<<<gpus.Grd,gpus.BT>>>(a, b, ans, sizea, sizeb);
}

template<>
void Opt::product<cufftComplex,float>(GPU gpus, cufftComplex *a, float *b, cufftComplex *ans, dim3 sizea, dim3 sizeb)
{
    Opt::product_Complex_by_Real<<<gpus.Grd,gpus.BT>>>(a, b, ans, sizea, sizeb);
}

template<>
void Opt::product<cufftComplex,cufftComplex>(GPU gpus, cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 sizea, dim3 sizeb)
{
    Opt::product_Complex_by_Complex<<<gpus.Grd,gpus.BT>>>(a, b, ans, sizea, sizeb);
}

__global__ void Opt::product_Real_by_Real(float *a, float *b, float *ans, dim3 sizea, dim3 sizeb)
{
    size_t indexa = getIndex3d(sizea);
    size_t indexb = getIndex3d(sizeb);
    size_t nsize  = sizea.x * sizea.y * sizea.z;
    
    if ( indexa >= nsize ) return;
    ans[indexa] = a[indexa] * b[indexb];	
}

__global__ void Opt::product_Real_by_Complex(float *a, cufftComplex *b, float *ans, dim3 sizea, dim3 sizeb)
{
    size_t indexa = getIndex3d(sizea);
    size_t indexb = getIndex3d(sizeb);
    size_t nsize  = sizea.x * sizea.y * sizea.z;
    
    if ( indexa >= nsize ) return;
    ans[indexa] = a[indexa] * b[indexb].x;	
}

__global__ void Opt::product_Complex_by_Real(cufftComplex *a, float *b, cufftComplex *ans, dim3 sizea, dim3 sizeb)
{
    size_t indexa = getIndex3d(sizea);
    size_t indexb = getIndex3d(sizeb);
    size_t nsize  = sizea.x * sizea.y * sizea.z;
    
    if ( indexa >= nsize ) return;
    ans[indexa].x = a[indexa].x * b[indexb];
    ans[indexa].y = a[indexa].y * b[indexb];	
}

__global__ void Opt::product_Complex_by_Complex(cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 sizea, dim3 sizeb)
{
    size_t indexa = getIndex3d(sizea);
    size_t indexb = getIndex3d(sizeb);
    size_t nsize  = sizea.x * sizea.y * sizea.z;
    
    if ( indexa >= nsize ) return;
    ans[indexa] = ComplexMult(a[indexa],b[indexb]);
}