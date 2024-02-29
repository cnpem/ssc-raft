#include "common/configs.hpp"
#include "common/complex.hpp"
#include "common/opt.hpp"
#include "common/configs.hpp"
#include "common/logerror.hpp"

/*============================================================================*/
/* namespace opt (in 'inc/commons/opt.hpp') functions definitions */

void opt::MPlanFFT(cufftHandle *mplan, const int dim, dim3 size)
{	
    /* int dim = { 1, 2 }
        1: if plan 1D multiples cuffts
        2: if plan 2D multiples cuffts */
    std::array<int, 2> sizeArray2D = {(int)size.x,(int)size.y};
    std::array<int, 1> sizeArray1D = {(int)size.x};

    int rank      = dim; /* 1D (1) or 2D (2) cufft */
    int *inembed  = nullptr;
    int istride   = 1;
    int idist     = size.x;
    int *onembed  = nullptr;
    int ostride   = 1;
    int odist     = size.x;
    int batch     = (2 - dim) * (size.y * size.z) +  (dim - 1)* size.z; /* batch of cuffts */
	
    if ( dim == 1 ){
        HANDLE_FFTERROR(cufftPlanMany(  mplan, rank, sizeArray1D.data(), 
                                        inembed, istride, idist, 
                                        onembed, ostride, odist, 
                                        CUFFT_C2C, batch));
    }else{
        HANDLE_FFTERROR(cufftPlanMany(  mplan, rank, sizeArray2D.data(), 
                                        inembed, istride, idist, 
                                        onembed, ostride, odist, 
                                        CUFFT_C2C, batch));
    }
}

__global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles)
{
    size_t k = blockIdx.x*blockDim.x + threadIdx.x;

    if ( (k >= nangles) ) return;

    sintable[k] = asinf(angles[k]);
    costable[k] = acosf(angles[k]);
}

void getLog(float *data, dim3 size)
{
    dim3 threadsPerBlock(64,64,size.z);
    dim3 gridBlock( (int)ceil((size.x)/threadsPerBlock.x) + 1, 
                    (int)ceil((size.y)/threadsPerBlock.y) + 1, 1);

    Klog<<<gridBlock, threadsPerBlock>>>(data, size);
}

static __global__ void Klog(float* data, dim3 size)
{  
    int i  = threadIdx.x + blockIdx.x*blockDim.x;
    int j  = threadIdx.y + blockIdx.y*blockDim.y;

    if( (i >= size.x) || (j >= size.y) || (blockIdx.z >= size.z)) return;
    
    size_t index = IND(i,j,blockIdx.z,size.x,size.y);    
    data[index]  = - logf( data[index] );
}

__global__ void opt::product_Real_Real(float *a, 
float *b, float *ans, 
dim3 sizea, dim3 sizeb)
{
    size_t indexa = opt::getIndex3d(sizea);
    size_t indexb = opt::getIndex3d(sizeb);
    size_t total_points_a = opt::get_total_points(sizea);
    size_t total_points_b = opt::get_total_points(sizeb);
    
    if ( indexa >= total_points_a || indexb >= total_points_b ) return;
    
    ans[indexa] = a[indexa] * b[indexb];	
}

__global__ void opt::product_Complex_Real(cufftComplex *a, 
float *b, cufftComplex *ans, 
dim3 sizea, dim3 sizeb)
{
    size_t indexa = opt::getIndex3d(sizea);
    size_t indexb = opt::getIndex3d(sizeb);
    size_t total_points_a = opt::get_total_points(sizea);
    size_t total_points_b = opt::get_total_points(sizeb);
    
    if ( indexa >= total_points_a || indexb >= total_points_b ) return;

    ans[indexa].x = a[indexa].x * b[indexb];
    ans[indexa].y = a[indexa].y * b[indexb];	
}

__global__ void opt::product_Complex_Complex(cufftComplex *a, 
cufftComplex *b, cufftComplex *ans, 
dim3 sizea, dim3 sizeb)
{
    size_t indexa = opt::getIndex3d(sizea);
    size_t indexb = blockIdx.x*blockDim.x + threadIdx.x;  //opt::getIndex3d(sizeb);
    size_t total_points_a = opt::get_total_points(sizea);
    size_t total_points_b = opt::get_total_points(sizeb);
    
    if ( indexa >= total_points_a || indexb >= sizeb.x ) return;

    ans[indexa] = ComplexMult(a[indexa],b[indexb]);
}

__global__ void opt::Normalize(cufftComplex *data, dim3 size, int dim)
{
    size_t norm         = (2 - dim) * (size.x * size.y) +  (dim - 1)* size.x;
    size_t index        = opt::getIndex3d(size);
    size_t total_points = opt::get_total_points(size);

    if ( index >= total_points ) return;
    
    data[index].x = data[index].x / norm; 
    data[index].y = data[index].y / norm; 
}