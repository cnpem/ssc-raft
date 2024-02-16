#include "common/configs.hpp"
#include "common/complex.hpp"
#include "common/opt.hpp"
#include "common/configs.hpp"
#include "common/logerror.hpp"

/*============================================================================*/
/* namespace opt (in 'inc/commons/opt.hpp') functions definitions */

template<typename Type>
Type* opt::allocGPU(size_t size)
{
    Type *ptr; HANDLE_ERROR(cudaMalloc((void **)&ptr, size * sizeof(Type) )); return ptr;
};

template<typename Type>
void opt::CPUToGPU(Type *cpuptr, Type *gpuptr, size_t size)
{
    HANDLE_ERROR(cudaMemcpy(gpuptr, cpuptr, size * sizeof(Type), cudaMemcpyHostToDevice));
};

template<typename Type>
void opt::GPUToCPU(Type *cpuptr, Type *gpuptr, size_t size)
{
    HANDLE_ERROR(cudaMemcpy(cpuptr, gpuptr, size * sizeof(Type), cudaMemcpyDeviceToHost));
};

void opt::MPlanFFT(cufftHandle mplan, const int dim, dim3 size)
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
    size_t batch  = size.z; /* batch of cuffts */
	
    if ( dim == 1 ){
        HANDLE_FFTERROR(cufftPlanMany(  &mplan, rank, sizeArray1D.data(), 
                                        inembed, istride, idist, 
                                        onembed, ostride, odist, 
                                        CUFFT_C2C, batch));
    }else{
        HANDLE_FFTERROR(cufftPlanMany(  &mplan, rank, sizeArray2D.data(), 
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
    int I  = threadIdx.x + blockIdx.x*blockDim.x;
    int J  = threadIdx.y + blockIdx.y*blockDim.y;

    if( (I >= size.x) || (J >= size.y) || (blockIdx.z >= size.z)) return;
    
    size_t index = IND(I,J,blockIdx.z,size.x,size.y);    
    data[index]  = - logf( data[index] );
}

__global__ void opt::product_Real_Real(float *a, float *b, float *ans, dim3 sizea, dim3 sizeb)
{
    size_t indexa = opt::getIndex3d(sizea);
    size_t indexb = opt::getIndex3d(sizeb);
    size_t nsize  = sizea.x * sizea.y * sizea.z;
    
    if ( indexa >= nsize ) return;
    ans[indexa] = a[indexa] * b[indexb];	
}

__global__ void opt::product_Complex_Real(cufftComplex *a, float *b, cufftComplex *ans, dim3 sizea, dim3 sizeb)
{
    size_t indexa = opt::getIndex3d(sizea);
    size_t indexb = opt::getIndex3d(sizeb);
    size_t nsize  = sizea.x * sizea.y * sizea.z;
    
    if ( indexa >= nsize ) return;
    ans[indexa].x = a[indexa].x * b[indexb];
    ans[indexa].y = a[indexa].y * b[indexb];	
}

__global__ void opt::product_Complex_Complex(cufftComplex *a, cufftComplex *b, cufftComplex *ans, dim3 sizea, dim3 sizeb)
{
    size_t indexa = opt::getIndex3d(sizea);
    size_t indexb = opt::getIndex3d(sizeb);
    size_t nsize  = sizea.x * sizea.y * sizea.z;
    
    if ( indexa >= nsize ) return;
    ans[indexa] = ComplexMult(a[indexa],b[indexb]);
}