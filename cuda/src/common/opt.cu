#include "common/configs.hpp"
#include "common/complex.hpp"
#include "common/opt.hpp"
#include "common/configs.hpp"
#include "common/logerror.hpp"

/*============================================================================*/
/* namespace opt (in 'inc/commons/opt.hpp') functions definitions */

void opt::MPlanFFT(cufftHandle *mplan, int RANK, dim3 DATASIZE, cufftType FFT_TYPE)
{	
    /* rank:
    Dimensionality of the transform: 1D (1), 2D (2) or 3D (3) cufft */

    /* FFT_TYPE: The transform data type
        CUFFT_R2C, CUFFT_C2R, CUFFT_C2C
    */

    /* Array of size rank, describing the size of each dimension, 
    n[0] being the size of the outermost and 
    n[rank-1] innermost (contiguous) dimension of a transform. */
    int *n = (int *)malloc(RANK); 
    n[0]   = (int)DATASIZE.x;
    
    int idist = DATASIZE.x; /* Input data distance between batches */
    int odist = DATASIZE.x; /* Output data distance between batches */

    if ( FFT_TYPE == CUFFT_C2R)
        idist = idist / 2 + 1;

    if ( FFT_TYPE == CUFFT_R2C)
        odist = odist / 2 + 1;

    int batch  = DATASIZE.y * DATASIZE.z; /* Number of batched executions */

    if ( RANK >= 2 )
        n[1]   = (int)DATASIZE.y;
        batch  = DATASIZE.z;
        idist  *= DATASIZE.y;
        odist  *= DATASIZE.y;
    
    if ( RANK >= 3 ){
        n[2]   = (int)DATASIZE.z;
        idist  *= DATASIZE.z;
        odist  *= DATASIZE.z;
        batch  = 1;
    }
    
    int *inembed = NULL, *onembed = NULL; /* Input/Output size with pitch (ignored for 1D transforms).
    If set to NULL all other advanced data layout parameters are ignored. */
    int istride = 1, ostride = 1; /* Distance between two successive input/output elements. */

    HANDLE_FFTERROR(cufftPlanMany(  mplan, RANK, n, 
                                    inembed, istride, idist, 
                                    onembed, ostride, odist, 
                                    FFT_TYPE, batch));
}

__global__ void setSinCosTable(float *sintable, float *costable, float *angles, int nangles)
{
    size_t k = blockIdx.x*blockDim.x + threadIdx.x;

    if ( (k >= nangles) ) return;

    sintable[k] = __sinf(angles[k]);
    costable[k] = __cosf(angles[k]);
}

void getLog(float *data, dim3 size)
{
    dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 gridBlock( (int)ceil( size.x / threadsPerBlock.x ) + 1, 
                    (int)ceil( size.y / threadsPerBlock.y ) + 1, 
                    (int)ceil( size.z / threadsPerBlock.z ) + 1);

    Klog<<<gridBlock,threadsPerBlock>>>(data, size);
}

static __global__ void Klog(float* data, dim3 size)
{  
    int i  = threadIdx.x + blockIdx.x*blockDim.x;
    int j  = threadIdx.y + blockIdx.y*blockDim.y;
    int k  = threadIdx.z + blockIdx.z*blockDim.z;

    if( (i >= size.x) || (j >= size.y) || (k >= size.z)) return;
    
    size_t index = IND(i,j,k,size.x,size.y);    
    // size_t index = (size_t)(i + j * size.x + k * size.x * size.y);    

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

dim3 setGridBlock(dim3 size){
    dim3 gridBlock( (int)ceil( size.x / TPBX ) + 1, 
                    (int)ceil( size.y / TPBY ) + 1, 
                    (int)ceil( size.z / TPBZ ) + 1);

    return gridBlock;
}
