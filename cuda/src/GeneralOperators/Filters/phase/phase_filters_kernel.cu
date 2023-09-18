#include "../../../../inc/include.h"
#include "../../../../inc/common/types.hpp"
#include "../../../../inc/common/kernel_operators.hpp"
#include "../../../../inc/common/complex.hpp"
#include "../../../../inc/common/operations.hpp"
#include "../../../../inc/common/logerror.hpp"

inline __device__ cufftComplex ComplexMult(cufftComplex a, cufftComplex b){ cufftComplex ans; ans.x = a.x * b.x - a.y * b.y; ans.y = a.x * b.y + a.y * b.x; return ans; }

extern "C" {

	void apply_filter(PAR param, float *data, float *kernel, float *ans, cufftComplex *dataPadded, size_t sizex, size_t sizey, size_t sizez)
	{
        zeropadding<<<param.Grd,param.BT>>>(data, dataPadded, sizex, sizey, sizez, param.padx, param.pady);
        HANDLE_FFTERROR(cufftExecC2C(param.mplan, dataPadded, dataPadded, CUFFT_FORWARD));
        CConvolve<<<param.Grd,param.BT>>>(dataPadded, kernel, dataPadded, param.Npadx, param.Npady, sizez);	
        HANDLE_FFTERROR(cufftExecC2C(param.mplan, dataPadded, dataPadded, CUFFT_INVERSE));
        fftNormalize<<<param.Grd,param.BT>>>(dataPadded, param.Npadx, param.Npady, sizez);
        recuperate_zeropadding<<<param.Grd,param.BT>>>(dataPadded, ans, sizex, sizey, sizez, param.padx, param.pady);
	}

    float find_matrix_max(float *matrix, size_t sizex, size_t sizey)
    {
        size_t i,j,ind;
        float maximum = 0;

        for (i = 0; i < sizex; i++){
            for (j = 0; j < sizey; j++){
                
                ind = i + j * sizex;

                maximum = MAX(matrix[ind],maximum);
            }
        }
        return maximum;
    }

    void print_matrix(float *matrix, size_t sizex, size_t sizey)
    {
        size_t i,j,ind;

        for (j = 0; j < sizey; j++){
            for (i = 0; i < sizex; i++){

                ind = i + j * sizex;

                printf("%e ",matrix[ind]);
            }
            printf("\n");
        }
        printf("\n");
    }

    __global__ void CConvolve(cufftComplex *a, float *b, cufftComplex *ans, size_t sizex, size_t sizey, size_t sizez)
    {
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t ind   = sizex * j + i;
        size_t index = sizey * k * sizex + ind;

        if ( (i >= sizex) || (j >= sizey) || (k >= sizez) ) return;
        ans[index].x = a[index].x * b[ind];	
        ans[index].y = a[index].y * b[ind];	
    }

    __global__ void fftNormalize(cufftComplex *c, size_t sizex, size_t sizey, size_t sizez)
    {
        int N = ( sizex * sizey );	
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t index = sizex * (k*sizey + j)  + i;
        
        if ( (i >= sizex) || (j >= sizey) || (k >= sizez) ) return;
        
        c[index].x /= N; c[index].y /= N; 
    }

    __global__ void Normalize(float *a, float b, size_t sizex, size_t sizey, size_t sizez)
    {
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        size_t index = sizex * (k*sizey + j)  + i;
        
        if ( (i >= sizex) || (j >= sizey) || (k >= sizez) ) return;
        
        a[index] /= b; 
    }

    __global__ void fftshiftKernel(float *c, size_t sizex, size_t sizey, size_t sizez)
    {
        int shift, N = ( (sizex * sizey) + sizex ) / 2 ;	
        int M = ( (sizex * sizey) - sizex ) / 2 ;	
        float temp;
        size_t i = blockIdx.x*blockDim.x + threadIdx.x;
        size_t j = blockIdx.y*blockDim.y + threadIdx.y;
        size_t k = blockIdx.z*blockDim.z + threadIdx.z;
        int index; 

        if ( (i >= sizex) || (j >= sizey) || (k >= sizez) ) return;
        
        if ( i < ( sizex / 2 ) ){	
            if ( j < ( sizey / 2 ) ){	
                index = sizex * (k*sizey + j)  + i;
                shift = index + N;
                temp 	 = c[index];	
                c[index] = c[shift];	
                c[shift] = temp;
            }
        }else{
            if ( j < ( sizey / 2 ) ){
                index = sizex * (k*sizey + j)  + i;
                shift = index + M;
                temp 	 = c[index];	
                c[index] = c[shift];	
                c[shift] = temp;
            }
        }
    }

}