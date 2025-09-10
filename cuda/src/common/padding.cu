#include <cstdio>
#include "common/configs.hpp"
#include "common/opt.hpp"

/*============================================================================*/
/* namespace opt (in 'inc/commons/opt.hpp') functions definitions */

int opt::compute_padding_size(int size, int pad)
{
    return int( (pad * size) / 100 );
}

int opt::compute_dim_padded(int size, int pad)
{
    return size + 2 * opt::compute_padding_size(size, pad);
}

dim3 opt::compute_size_padded(dim3 size, dim3 pad)
{
    dim3 size_padded(   
                    size.x + 2 * opt::compute_dim_padded(size.x, pad.x),
                    size.y + 2 * opt::compute_dim_padded(size.y, pad.y),
                    size.z + 2 * opt::compute_dim_padded(size.z, pad.z) 
                    );

    return size_padded;
}

__global__ void opt::paddR2C(float *in, cufftComplex *outpadded, 
int padding_mode, dim3 size, dim3 pad)
{
    int Npadx = PDIM(size.x, pad.x); 
    int Npady = PDIM(size.y, pad.y);    

    int padx  = PADS(size.x, pad.x); 
    int pady  = PADS(size.y, pad.y); 

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii    = (int)( i - padx );
    int jj    = (int)( j - pady );

    long long int index  = IND(ii,jj,k,size.x,size.y);
    long long int indpad = IND( i, j,k, Npadx, Npady); 

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    outpadded[indpad].y = 0.0;

    opt::PaddingMode mode = static_cast<opt::PaddingMode>(padding_mode);

    switch(mode){
        case opt::PaddingMode::zero:
            /* Zero padding */
            outpadded[indpad].x = 0.0; 
        break;        
        case opt::PaddingMode::ones:
            /* Ones padding */
            outpadded[indpad].x = 1.0; 
        break;
        case opt::PaddingMode::edge:
            /* Edge padding */
            if ( ( i <=          padx ) && ( j <=          pady )) outpadded[indpad].x = in[IND(           0,           0,k,size.x,size.y)];
            if ( ( i >= size.x + padx ) && ( j <=          pady )) outpadded[indpad].x = in[IND((size.x - 1),           0,k,size.x,size.y)];
            if ( ( i <=          padx ) && ( j >= size.y + pady )) outpadded[indpad].x = in[IND(           0,(size.y - 1),k,size.x,size.y)];
            if ( ( i >= size.x + padx ) && ( j >= size.y + pady )) outpadded[indpad].x = in[IND((size.x - 1),(size.y - 1),k,size.x,size.y)];
        
            if ( ( j > pady ) && ( j < size.y + pady ) ){
                if ( ( i <=          padx ) ) outpadded[indpad].x = in[IND(           0, jj,k,size.x,size.y)];
                if ( ( i >= size.x + padx ) ) outpadded[indpad].x = in[IND((size.x - 1), jj,k,size.x,size.y)];
            }

            if ( ( i > padx ) && ( i < size.x + padx ) ){
                if ( ( j <=          pady ) ) outpadded[indpad].x = in[IND(ii,           0,k,size.x,size.y)];
                if ( ( j >= size.y + pady ) ) outpadded[indpad].x = in[IND(ii,(size.y - 1),k,size.x,size.y)];
            }
        break;
        default:
            /* Zero padding */
            outpadded[indpad].x = 0.0; 
        break;
    }
    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) ) return;
    outpadded[indpad].x = in[index];
}

__global__ void opt::paddC2C(cufftComplex *in, cufftComplex *outpadded,
int padding_mode, dim3 size, dim3 pad)
{
    int Npadx = PDIM(size.x, pad.x); 
    int Npady = PDIM(size.y, pad.y);    

    int padx  = PADS(size.x, pad.x); 
    int pady  = PADS(size.y, pad.y); 

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii    = (int)( i - padx );
    int jj    = (int)( j - pady );

    long long int index  = IND(ii,jj,k,size.x,size.y);
    long long int indpad = IND( i, j,k, Npadx, Npady); 

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    opt::PaddingMode mode = static_cast<opt::PaddingMode>(padding_mode);

    switch(mode){
        case opt::PaddingMode::zero:
            /* Zero padding */
            outpadded[indpad].x = 0.0; 
            outpadded[indpad].y = 0.0; 
        break;
        case opt::PaddingMode::ones:
            /* Ones padding */
            outpadded[indpad].x = 1.0;
            outpadded[indpad].y = 0.0; 
        break;
        case opt::PaddingMode::edge:
            /* Edge padding */
            if ( ( i <=          padx ) && ( j <=          pady )) outpadded[indpad].x = in[IND(           0,           0,k,size.x,size.y)].x;
            if ( ( i >= size.x + padx ) && ( j <=          pady )) outpadded[indpad].x = in[IND((size.x - 1),           0,k,size.x,size.y)].x;
            if ( ( i <=          padx ) && ( j >= size.y + pady )) outpadded[indpad].x = in[IND(           0,(size.y - 1),k,size.x,size.y)].x;
            if ( ( i >= size.x + padx ) && ( j >= size.y + pady )) outpadded[indpad].x = in[IND((size.x - 1),(size.y - 1),k,size.x,size.y)].x;
        
            if ( ( i <=          padx ) && ( j <=          pady )) outpadded[indpad].y = in[IND(           0,           0,k,size.x,size.y)].y;
            if ( ( i >= size.x + padx ) && ( j <=          pady )) outpadded[indpad].y = in[IND((size.x - 1),           0,k,size.x,size.y)].y;
            if ( ( i <=          padx ) && ( j >= size.y + pady )) outpadded[indpad].y = in[IND(           0,(size.y - 1),k,size.x,size.y)].y;
            if ( ( i >= size.x + padx ) && ( j >= size.y + pady )) outpadded[indpad].y = in[IND((size.x - 1),(size.y - 1),k,size.x,size.y)].y;
        
            if ( ( j > pady ) && ( j < size.y + pady ) ){
                if ( ( i <=          padx ) ) outpadded[indpad].x = in[IND(           0, jj,k,size.x,size.y)].x;
                if ( ( i >= size.x + padx ) ) outpadded[indpad].x = in[IND((size.x - 1), jj,k,size.x,size.y)].x;
                if ( ( i <=          padx ) ) outpadded[indpad].y = in[IND(           0, jj,k,size.x,size.y)].y;
                if ( ( i >= size.x + padx ) ) outpadded[indpad].y = in[IND((size.x - 1), jj,k,size.x,size.y)].y;
            }

            if ( ( i > padx ) && ( i < size.x + padx ) ){
                if ( ( j <=          pady ) ) outpadded[indpad].x = in[IND(ii,           0,k,size.x,size.y)].x;
                if ( ( j >= size.y + pady ) ) outpadded[indpad].x = in[IND(ii,(size.y - 1),k,size.x,size.y)].x;
                if ( ( j <=          pady ) ) outpadded[indpad].y = in[IND(ii,           0,k,size.x,size.y)].y;
                if ( ( j >= size.y + pady ) ) outpadded[indpad].y = in[IND(ii,(size.y - 1),k,size.x,size.y)].y;
            }
        break;
        default:
            /* Zero padding */
            outpadded[indpad].x = 0.0; 
            outpadded[indpad].y = 0.0; 
        break;
    }
    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) ) return;
    outpadded[indpad].x = in[index].x;
    outpadded[indpad].y = in[index].y;
}

__global__ void opt::paddC2R(cufftComplex *in, float *outpadded,
int padding_mode, dim3 size, dim3 pad)
{
    int Npadx = PDIM(size.x, pad.x); 
    int Npady = PDIM(size.y, pad.y);    

    int padx  = PADS(size.x, pad.x); 
    int pady  = PADS(size.y, pad.y); 

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii    = (int)( i - padx );
    int jj    = (int)( j - pady );

    long long int index  = IND(ii,jj,k,size.x,size.y);
    long long int indpad = IND( i, j,k, Npadx, Npady);

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    opt::PaddingMode mode = static_cast<opt::PaddingMode>(padding_mode);

    switch(mode){
        case opt::PaddingMode::zero:
            /* Zero padding */
            outpadded[indpad] = 0.0; 
        break;
        case opt::PaddingMode::ones:
            /* Ones padding */
            outpadded[indpad] = 1.0;
        break;
        case opt::PaddingMode::edge:
            /* Edge padding */
            if ( ( i <=          padx ) && ( j <=          pady )) outpadded[indpad] = in[IND(           0,           0,k,size.x,size.y)].x;
            if ( ( i >= size.x + padx ) && ( j <=          pady )) outpadded[indpad] = in[IND((size.x - 1),           0,k,size.x,size.y)].x;
            if ( ( i <=          padx ) && ( j >= size.y + pady )) outpadded[indpad] = in[IND(           0,(size.y - 1),k,size.x,size.y)].x;
            if ( ( i >= size.x + padx ) && ( j >= size.y + pady )) outpadded[indpad] = in[IND((size.x - 1),(size.y - 1),k,size.x,size.y)].x;
        
            if ( ( j > pady ) && ( j < size.y + pady ) ){
                if ( ( i <=          padx ) ) outpadded[indpad] = in[IND(           0, jj,k,size.x,size.y)].x;
                if ( ( i >= size.x + padx ) ) outpadded[indpad] = in[IND((size.x - 1), jj,k,size.x,size.y)].x;
            }
            if ( ( i > padx ) && ( i < size.x + padx ) ){
                if ( ( j <=          pady ) ) outpadded[indpad] = in[IND(ii,           0,k,size.x,size.y)].x;
                if ( ( j >= size.y + pady ) ) outpadded[indpad] = in[IND(ii,(size.y - 1),k,size.x,size.y)].x;
            }
        break;
        default:
            /* Zero padding */
            outpadded[indpad] = 0.0; 
        break;
    }
    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) ) return;
    outpadded[indpad] = in[index].x;
}

__global__ void opt::paddR2R(float *in, float *outpadded,
int padding_mode, dim3 size, dim3 pad)
{
    int Npadx = PDIM(size.x, pad.x); 
    int Npady = PDIM(size.y, pad.y);    

    int padx  = PADS(size.x, pad.x); 
    int pady  = PADS(size.y, pad.y); 

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii    = (int)( i - padx );
    int jj    = (int)( j - pady );

    long long int index  = IND(ii,jj,k,size.x,size.y);
    long long int indpad = IND( i, j,k, Npadx, Npady);

    if ( (i >= Npadx) || (j >= Npady) || (k >= size.z) ) return;

    opt::PaddingMode mode = static_cast<opt::PaddingMode>(padding_mode);

    switch(mode){
        case opt::PaddingMode::zero:
            /* Zero padding */
            outpadded[indpad] = 0.0; 
        break;
        case opt::PaddingMode::ones:
            /* Ones padding */
            outpadded[indpad] = 1.0;
        break;
        case opt::PaddingMode::edge:
            /* Edge padding */
            if ( ( i <=          padx ) && ( j <=          pady )) outpadded[indpad] = in[IND(           0,           0,k,size.x,size.y)];
            if ( ( i >= size.x + padx ) && ( j <=          pady )) outpadded[indpad] = in[IND((size.x - 1),           0,k,size.x,size.y)];
            if ( ( i <=          padx ) && ( j >= size.y + pady )) outpadded[indpad] = in[IND(           0,(size.y - 1),k,size.x,size.y)];
            if ( ( i >= size.x + padx ) && ( j >= size.y + pady )) outpadded[indpad] = in[IND((size.x - 1),(size.y - 1),k,size.x,size.y)];
        
            if ( ( j > pady ) && ( j < size.y + pady ) ){
                if ( ( i <=          padx ) ) outpadded[indpad] = in[IND(           0, jj,k,size.x,size.y)];
                if ( ( i >= size.x + padx ) ) outpadded[indpad] = in[IND((size.x - 1), jj,k,size.x,size.y)];
            }
            if ( ( i > padx ) && ( i < size.x + padx ) ){
                if ( ( j <=          pady ) ) outpadded[indpad] = in[IND(ii,           0,k,size.x,size.y)];
                if ( ( j >= size.y + pady ) ) outpadded[indpad] = in[IND(ii,(size.y - 1),k,size.x,size.y)];
            }
        break;
        default:
            /* Zero padding */
            outpadded[indpad] = 0.0; 
        break;
    }
    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) ) return;
    outpadded[indpad] = in[index];
}

__global__ void opt::remove_paddC2R(cufftComplex *inpadded, float *out, 
dim3 size, dim3 pad)
{
    int Npadx = PDIM(size.x, pad.x); 
    int Npady = PDIM(size.y, pad.y);    

    int padx  = PADS(size.x, pad.x); 
    int pady  = PADS(size.y, pad.y); 

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii    = (int)( i - padx );
    int jj    = (int)( j - pady );

    long long int index  = IND(ii,jj,k,size.x,size.y);
    long long int indpad = IND( i, j,k, Npadx, Npady);
    
    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index] = inpadded[indpad].x;
}

__global__ void opt::remove_paddC2C(cufftComplex *inpadded, cufftComplex *out, 
dim3 size, dim3 pad)
{
    int Npadx = PDIM(size.x, pad.x); 
    int Npady = PDIM(size.y, pad.y);    

    int padx  = PADS(size.x, pad.x); 
    int pady  = PADS(size.y, pad.y); 

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii    = (int)( i - padx );
    int jj    = (int)( j - pady );

    long long int index  = IND(ii,jj,k,size.x,size.y);
    long long int indpad = IND( i, j,k, Npadx, Npady);

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index].x = inpadded[indpad].x;
    out[index].y = inpadded[indpad].y; 
}

__global__ void opt::remove_paddR2C(float *inpadded, cufftComplex *out, 
dim3 size, dim3 pad)
{
    int Npadx = PDIM(size.x, pad.x); 
    int Npady = PDIM(size.y, pad.y);    

    int padx  = PADS(size.x, pad.x); 
    int pady  = PADS(size.y, pad.y); 

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii    = (int)( i - padx );
    int jj    = (int)( j - pady );

    long long int index  = IND(ii,jj,k,size.x,size.y);
    long long int indpad = IND( i, j,k, Npadx, Npady);

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index].x = inpadded[indpad];         
}

__global__ void opt::remove_paddR2R(float *inpadded, float *out, 
dim3 size, dim3 pad)
{
    int Npadx = PDIM(size.x, pad.x); 
    int Npady = PDIM(size.y, pad.y);    

    int padx  = PADS(size.x, pad.x); 
    int pady  = PADS(size.y, pad.y); 

    int i     = blockIdx.x*blockDim.x + threadIdx.x;
    int j     = blockIdx.y*blockDim.y + threadIdx.y;
    int k     = blockIdx.z*blockDim.z + threadIdx.z;

    int ii    = (int)( i - padx );
    int jj    = (int)( j - pady );

    long long int index  = IND(ii,jj,k,size.x,size.y);
    long long int indpad = IND( i, j,k, Npadx, Npady);

    if ( (ii < 0) || (ii >= size.x) || (jj < 0) || (jj >= size.y) || (k >= size.z) ) return;

    out[index] = inpadded[indpad];   
}


