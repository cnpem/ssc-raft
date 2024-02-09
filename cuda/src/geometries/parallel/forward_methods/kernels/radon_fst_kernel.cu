#include <cufft.h>
#include ".geometries/gp/fst.hpp"

// in case it is compiled with host compiler instead of nvcc:
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufftXt.h>


// Did not work as expected, not sure why:
// // http://www.fftw.org/faq/section3.html#centerorigin
// __global__ void shift_2d_fftw_way(
// 	cufftComplex *c,
// 	size_t sizex, size_t sizey, size_t sizez)
// {
// 	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
// 	size_t j = (index % (sizey*sizex)) / sizex;
// 	size_t i = (index % (sizey*sizex)) % sizex;

// 	if (index < sizex * sizey * sizez)  {
// 		c[index].x *= 1 - 2*((i + j) & 1);
// 		c[index].y *= 1 - 2*((i + j) & 1);
// 	}
// }


__global__ void calc_counts(
	float *sino, float *flat,
    int nrays, int nangles, int nslices)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	size_t slice_idx = idx / (nangles*nrays);
	size_t nray_idx = (idx % (nangles*nrays)) % nrays;
	size_t flat_idx = slice_idx*nrays + nray_idx;
	size_t size = (size_t)nrays * nangles * nslices;
	if (idx < size) {
		if (sino[idx] < 0) {
			sino[idx] = flat[flat_idx];
		} else {
			sino[idx] = flat[flat_idx] * expf(-sino[idx]);
		}
	}
}


__global__ void set_value(
	float *arr,
	float val,
	size_t size)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size) {
		arr[idx] = val;
	}
}


__global__ void calc_reciprocal_element_wise(
	float *arr,
	size_t size)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size) {
		arr[idx] = 1.0/arr[idx];
	}
}

// __global__ void add_element_wise(
// 	float *arr1,
// 	float *arr2,
// 	size_t size)
// {
// 	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
// 	if (idx < size) {
// 		arr1[idx] += arr2[idx];
// 	}
// }

__global__ void multiply_element_wise(
	float *arr1,
	float *arr2,
	size_t size)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size) {
		arr1[idx] *= arr2[idx];
	}
}

__global__ void total_variation_2d(
    float *back,
    float *recon,
    float *backcounts,
    size_t size,
    int nx, int ny, int nz,
    float tv_param)
{
    float curr, xnext, ynext, xprev, yprev, dnext, dprev;
    float sum_diff, sqrt_sum_sq_diff;
    bool ok = false;
    float tv_term = 0.0;
    float mean_abs_diff = 0.0;
    int count_halve = 0, max_halve = 8; // trocar "do{} while();" por "max(epsilon, back[i]-tv);".
    long long int i = blockDim.x * blockIdx.x + threadIdx.x;
    const float TV_EPSILON = 1.0e-9;
    const float TV_MAX_RELATIVE_INTENSITY = 0.05;
    if (    0 < i && i < size-1 
            && 0 < i-nx && i+nx < size-1) {
        curr = recon[i];
        xnext = recon[i+1];
        ynext = recon[i+nx];
        xprev = recon[i-1];
        yprev = recon[i-nx];
        dnext = recon[i+nx-1]; // left "next" diagonal.
        dprev = recon[i-nx+1]; // right "previous" diagonal.
        mean_abs_diff = (fabsf(curr-xnext) + fabsf(curr-ynext) + fabsf(curr-xprev) + fabsf(curr-yprev))/4.0;
        sum_diff = 2*curr - xnext - ynext;
        sqrt_sum_sq_diff = (curr-xnext)*(curr-xnext) + (curr-ynext)*(curr-ynext);
        sqrt_sum_sq_diff = sqrtf(sqrt_sum_sq_diff);
        if (TV_EPSILON < sqrt_sum_sq_diff) {
            tv_term += tv_param * sum_diff / (sqrt_sum_sq_diff * backcounts[i]);
        }
        sqrt_sum_sq_diff = (xprev-curr)*(xprev-curr) + (xprev-dnext)*(xprev-dnext);
        sqrt_sum_sq_diff = sqrtf(sqrt_sum_sq_diff);
        if (TV_EPSILON < sqrt_sum_sq_diff) {
            tv_term += tv_param * (curr-xprev) / (sqrt_sum_sq_diff * backcounts[i]);
        }
        sqrt_sum_sq_diff = (yprev-curr)*(yprev-curr) + (yprev-dprev)*(yprev-dprev);
        sqrt_sum_sq_diff = sqrtf(sqrt_sum_sq_diff);
        if (TV_EPSILON < sqrt_sum_sq_diff) {
            tv_term += tv_param * (curr-yprev) / (sqrt_sum_sq_diff * backcounts[i]);
        }
        do {
            ok =   (0.0 <= (back[i] - tv_term)) 
                && (fabsf(tv_term/back[i]) < TV_MAX_RELATIVE_INTENSITY);
                // && (fabsf(tv_term) < mean_abs_diff/2.0); // INTENSITY_MAX_CONSUMPTION = 1.0 para figuras artisticamente bonitas. 
            if (ok) {
                back[i] -= tv_term;
            }
            tv_term = 0.5 * tv_term;
            count_halve++;
        } while(!ok && count_halve < max_halve);
    }
}


__global__ void scale_data(
	float *data,
	size_t size,
	float scale)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx] *= scale;
	}
}


__global__ void scale_data_complex(
	cufftComplex *data,
	size_t size,
	float scale)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx].x *= scale;
		data[idx].y *= scale;
	}
}

__global__ void scale_data_real_only(
	cufftComplex *data,
	size_t size,
	float scale)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx].x *= scale;
	}
}

__global__ void scale_data_imag_only(
	cufftComplex *data,
	size_t size,
	float scale)
{
	size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx < size) {
		data[idx].y *= scale;
	}
}

// FFT copy input C2C: input data (complex) to the corners of the FFT workspace (complex).
__global__ void copy_to_fft_workspace(
	cufftComplex *workspace, cufftComplex *src,
	int n1, int n2, int n1_src, int n2_src, int blocksize)
{
	size_t index_src = blockIdx.x*blockDim.x + threadIdx.x;
	size_t k =  index_src / (n2_src*n1_src); // k = k_src.
	size_t j_src = (index_src % (n2_src*n1_src)) / n1_src;
	size_t i_src = (index_src % (n2_src*n1_src)) % n1_src;
	size_t pad1 = n1 - n1_src;
	size_t pad2 = n2 - n2_src;
	size_t index = k*n1*n2;

	if (index_src < static_cast<size_t>(blocksize)*n1_src*n2_src) {
		if (i_src < n1_src/2) {
			if (j_src < n2_src/2) {
				index += j_src*n1 + i_src;
				workspace[index] = src[index_src];
			} else {
				index += (j_src + pad2)*n1 + i_src;
				workspace[index] = src[index_src];
			}
		} else {
			if (j_src < n2_src/2) {
				index += j_src*n1 + i_src + pad1;
				workspace[index] = src[index_src];
			} else {
				index += (j_src + pad2)*n1 + i_src + pad1;
				workspace[index] = src[index_src];
			}
		}	
	}
}

// Same as above but R2C instead of C2C:
__global__ void copy_to_fft_workspace_R2C(
	cufftComplex *workspace, float *src,
	int n1, int n2, int n1_src, int n2_src, int blocksize)
{
	size_t index_src = blockIdx.x*blockDim.x + threadIdx.x;
	size_t k =  index_src / (n2_src*n1_src); // k = k_src.
	size_t j_src = (index_src % (n2_src*n1_src)) / n1_src;
	size_t i_src = (index_src % (n2_src*n1_src)) % n1_src;
	size_t pad1 = n1 - n1_src;
	size_t pad2 = n2 - n2_src;
	size_t index = k*n1*n2;

	if (index_src < static_cast<size_t>(blocksize)*n1_src*n2_src) {
		if (i_src < n1_src/2) {
			if (j_src < n2_src/2) {
				index += j_src*n1 + i_src;
				workspace[index].x = src[index_src];
			} else {
				index += (j_src + pad2)*n1 + i_src;
				workspace[index].x = src[index_src];
			}
		} else {
			if (j_src < n2_src/2) {
				index += j_src*n1 + i_src + pad1;
				workspace[index].x = src[index_src];
			} else {
				index += (j_src + pad2)*n1 + i_src + pad1;
				workspace[index].x = src[index_src];
			}
		}	
	}
}

// FFT copy output C2C.
__global__ void copy_from_fft_workspace(
	cufftComplex *workspace, cufftComplex *dst,
	int n1, int n2, int n1_dst, int n2_dst, int blocksize)
{
	size_t index_dst = blockIdx.x*blockDim.x + threadIdx.x;
	size_t k 	 =  index_dst / (n2_dst*n1_dst); // k = k_dst.
	size_t j_dst = (index_dst % (n2_dst*n1_dst)) / n1_dst;
	size_t i_dst = (index_dst % (n2_dst*n1_dst)) % n1_dst;
	size_t pad1 = n1 - n1_dst;
	size_t pad2 = n2 - n2_dst;
	size_t index = k*n1*n2 + (j_dst + pad2/2)*n1 + (pad1/2) + i_dst;

	if (index_dst < static_cast<size_t>(blocksize)*n1_dst*n2_dst) {
		dst[index_dst] = workspace[index];
	}
}

// Same as above but C2R:
__global__ void copy_from_fft_workspace_C2R(
	cufftComplex *workspace, float *dst,
	int n1, int n2, int n1_dst, int n2_dst, int blocksize)
{
	size_t index_dst = blockIdx.x*blockDim.x + threadIdx.x;
	size_t k 	 =  index_dst / (n2_dst*n1_dst); // k = k_dst.
	size_t j_dst = (index_dst % (n2_dst*n1_dst)) / n1_dst;
	size_t i_dst = (index_dst % (n2_dst*n1_dst)) % n1_dst;
	size_t pad1 = n1 - n1_dst;
	size_t pad2 = n2 - n2_dst;
	size_t index = k*n1*n2 + (j_dst + pad2/2)*n1 + (pad1/2) + i_dst;

	if (index_dst < static_cast<size_t>(blocksize)*n1_dst*n2_dst) {
		dst[index_dst] = workspace[index].x;
	}
}


// My way for 1 dimensional fft shift:
__global__ void shift_1d(
	cufftComplex *c,
	size_t sizex, size_t sizey, size_t sizez)
{
	cufftComplex temp;
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	size_t i = (index % (sizey*sizex)) % sizex;

	if (index < sizex*sizey*sizez / 2) {
		if (sizex/2 <= i) {
			index += -sizex/2 + sizex*sizey*sizez/2;
			temp = c[index];
			c[index] = c[index + sizex/2];
			c[index + sizex/2] = temp;
		} else {
			temp = c[index];
			c[index] = c[index + sizex/2];
			c[index + sizex/2] = temp;
		}
	}
}

// As above but for real/non-complex float data.
__global__ void shift_1d_real(
	float *data,
	size_t sizex, size_t sizey, size_t sizez)
{
	float temp;
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	size_t i = (index % (sizey*sizex)) % sizex;

	if (index < sizex*sizey*sizez / 2) {
		if (sizex/2 <= i) {
			index += -sizex/2 + sizex*sizey*sizez/2;
			temp = data[index];
			data[index] = data[index + sizex/2];
			data[index + sizex/2] = temp;
		} else {
			temp = data[index];
			data[index] = data[index + sizex/2];
			data[index + sizex/2] = temp;
		}
	}
}


// Payola's 2 dimensional fft shift kernel:
__global__ void fftshift2d(cufftComplex *c, size_t sizex, size_t sizey, size_t sizez)
{
	cufftComplex temp;
	size_t shift; 
	size_t N = ( (sizex * sizey) + sizex ) / 2;	
	size_t M = ( (sizex * sizey) - sizex ) / 2;	
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	size_t k =  index / (sizey*sizex);
	size_t j = (index % (sizey*sizex)) / sizex;
	size_t i = (index % (sizey*sizex)) % sizex;

	if ( (i >= sizex) || (j >= sizey) || (k >= sizez) ) return;
	
	if ( i < ( sizex / 2 ) ){	
		if ( j < ( sizey / 2 ) ){	
			// index = sizex * (k*sizey + j)  + i;
			shift = index + N;
			temp 	 = c[index];	
			c[index] = c[shift];	
			c[shift] = temp;
		}
	} else {
		if ( j < ( sizey / 2 ) ){
			// index = sizex * (k*sizey + j)  + i;
			shift = index + M;
			temp 	 = c[index];	
			c[index] = c[shift];	
			c[shift] = temp;
		}
	}
}

// Payola's 2 dimensional fft shift kernel for real/non-complex float data:
__global__ void fftshift2d_real(float *data, size_t sizex, size_t sizey, size_t sizez)
{
	float temp;
	size_t shift; 
	size_t N = ( (sizex * sizey) + sizex ) / 2;	
	size_t M = ( (sizex * sizey) - sizex ) / 2;	
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	size_t k =  index / (sizey*sizex);
	size_t j = (index % (sizey*sizex)) / sizex;
	size_t i = (index % (sizey*sizex)) % sizex;

	if ( (i >= sizex) || (j >= sizey) || (k >= sizez) ) return;
	
	if ( i < ( sizex / 2 ) ){	
		if ( j < ( sizey / 2 ) ){	
			// index = sizex * (k*sizey + j)  + i;
			shift = index + N;
			temp 	 = data[index];	
			data[index] = data[shift];	
			data[shift] = temp;
		}
	} else {
		if ( j < ( sizey / 2 ) ){
			// index = sizex * (k*sizey + j)  + i;
			shift = index + M;
			temp 	 = data[index];	
			data[index] = data[shift];	
			data[shift] = temp;
		}
	}
}

// Payola's -MODIFIED FOR BETTER PERFORMANCE- 2 dimensional fft shift kernel - CURRENTLY NOT WORKING.
__global__ void fftshift2d_v2(cufftComplex *c, size_t sizex, size_t sizey, size_t sizez)
{
	cufftComplex temp;
	size_t shift; 
	size_t N = ( (sizex * sizey) + sizex ) / 2;	
	size_t M = ( (sizex * sizey) - sizex ) / 2;	
	size_t index = blockIdx.x*blockDim.x + threadIdx.x;
	size_t k =  index / (sizey*sizex);
	size_t j = (index % (sizey*sizex)) / sizex;
	size_t i = (index % (sizey*sizex)) % sizex;

	if (k >= sizez) {
		return;
	}
	if ( i < ( sizex / 2 ) ) {	
		if ( j < ( sizey / 2 ) ) {	
			shift = index + N;
			temp 	 = c[index];
			c[index] = c[shift];
			c[shift] = temp;
		} else {
			index += -j*sizex + sizex*sizey*sizez/2;
			shift = index + N;
			temp 	 = c[index];
			c[index] = c[shift];
			c[shift] = temp;
		}
	} else {
		if ( j < ( sizey / 2 ) ) {
			shift = index + M;
			temp 	 = c[index];
			c[index] = c[shift];
			c[shift] = temp;
		} else {
			index += -j*sizex + sizex*sizey*sizez/2;
			shift = index + M;
			temp 	 = c[index];
			c[index] = c[shift];
			c[shift] = temp;
		}
	}
}

__global__ void cartesian2polar_nn(
	cufftComplex *ft_recon,
	cufftComplex *ft_sino,
	float *angles,
	size_t nrays, size_t nangles, size_t blocksize)
{
	size_t sino_size = nangles*nrays;
	size_t sino_idx = blockIdx.x * blockDim.x + threadIdx.x;
	float ang = angles[sino_idx / nrays];
	float rho = (sino_idx % nrays) - nrays/2.0; // rho_idx - rho_max.
	float xpos = cosf(ang)*rho;
	float ypos = sinf(ang)*rho;
	int x_idx = __float2int_rn(xpos) + (int)nrays/2;
	int y_idx = __float2int_rn(ypos) + (int)nrays/2;
	if (sino_idx < sino_size && 
		(blocksize-1)*(nrays*nrays) + y_idx*nrays + x_idx < blocksize*(nrays*nrays)) {
		for (size_t slice_idx = 0; slice_idx < blocksize; ++slice_idx) {
			ft_sino[slice_idx*sino_size + sino_idx] = 
				ft_recon[slice_idx*(nrays*nrays) + y_idx*nrays + x_idx];
		}
	}
}

// DO NOT USE: Bilinear interpolation fails when xd = xu (e.g., ang = pi/2) or when yd = yu (e.g., ang = 0).
// Use, instead, 'cartesian2polar_bi_v1' or 'cartesian2polar_bi_v2' bellow.
__global__ void cartesian2polar_bi_v0(
	cufftComplex *ft_recon,
	cufftComplex *ft_sino,
	float *angles,
	size_t nrays, size_t nangles, size_t blocksize)
{
	size_t sino_size = nangles*nrays;
	size_t sino_idx = blockIdx.x * blockDim.x + threadIdx.x;
	float ang = angles[sino_idx / nrays];
	float rho = 0.5 + (sino_idx % nrays) - nrays/2.0; // rho_idx - rho_max.
	float xpos = cosf(ang)*rho;
	float ypos = sinf(ang)*rho;
	size_t xd_idx = __float2int_rd(xpos) + nrays/2;
	size_t xu_idx = __float2int_ru(xpos) + nrays/2;
	size_t yd_idx = __float2int_rd(ypos) + nrays/2;
	size_t yu_idx = __float2int_ru(ypos) + nrays/2;
	cuFloatComplex fdd, fdu, fud, fuu; // fdd = f(xd_idx, yd_idx), fdu = f(xd_idx, yu_idx) ...
	float xd_pos, xu_pos, yd_pos, yu_pos;

	if (sino_idx < sino_size) {
		for (size_t slice_idx = 0; slice_idx < blocksize; ++slice_idx) {
			fdd = ft_recon[slice_idx*(nrays*nrays) + yd_idx*nrays + xd_idx];
			fdu = ft_recon[slice_idx*(nrays*nrays) + yu_idx*nrays + xd_idx];
			fud = ft_recon[slice_idx*(nrays*nrays) + yd_idx*nrays + xu_idx];
			fuu = ft_recon[slice_idx*(nrays*nrays) + yu_idx*nrays + xu_idx];
			xd_pos = (float) xd_idx - nrays/2;
			xu_pos = (float) xu_idx - nrays/2;
			yd_pos = (float) yd_idx - nrays/2;
			yu_pos = (float) yu_idx - nrays/2;
			ft_sino[slice_idx*sino_size + sino_idx].x = 
				(yu_pos-ypos)*((xu_pos-xpos)*fdd.x + (xpos-xd_pos)*fud.x) +
				(ypos-yd_pos)*((xu_pos-xpos)*fdu.x + (xpos-xd_pos)*fuu.x);
			ft_sino[slice_idx*sino_size + sino_idx].y = 
				(yu_pos-ypos)*((xu_pos-xpos)*fdd.y + (xpos-xd_pos)*fud.y) +
				(ypos-yd_pos)*((xu_pos-xpos)*fdu.y + (xpos-xd_pos)*fuu.y);
		}
	}
}

// A correction for when xd = xu and yd = yu.
__global__ void cartesian2polar_bi_v1(
	cufftComplex *ft_recon,
	cufftComplex *ft_sino,
	float *angles,
	size_t nrays, size_t nangles, size_t blocksize)
{
	size_t sino_size = nangles*nrays;
	size_t sino_idx = blockIdx.x * blockDim.x + threadIdx.x;
	float ang = angles[sino_idx / nrays];
	float rho = 0.0 + (sino_idx % nrays) - nrays/2.0; // rho_idx - rho_max.
	float xpos = cosf(ang)*rho;
	float ypos = sinf(ang)*rho;
	size_t xd_idx = __float2int_rd(xpos) + nrays/2;
	size_t xu_idx = __float2int_ru(xpos) + nrays/2;
	size_t yd_idx = __float2int_rd(ypos) + nrays/2;
	size_t yu_idx = __float2int_ru(ypos) + nrays/2;
	cuFloatComplex fdd, fdu, fud, fuu; // fdd = f(xd_idx, yd_idx), fdu = f(xd_idx, yu_idx) ...
	float xd_pos, xu_pos, yd_pos, yu_pos;

    if (sino_idx < sino_size) {
        for (size_t slice_idx = 0; slice_idx < blocksize; ++slice_idx) {
			fdd = ft_recon[slice_idx*(nrays*nrays) + yd_idx*nrays + xd_idx];
			fdu = ft_recon[slice_idx*(nrays*nrays) + yu_idx*nrays + xd_idx];
			fud = ft_recon[slice_idx*(nrays*nrays) + yd_idx*nrays + xu_idx];
			fuu = ft_recon[slice_idx*(nrays*nrays) + yu_idx*nrays + xu_idx];

            xd_pos = (float) xd_idx - nrays/2;
      		xu_pos = (float) xu_idx - nrays/2;
       		yd_pos = (float) yd_idx - nrays/2;
       		yu_pos = (float) yu_idx - nrays/2;

            if (xd_idx == xu_idx && yd_idx == yu_idx) { // fdd == fud == fdu == fuu.
                ft_sino[slice_idx*sino_size + sino_idx].x = fdd.x;
                ft_sino[slice_idx*sino_size + sino_idx].y = fdd.y;
            } else if (xd_idx == xu_idx) { // fdd == fud, fdu == fuu.
        		ft_sino[slice_idx*sino_size + sino_idx].x = 
        			(yu_pos-ypos)*fdd.x + (ypos-yd_pos)*fuu.x;
        		ft_sino[slice_idx*sino_size + sino_idx].y = 
        			(yu_pos-ypos)*fdd.y + (ypos-yd_pos)*fuu.y;
            } else if (yd_idx == yu_idx) { // fdd == fdu, fud == fuu.
        		ft_sino[slice_idx*sino_size + sino_idx].x = 
        			(xu_pos-xpos)*fdd.x + (xpos-xd_pos)*fuu.x;
        		ft_sino[slice_idx*sino_size + sino_idx].y = 
        			(xu_pos-xpos)*fdd.y + (xpos-xd_pos)*fuu.y;
            } else {

//              xd_pos = (float) xd_idx - nrays/2;
//      		xu_pos = (float) xu_idx - nrays/2;
//      		yd_pos = (float) yd_idx - nrays/2;
//      		yu_pos = (float) yu_idx - nrays/2;

        		ft_sino[slice_idx*sino_size + sino_idx].x = 
        			(yu_pos-ypos)*((xu_pos-xpos)*fdd.x + (xpos-xd_pos)*fud.x) +
        			(ypos-yd_pos)*((xu_pos-xpos)*fdu.x + (xpos-xd_pos)*fuu.x);
        		ft_sino[slice_idx*sino_size + sino_idx].y = 
        			(yu_pos-ypos)*((xu_pos-xpos)*fdd.y + (xpos-xd_pos)*fud.y) +
        			(ypos-yd_pos)*((xu_pos-xpos)*fdu.y + (xpos-xd_pos)*fuu.y);
            }
		}
	}
}

// Another correction approach for when xd = xu or yd = yu.
// Here we don't use 'if's to avoid branch divergence. Instead, the ajustments are performed on the data.
__global__ void cartesian2polar_bi_v2(
	cufftComplex *ft_recon,
	cufftComplex *ft_sino,
	float *angles,
	size_t nrays, size_t nangles, size_t blocksize)
{
	size_t sino_size = nangles*nrays;
	size_t sino_idx = blockIdx.x * blockDim.x + threadIdx.x;
	float ang = angles[sino_idx / nrays];
	float rho = 0.5 + (sino_idx % nrays) - nrays/2.0; // rho_idx - rho_max.
	float xpos = cosf(ang)*rho;
	float ypos = sinf(ang)*rho;
	size_t xd_idx = __float2int_rd(xpos) + nrays/2;
	size_t xu_idx = __float2int_ru(xpos) + nrays/2;
	size_t yd_idx = __float2int_rd(ypos) + nrays/2;
	size_t yu_idx = __float2int_ru(ypos) + nrays/2;
	cuFloatComplex fdd, fdu, fud, fuu; // fdd = f(xd_idx, yd_idx), fdu = f(xd_idx, yu_idx) ...
	float xd_pos, xu_pos, yd_pos, yu_pos;
	float xd_diff, xu_diff, yd_diff, yu_diff;
	float xd_ratio, xu_ratio, yd_ratio, yu_ratio;
	float x_diff, y_diff;

	if (sino_idx < sino_size) {
		for (size_t slice_idx = 0; slice_idx < blocksize; ++slice_idx) {
			fdd = ft_recon[slice_idx*(nrays*nrays) + yd_idx*nrays + xd_idx];
			fdu = ft_recon[slice_idx*(nrays*nrays) + yu_idx*nrays + xd_idx];
			fud = ft_recon[slice_idx*(nrays*nrays) + yd_idx*nrays + xu_idx];
			fuu = ft_recon[slice_idx*(nrays*nrays) + yu_idx*nrays + xu_idx];
			xd_pos = (float) xd_idx - nrays/2;
			xu_pos = (float) xu_idx - nrays/2;
			yd_pos = (float) yd_idx - nrays/2;
			yu_pos = (float) yu_idx - nrays/2;
			xd_diff = (xpos - xd_pos) + 1e-12;
			xu_diff = (xu_pos - xpos) + 1e-12;
			yd_diff = (ypos - yd_pos) + 1e-12;
			yu_diff = (yu_pos - ypos) + 1e-12;
			x_diff = (xu_pos - xd_pos) + 2e-12;
			y_diff = (yu_pos - yd_pos) + 2e-12;
			xd_ratio = xd_diff / x_diff;
			xu_ratio = xu_diff / x_diff;
			yd_ratio = yd_diff / y_diff;
			yu_ratio = yu_diff / y_diff;
			ft_sino[slice_idx*sino_size + sino_idx].x = 
				yu_ratio*(xu_ratio*fdd.x + xd_ratio*fud.x) +
				yd_ratio*(xu_ratio*fdu.x + xd_ratio*fuu.x);
			ft_sino[slice_idx*sino_size + sino_idx].y = 
				yu_ratio*(xu_ratio*fdd.y + xd_ratio*fud.y) +
				yd_ratio*(xu_ratio*fdu.y + xd_ratio*fuu.y);
		}
	}
}
