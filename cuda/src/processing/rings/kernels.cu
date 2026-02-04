#include "processing/rings_all_stripes.cuh"

__forceinline__ __device__ 
void _idx_to_ijk(int idx, int *i, int *j, int *k, int Nj, int Nk ){
    *k = idx % Nk;
    *j = (idx / Nk) % Nj;
    *i = (idx / Nk) / Nj;
}


// Reorders: tomo[i, jj, k] = sorted_tomo[i, j, k]
// Shapes (C-order): (n_slices, n_angles, n_length), k is fastest
__global__ void _sino_reorder_from_sort(float* __restrict__ tomo, const float* __restrict__ sorted_tomo, const int* __restrict__ indexes, int n_slices, int n_angles, int n_length){
    // Thread -> coordinates (no div/mod)
    const int k = blockIdx.x * blockDim.x + threadIdx.x; // fastest
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= n_slices || j >= n_angles || k >= n_length) return;

    const size_t base_i  = (size_t)i * (size_t)n_angles * (size_t)n_length;
    const size_t base_ij = base_i + (size_t)j * (size_t)n_length;

    const int   jj  = __ldg(&indexes[base_ij + (size_t)k]);
    const float val = __ldg(&sorted_tomo[base_ij + (size_t)k]);
    // const int   jj  = indexes[base_ij + (size_t)k];
    // const float val = sorted_tomo[base_ij + (size_t)k];

    tomo[base_i + (size_t)jj * (size_t)n_length + (size_t)k] = val;
}


__global__ void _swap_and_copy(float* __restrict__ output, const float* __restrict__ input, int n_slices, int n_angles, int n_length) {

    // Thread -> coordinates (no div/mod)
    const int k = blockIdx.x * blockDim.x + threadIdx.x; // fastest
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= n_slices || j >= n_angles || k >= n_length) return;

    output[i * n_length * n_angles + j * n_length + k] = input[i * n_length * n_angles + j * n_length + k];

}


// list[i,k] = sum_j |sorted(i,j,k)| / sum_j |smooth(i,j,k)|
// Shapes (C-order): [n_slice, n_angles, n_length], with k fastest.
__global__ void _large_list(float* __restrict__ list, const float* __restrict__ sorted_data, const float* __restrict__ smooth_data, float drop_ratio,int n_slices, int n_angles, int n_length){

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / (n_length);
    int k = idx % (n_length);
    if(i >= n_slices || k >= n_length) return;
    
    drop_ratio = max(0.0, min(0.8, drop_ratio));
    float ndrop = n_angles * drop_ratio / 2;
    
    float sum_1 = 0.0;
    float sum_2 = 0.0;
    for (int j = int(ndrop); j < n_angles - int(ndrop); j++) {
        sum_1 += abs(sorted_data[i * n_length * n_angles + j * n_length + k]);
        sum_2 += abs(smooth_data[i * n_length * n_angles + j * n_length + k]);
    }
    
    if (sum_2 != 0.0 && sum_1 != 0.0) {
        list[idx] = sum_1 / sum_2;
    } else {
        list[idx] = 1.0;
    }
}
      

__device__ void insertion_sort_(float arr[], int low, int high) {
    for (int i = low; i < high; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

__device__ void linear_fit(int n, float y[], int idx, double *slope, double *intercept) {
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_xx = 0.0;
  
      for (int i = idx; i < n + idx; i++) {
          double x_val = (double)i;
          double y_val = (double)y[i];
          sum_x += x_val;
          sum_y += y_val;
          sum_xy += x_val * y_val;
          sum_xx += x_val * x_val;
      }
      double nn = (double) n;
  
      double denominator = (double) nn * sum_xx - sum_x * sum_x;
      if (denominator == 0.0) {
          // Handle the case where the matrix is singular
          *slope = 0.0;
          *intercept = 0.0;
          return;
      }
  
      *slope = (nn * sum_xy - sum_x * sum_y) / denominator;
      *intercept = (sum_y - (*slope) * sum_x) / nn;
  }
  
  


// Assumptions kept: one thread per slice i (gridDim.x * blockDim.x >= n_slices)
// list_data:  (n_slices, n_lengths)  original values
// sorted_list: scratch (n_slices, n_lengths); reused later as a temp mask buffer
// list_mask:  (n_slices, n_lengths)  output boolean mask
// background_ratio == snr in the Python
__global__ void _detect_stripe_corrected(bool* __restrict__ list_mask, const float* __restrict__ list_data, float* __restrict__ sorted_list, int n_slices, int n_lengths, float background_ratio) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_slices) return;
  
    // 1) Copy row -> sorted_list
    for (int j = 0; j < n_lengths; ++j)
        sorted_list[i * n_lengths + j] = list_data[i * n_lengths + j];
  
    // 2) Sort; assume ascending (small->large). If your sort is descending, swap y_min/y_max below.
    insertion_sort_(sorted_list, i * n_lengths + 0, i * n_lengths + n_lengths);
  
    // 3) Linear fit on the central band: indices [ndrop, n_lengths - ndrop - 1)
    const int ndrop = (int)(0.25f * n_lengths);            // matches np.int16(0.25*numdata)
    int x0   = ndrop;
    int len  = n_lengths - 2 * ndrop - 1;                  // matches [ndrop:-ndrop-1]
    double slope = 0.0, intercept = 0.0;
  
    if (len >= 2) {
        // If your linear_fit signature is (len, y_ptr, x_start, &slope, &intercept)
        linear_fit(len, &sorted_list[i * n_lengths + x0], x0, &slope, &intercept);
    } else {
        // Degenerate case: not enough points → flat line at mid value
        intercept = sorted_list[i * n_lengths + n_lengths / 2];
        slope = 0.0;
    }
  
    // 4) End-point on the right (x = n_lengths - 1)
    const double numt1 = intercept + slope * (double)(n_lengths - 1);
  
    // 5) Noise level
    double noiselevel = fabs(numt1 - intercept);
    if (noiselevel < 1e-6) noiselevel = 1e-6;
  
    // 6) Extremes from the sorted array (ascending)
    const double y_min = (double)sorted_list[i * n_lengths + 0];
    const double y_max = (double)sorted_list[i * n_lengths + (n_lengths - 1)];
  
    // 7) Normalized deviations (match Python)
    const double val1 = fabs(y_max - intercept) / noiselevel; // max vs left end
    const double val2 = fabs(y_min - numt1)   / noiselevel;   // min vs right end
  
    const double snr = (double)background_ratio;
  
    // 8) Thresholds only if the deviation is large enough
    const bool use_upper = (val1 >= snr);
    const bool use_lower = (val2 >= snr);
  
    const double upper_thresh = intercept + noiselevel * snr * 0.5; // Python's upper
    const double lower_thresh = numt1    - noiselevel * snr * 0.5;  // Python's lower
  
    // 9) Build initial mask (on original list_data)
    for (int j = 0; j < n_lengths; ++j) {
        const double v = (double)list_data[i * n_lengths + j];
        const bool m = (use_upper && (v >  upper_thresh)) || (use_lower && (v <= lower_thresh));
        list_mask[i * n_lengths + j] = m;
  
        // reuse sorted_list as a temp mask buffer for dilation
        sorted_list[i * n_lengths + j] = m ? 1.0f : 0.0f;
    }
  
    // 10) 1D binary dilation by one (read temp, write list_mask)
    // out[j] = in[j-1] || in[j] || in[j+1]
    // interior
    for (int j = 1; j < n_lengths - 1; ++j) {
        if (sorted_list[i * n_lengths + j - 1] > 0.5f || sorted_list[i * n_lengths + j    ] > 0.5f || sorted_list[i * n_lengths + j + 1] > 0.5f) {
            list_mask[i * n_lengths + j] = true;
        }
    }
    // edges
    if (n_lengths >= 2) {
        if (sorted_list[i * n_lengths + 0] > 0.5f || sorted_list[i * n_lengths + 1] > 0.5f)
            list_mask[i * n_lengths + 0] = true;
        if (sorted_list[i * n_lengths + (n_lengths - 2)] > 0.5f || sorted_list[i * n_lengths + (n_lengths - 1)] > 0.5f)
            list_mask[i * n_lengths + (n_lengths - 1)] = true;
    }
}


// out[i,j,k] = tomo[i,j,k] / list[i,k], but only if !mask[i,k]
// Shapes (C-order): tomo/out: [Ni, Nj, Nk] (k fastest)
// list, mask: [Ni, Nk]
__global__ void _large_normalization_and_mask(
    float* __restrict__ out,
    float* __restrict__ tomo,
    const float* __restrict__ list,
    const bool*  __restrict__ mask,   
    int n_slices, int n_angles, int n_lengths)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x; // fastest
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= n_slices || j >= n_angles || k >= n_lengths) return;

    // Flat indices (64-bit to avoid overflow)
    const size_t data_idx   = (size_t)i * (size_t)n_lengths * (size_t)n_angles + (size_t)j * (size_t)n_lengths + (size_t)k;
    const size_t list_idx   = (size_t)i * (size_t)n_lengths + (size_t)k;

    // Early-out keeps loads coalesced and avoids wasted traffic if many masked
    if (!mask[list_idx]) {
        const float denom = __ldg(&list[list_idx]);
        const float val   = __ldg(&tomo[data_idx]);

        // denom != 0
        out[data_idx] = val / denom;
    }
}

// This duplicates edge pixels at the boundary (half-sample symmetric).
__device__ __forceinline__ int _reflect_index(int x, int n) {
  if (n <= 1) return 0;
  // Handle arbitrary out-of-range by "bouncing" back in.
  while (x < 0 || x >= n) {
      if (x < 0)       x = -x - 1;        // ... -1 -> 0, -2 -> 1, ...
      else /* x >= n */ x = 2 * n - 1 - x; // n -> n-1, n+1 -> n-2, ...
  }
  return x;
}

// 3D uniform_filter1d along axis=1 (angles), using CLAMP/TRUNCATE edges.
// NOTE: results will differ from SciPy's mode='reflect' near boundaries.
__global__ void _uniform_filter1d_over_angles(float* __restrict__ output, const float* __restrict__ input, int size, int Ni, int Nj, int Nk) {
    if (size < 1) return;

    const int k = blockIdx.x * blockDim.x + threadIdx.x; // fastest
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= Ni || j >= Nj || k >= Nk) return;

    // Even-size split: j-left .. j+right (e.g., size=10 -> j-5..j+4)
    const int left  = size / 2;
    const int right = size - left - 1;

    // Clamp window to [0, Nj-1]
    int start = j - left;           if (start < 0)     start = 0;
    int end   = j + right;          if (end   >= Nj)   end   = Nj - 1;
    const int count = end - start + 1;  // varies near edges

    // Sum over [start, end] at fixed (i, k)
    const size_t base_i  = (size_t)i * (size_t)Nj * (size_t)Nk;
    const size_t out_idx = base_i + (size_t)j * (size_t)Nk + (size_t)k;

    double sum = 0.0;
    for (int jj = start; jj <= end; ++jj) {
        const size_t pos = base_i + (size_t)jj * (size_t)Nk + (size_t)k;
        sum += (double)__ldg(&input[pos]);
    }

    output[out_idx] = (float)(sum / (double)count);
}

// list_data[i,k] = sum_j |tomo[i,j,k] - tomo_smooth[i,j,k]|
// Shapes (C-order): tomo: [n_slices, n_angles, n_length], k is fastest
__global__ void _list_unresponsive_sum(float* __restrict__ list_data, const float* __restrict__ tomo, const float* __restrict__ tomo_smooth, int n_slices, int n_angles, int n_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_slices * n_length)
      return; // Ensure not processing out of bounds indices
  
    int i = idx / n_length;
    int k = idx % n_length;
    float sum = 0.0;
  
    for (int j = 0; j < n_angles; ++j) {
      int idx_ = i * n_length * n_angles + j * n_length + k;
      sum += fabs(tomo[idx_] - tomo_smooth[idx_]); // Use fabs for floating-point abs
    }
    list_data[idx] = sum;
  
}


// In-place: smooth_list[i,k] = (list[i,k] / smooth_list[i,k])  if denom >= 1e-6
//            smooth_list[i,k] = 1.0                             otherwise
// Shapes (C-order): [n_slices, n_length]
__global__ void _list_unresponsive_divide(float* __restrict__ smooth_list, const float* __restrict__ list, int size_median, int n_slices, int n_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int k = idx % (n_length);
    if (idx >= n_slices * n_length) return;

    if (smooth_list[idx] < 1e-6)
        smooth_list[idx] = 1.0;
    else
        smooth_list[idx] = list[idx] / smooth_list[idx];
}



__global__ void _mask_unresponsive(bool* __restrict__ list_mask, int n_slices, int n_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_slices)
      return;
  
    list_mask[idx * n_length + 0] = false;
    list_mask[idx * n_length + 1] = false;
    list_mask[idx * n_length + n_length - 1] = false;
    list_mask[idx * n_length + n_length - 2] = false;
  }

// out[i,j,k] = mask[i,k] ? interp_between_neighbors : tomo[i,j,k]
// Uses nearest neighbors from left_idx/right_idx. Fallbacks:
// - Only L or only R available -> copy that neighbor (nearest neighbor).
// - Neither available (row fully masked) -> keep original value.
__global__ void _interpolate_with_neighbors(float* __restrict__ out, const float* __restrict__ tomo,const bool*  __restrict__ mask,const int*   __restrict__ left_idx,const int*   __restrict__ right_idx,int n_slices, int n_angles, int n_length) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x; // fastest
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= n_slices || j >= n_angles || k >= n_length) return;

    // 64-bit offsets for large volumes
    const size_t base_i   = (size_t)i * (size_t)n_angles * (size_t)n_length;
    const size_t base_ij  = base_i + (size_t)j * (size_t)n_length;
    const size_t data_idx = base_ij + (size_t)k;
    const size_t ik_idx   = (size_t)i * (size_t)n_length + (size_t)k;

    if (!mask[ik_idx]) {
        // Fast path: not masked, just copy
        out[data_idx] = tomo[data_idx];
        return;
    }

    const int kL = left_idx[ik_idx];   // -1 if none
    const int kR = right_idx[ik_idx];  // -1 if none
    const bool hasL = (kL >= 0);
    const bool hasR = (kR >= 0);

    float val;
    if (hasL && hasR && kR != kL) {
        const float denom = float(kR - kL);
        const float wL = float(kR - k) / denom;
        const float wR = float(k - kL) / denom;
        const float vL = tomo[base_ij + (size_t)kL];
        const float vR = tomo[base_ij + (size_t)kR];
        val = wL * vL + wR * vR;       // linear interp along k
    } else if (hasL) {
        val = tomo[base_ij + (size_t)kL]; // nearest-left
    } else if (hasR) {
        val = tomo[base_ij + (size_t)kR]; // nearest-right
    } else {
        val = tomo[data_idx];             // no neighbors: keep original
    }

    out[data_idx] = val;
}

// Build nearest-left / nearest-right good column indices per (i,k).
// If mask[i,k] == false (good), left_idx=right_idx=k.
// If there is no good neighbor to a side, that side is -1.
__global__ void _build_neighbors_interpolation(int* __restrict__ left_idx, int* __restrict__ right_idx, const bool* __restrict__ mask, int n_slices, int n_length){
    const int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    if (i >= n_slices) return;

    const size_t row = (size_t)i * (size_t)n_length;

    // Forward scan: nearest good to the LEFT (including self if good)
    int last_good = -1;
    for (int k = 0; k < n_length; ++k) {
        if (!mask[row + (size_t)k]) last_good = k;
        left_idx[row + (size_t)k] = last_good;
    }

    // Backward scan: nearest good to the RIGHT (including self if good)
    int next_good = -1;
    for (int k = n_length - 1; k >= 0; --k) {
        if (!mask[row + (size_t)k]) next_good = k;
        right_idx[row + (size_t)k] = next_good;
    }
}



// DESCENDING insertion sort on [low, high)
__device__ __forceinline__
void insertion_sort_desc(float* __restrict__ arr, int low, int high) {
    for (int i = low; i < high; ++i) {
        float key = arr[i];
        int j = i - 1;
        while (j >= low && arr[j] < key) {  // flip comparator for DESC
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

// Linear fit y ~ a*x + b over x = x0, x0+1, ..., x0+len-1
// y_ptr must point at the first element of the segment (list_sorted[x0]).
__device__ __forceinline__
void _linear_fit_segment(int len, const float* __restrict__ y_ptr, int x0, double* __restrict__ slope, double* __restrict__ intercept) {
    if (len <= 1) {
        *slope = 0.0;
        *intercept = (len == 1) ? (double)y_ptr[0] : 0.0;
        return;
    }

    const double n   = (double)len;
    const double t1  = (double)(len - 1);
    const double S_t = 0.5 * n * t1;                         // sum t
    const double S_t2= (t1 * (t1 + 1.0) * (2.0*t1 + 1.0)) / 6.0; // sum t^2

    const double Sx  = n * (double)x0 + S_t;                 // sum x
    const double Sxx = n * (double)x0 * (double)x0 + 2.0*(double)x0 * S_t + S_t2;

    double Sy  = 0.0;
    double Sty = 0.0;
    for (int t = 0; t < len; ++t) {
        const double vy = (double)y_ptr[t];
        Sy  += vy;
        Sty += (double)t * vy;
    }

    const double Sxy = (double)x0 * Sy + Sty;
    const double denom = n * Sxx - Sx * Sx;

    // Robust singularity guard (rare but safer than == 0.0)
    if (fabs(denom) < 1e-24) {
        *slope = 0.0;
        *intercept = Sy / n;  // horizontal line at mean
        return;
    }

    const double a = (n * Sxy - Sx * Sy) / denom;
    const double b = (Sy - a * Sx) / n;

    *slope = a;
    *intercept = b;
}
