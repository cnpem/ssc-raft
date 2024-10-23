// Cuda (single and double precision) Mathematical Functions: 
//     "To use these functions you do not need to include any additional header files in your program.".
//     https://docs.nvidia.com/cuda/pdf/CUDA_Math_API.pdf
#include <stddef.h>

// int const NUM_THREADS_UNARY_OPERATION  = 1024;
// int const NUM_THREADS_BINARY_OPERATION = 1024;
// int const NUM_THREADS_PER_BLOCK = 128;


// Kernels (cuda __global__ functions):

/** @brief Calculate 1/arr[i] for each element arr[i] in the array arr (inplace function).
 *
 * This cuda kernel is a unary operator and should be launched with NUM_THREADS_UNARY_OPERATION 
 * threads per block.
 * @param[in]  arr  array pointer.
 * @param[in]  size  array size.
 * @return void function.
 */
__global__ void reciprocal(
    float *arr, 
    size_t size);

/** @brief Calculate arr1[i]*arr2[i] and store the result in arr1[i].
 *
 * This cuda kernel is a binary operator and should be launched with NUM_THREADS_BINARY_OPERATION 
 * threads per block.
 * @param[in]  arr1  array 1 pointer.
 * @param[in]  arr2  array 2 pointer.
 * @param[in]  size  arrays size.
 * @return void function.
 */
__global__ void multiply(
    float *arr1, 
    float *arr2,
    size_t size);

__global__ void apply_limits(
    float *recon,
    float max_val,
    size_t size);

__global__ void total_variation_3d(
    float *recon, 
    size_t size,
    struct Lab lab,
    float tv_param);

__global__ void total_variation_2d(
    float *recon, 
    size_t size,
    struct Lab lab,
    float tv_param);

__global__ void backpropagation(
    float *px, float *py, float *pz, float *beta, 
    float *counts, float *backproj, 
    struct Lab lab);

__global__ void backproj_of_radon(
    float *flat, float *px, float *py, float *pz, float *beta, 
    float *recon, float *new_recon, 
    struct Lab lab,
    struct BulkBoundary boundary);

__global__ void backproj_of_radon_2(
    float *flat, float *px, float *py, float *pz, float *beta, 
    float *recon, float *new_recon, 
    struct Lab lab,
    struct BulkBoundary boundary);


// Cuda __device__ functions:
__device__ void initialize_ray(
    long long int *i, long long int *j, long long int *k,
    float *cosb, float *sinb, double *tm, double *dt,
    float *ray_versor, float *px, float *py, float *pz, float *beta,
    struct Lab lab);
__device__ bool ray_intersects_recon(float t, float *ray_versor, float cosb, float sinb, struct Lab lab);
__device__ bool still_inside_the_same_voxel(
    long long int i, long long int j, long long int k,
    long long int iprev, long long int jprev, long long int kprev);
__device__ size_t tomo_Idx();
__device__ size_t pixel_Idx(struct Lab lab);
__device__ size_t beta_Idx(struct Lab lab);
__device__ float radius(struct Lab lab);
__device__ double xray(double t, float ray_versor[3], struct Lab lab, float cosb, float sinb);
__device__ double yray(double t, float ray_versor[3], struct Lab lab, float cosb, float sinb);
__device__ double zray(double t, float ray_versor[3], float sz);
__device__ void apply_rotations(
    double *tm,
    float ray_versor[3],
    float detector_x,
    float detector_y,
    float detector_z,
    float cosb,
    float sinb,
    struct Lab lab);
__device__ void set_tomo_idxs(long long int n, int *i_detector, short int *m, struct Lab lab);
__device__ void set_phantom_idxs(float x, float y, float z, long long int *i, long long int *j, long long int *k, struct Lab lab);
__device__ bool not_inside_phantom(long long int i, long long int j, long long int k, struct Lab lab);
