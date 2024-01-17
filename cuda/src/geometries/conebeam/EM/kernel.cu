// Cuda (single and double precision) Mathematical Functions: 
//     "To use these functions you do not need to include any additional header files in your program.".
//     https://docs.nvidia.com/cuda/pdf/CUDA_Math_API.pdf
#include <stddef.h>
#include <stdio.h>
#include "../../../../inc/geometries/conebeam/emc_kernel.cuh"
#include "../../../../inc/geometries/conebeam/emc_cone_data_types.cuh"


#define MAX_NUM_OF_VOXELS_IN_RAY_PATH 8192


__global__ void reciprocal(
    float *arr,
    size_t size)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        arr[i] = 1/arr[i];
}

__global__ void multiply(
    float *arr1,
    float *arr2,
    size_t size)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        arr1[i] *= arr2[i];
}

__global__ void apply_limits(
    float *recon,
    float max_val,
    size_t size)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size) {
        if (max_val < recon[i]) {
            recon[i] = max_val;
        }
    }
}

__global__ void total_variation_3d(
    float *recon, 
    size_t size,
    struct Lab lab,
    float tv_param)
{
    float curr, tv_term;
    float xnext, ynext, znext;
    float sum_diff, sqrt_sum_sq_diff;
    bool ok;
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (    0 < i && i < size-1 
            && 0 < i-lab.nx && i+lab.nx < size-1
            && 0 < i-lab.nx*lab.ny && i+lab.nx*lab.ny < size-1) {
        curr = recon[i];
        xnext = recon[i+1];
        ynext = recon[i+lab.nx];
        znext = recon[i+lab.nx*lab.ny];
        sum_diff = 3*curr -xnext -ynext -znext;
        sqrt_sum_sq_diff = 
           +(curr-xnext) * (curr-xnext)
           +(curr-ynext) * (curr-ynext)
           +(curr-znext) * (curr-znext);
        sqrt_sum_sq_diff = sqrtf(sqrt_sum_sq_diff);
        tv_term = tv_param * sum_diff; // / sqrt_sum_sq_diff; fora pq pode ser div por zero.
        do {
            ok = 0 <= (curr - tv_term);
            if (ok) {
                recon[i] -= tv_term;
            }
            tv_term = .5 * tv_term;
        } while(!ok);
    }
}


__global__ void total_variation_2d(
    float *recon, 
    size_t size,
    struct Lab lab,
    float tv_param)
{
    float curr, tv_term;
    float xnext, ynext;
    float sum_diff, sqrt_sum_sq_diff;
    bool ok;
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (    0 < i && i < size-1 
            && 0 < i-lab.nx && i+lab.nx < size-1) {
        curr = recon[i];
        xnext = recon[i+1];
        ynext = recon[i+lab.nx];
        sum_diff = 2*curr - xnext - ynext;
        sqrt_sum_sq_diff = (curr-xnext)*(curr-xnext) + (curr-ynext)*(curr-ynext);
        sqrt_sum_sq_diff = sqrtf(sqrt_sum_sq_diff);
        tv_term = tv_param * sum_diff; // / sqrt_sum_sq_diff; fora pq pode ser div por zero.
        do {
            ok = 0 <= (curr - tv_term);
            if (ok) {
                recon[i] -= tv_term;
            }
            tv_term = .5 * tv_term;
        } while(!ok);
    }
}


__global__ void single_pixel_carries_no_information_2d(
    float *recon, 
    size_t size,
    struct Lab lab,
    float tv_param)
{
    float curr, tv_term;
    float xprev, xnext;
    float yprev, ynext;
    // float zprev, znext;
    float sum_diff, sqrt_sum_sq_diff;
    bool ok;
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (    0 < i && i < size-1 
            && 0 < i-lab.nx && i+lab.nx < size-1) {
        curr = recon[i];
        xprev = recon[i-1];
        xnext = recon[i+1];
        yprev = recon[i-lab.nx];
        ynext = recon[i+lab.nx];
        sum_diff = 4*curr
                 -xprev -xnext
                 -yprev -ynext;
        sqrt_sum_sq_diff = 
            (curr-xprev) * (curr-xprev)
           +(curr-xnext) * (curr-xnext)
           +(curr-yprev) * (curr-yprev)
           +(curr-ynext) * (curr-ynext);
        sqrt_sum_sq_diff = sqrtf(sqrt_sum_sq_diff);
        tv_term = tv_param * sum_diff / sqrt_sum_sq_diff;
        do {
            ok = 0 < (curr - tv_term);
            if (ok) {
                recon[i] -= tv_term;
            }
            tv_term = .5 * tv_term;
        } while(!ok);
    }
}


__global__ void single_pixel_carries_no_information_3d(
    float *recon, 
    size_t size,
    struct Lab lab,
    float tv_param)
{
    float curr, tv_term;
    float xprev, xnext;
    float yprev, ynext;
    float zprev, znext;
    float sum_diff, sqrt_sum_sq_diff;
    bool ok;
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (    0 < i && i < size-1 
            && 0 < i-lab.nx && i+lab.nx < size-1
            && 0 < i-lab.nx*lab.ny && i+lab.nx*lab.ny < size-1) {
        curr = recon[i];
        xprev = recon[i-1];
        xnext = recon[i+1];
        yprev = recon[i-lab.nx];
        ynext = recon[i+lab.nx];
        zprev = recon[i-lab.nx*lab.ny];
        znext = recon[i+lab.nx*lab.ny];
        sum_diff = 6*curr
                 -xprev -xnext
                 -yprev -ynext
                 -zprev -znext;
        sqrt_sum_sq_diff = 
            (curr-xprev) * (curr-xprev)
           +(curr-xnext) * (curr-xnext)
           +(curr-yprev) * (curr-yprev)
           +(curr-ynext) * (curr-ynext)
           +(curr-zprev) * (curr-zprev)
           +(curr-znext) * (curr-znext);
        sqrt_sum_sq_diff = sqrtf(sqrt_sum_sq_diff);
        tv_term = tv_param * sum_diff / sqrt_sum_sq_diff;
        do {
            ok = 0 < (curr - tv_term);
            if (ok) {
                recon[i] -= tv_term;
            }
            tv_term = .5 * tv_term;
        } while(!ok);
    }
}

__global__ void backpropagation(
    float *px, 
    float *py, 
    float *pz, 
    float *beta, 
    float *counts, 
    float *backproj, 
    struct Lab lab)
{
    long long int i, j, k; // voxel index position.
    long long int iprev, jprev, kprev; // index of the previous position.
    float cosb, sinb; 
    double t, dt; // ray parametrization variable and its integration step.
    float ray_versor[3]; // ray versor direction.
    double a = 0; // ray length inside voxel.

    // lab.n_ray_points = 2*lab.n_ray_points; // para deixar o denominador mais preciso sem muito custo já que essa etapa só é feita uma vez.

    initialize_ray(
        &i, &j, &k, &cosb, &sinb, &t, &dt,
        ray_versor, px, py, pz, beta, lab);
    
    if (!ray_intersects_recon(t, ray_versor, cosb, sinb, lab)) {
        return;
    }

    t -= radius(lab); // verificar.
    for (size_t it = 0; it < lab.n_ray_points + 1; it++) {
        iprev = i;
        jprev = j;
        kprev = k;
        t += dt;
        set_phantom_idxs(
            xray(t, ray_versor, lab, cosb, sinb), 
            yray(t, ray_versor, lab, cosb, sinb), 
            zray(t, ray_versor, lab.sz), 
            &i, &j, &k, lab);
        if (not_inside_phantom(iprev, jprev, kprev, lab)) 
            continue;
        if (still_inside_the_same_voxel(i, j, k, iprev, jprev, kprev)) {
            a += dt;
        } else {
            atomicAdd(
                &backproj[kprev*lab.nx*lab.ny + jprev*lab.nx + iprev], 
                (float) (a * counts[beta_Idx(lab)*lab.ndetc + pixel_Idx(lab)]));
            a = 0;
        }
    }
}


__global__ void backproj_of_radon(
    float *flat,
    float *px, 
    float *py, 
    float *pz, 
    float *beta, 
    float *recon, 
    float *new_recon, 
    struct Lab lab,
    struct BulkBoundary boundary)
{
    long long int i, j, k; // voxel index position.
    long long int iprev, jprev, kprev; // index of the previous position.
    float cosb, sinb; 
    double t, dt; // ray parametrization variable and its integration step.
    float ray_versor[3]; // ray versor direction.
    double a = 0; // arc of ray length inside voxel.
    double ray_sum = 0;
    int nvoxels = 0,
        ipathist[MAX_NUM_OF_VOXELS_IN_RAY_PATH], 
        jpathist[MAX_NUM_OF_VOXELS_IN_RAY_PATH], 
        kpathist[MAX_NUM_OF_VOXELS_IN_RAY_PATH];
    float a_valuehist[MAX_NUM_OF_VOXELS_IN_RAY_PATH];
    double exp_ray_sum;

    initialize_ray(
        &i, &j, &k, &cosb, &sinb, &t, &dt,
        ray_versor,
        px, py, pz, beta,
        lab);
        
    if (!ray_intersects_recon(t, ray_versor, cosb, sinb, lab)) {
        return;
    }

    t -= radius(lab); // verificar. 
    for (int it = 0; it < lab.n_ray_points + 1; it++) {
        iprev = i;
        jprev = j;
        kprev = k;
        t += dt;
        set_phantom_idxs(
            xray(t, ray_versor, lab, cosb, sinb), 
            yray(t, ray_versor, lab, cosb, sinb), 
            zray(t, ray_versor, lab.sz), 
            &i, &j, &k, 
            lab);
        if (not_inside_phantom(iprev, jprev, kprev, lab))
            continue;
        if (still_inside_the_same_voxel(i, j, k, iprev, jprev, kprev)) {
            a += dt;
        } else {
            ray_sum += a*recon[kprev*lab.nx*lab.ny + jprev*lab.nx + iprev];
            ipathist[nvoxels] = iprev;
            jpathist[nvoxels] = jprev;
            kpathist[nvoxels] = kprev;
            a_valuehist[nvoxels] = (float) a;
            nvoxels += 1;
            a = 0;
        }
    }
    exp_ray_sum = expf(-ray_sum);
    for (int vox = 0; vox < nvoxels; ++vox) {
        atomicAdd(
            &new_recon[kpathist[vox]*lab.nx*lab.ny + jpathist[vox]*lab.nx + ipathist[vox]], 
            a_valuehist[vox] * flat[pixel_Idx(lab)] * exp_ray_sum);
            // a_valuehist[vox]*flat[pixel_Idx(lab)]*(1 - ray_sum));
    }
}


__global__ void backproj_of_radon_2(
    float *flat,
    float *px, 
    float *py, 
    float *pz, 
    float *beta, 
    float *recon, 
    float *new_recon, 
    struct Lab lab,
    struct BulkBoundary boundary)
{
    long long int i, j, k; // voxel index position.
    long long int iprev, jprev, kprev; // index of the previous position.
    float cosb, sinb; 
    double t, dt; // ray parametrization variable and its integration step.
    float ray_versor[3]; // ray versor direction.
    double a = 0; // arc of ray length inside voxel.
    double ray_sum = 0;
    double exp_ray_sum;

    initialize_ray(
        &i, &j, &k, &cosb, &sinb, &t, &dt,
        ray_versor,
        px, py, pz, beta,
        lab);
        
    if (!ray_intersects_recon(t, ray_versor, cosb, sinb, lab)) {
        return;
    }

    t -= radius(lab); // verificar. 
    for (int it = 0; it < lab.n_ray_points + 1; it++) {
        iprev = i;
        jprev = j;
        kprev = k;
        t += dt;
        set_phantom_idxs(
            xray(t, ray_versor, lab, cosb, sinb), 
            yray(t, ray_versor, lab, cosb, sinb), 
            zray(t, ray_versor, lab.sz), 
            &i, &j, &k, 
            lab);
        if (not_inside_phantom(iprev, jprev, kprev, lab))
            continue;
        if (still_inside_the_same_voxel(i, j, k, iprev, jprev, kprev)) {
            a += dt;
        } else {
            ray_sum += a*recon[kprev*lab.nx*lab.ny + jprev*lab.nx + iprev];
            a = 0;
        }
    }

    exp_ray_sum = expf(-ray_sum);

    initialize_ray(
        &i, &j, &k, &cosb, &sinb, &t, &dt,
        ray_versor,
        px, py, pz, beta,
        lab);
    t -= radius(lab); // verificar. 
    for (int it = 0; it < lab.n_ray_points + 1; it++) {
        iprev = i;
        jprev = j;
        kprev = k;
        t += dt;
        set_phantom_idxs(
            xray(t, ray_versor, lab, cosb, sinb), 
            yray(t, ray_versor, lab, cosb, sinb), 
            zray(t, ray_versor, lab.sz), 
            &i, &j, &k, 
            lab);
        if (not_inside_phantom(iprev, jprev, kprev, lab))
            continue;
        if (still_inside_the_same_voxel(i, j, k, iprev, jprev, kprev)) {
            a += dt;
        } else {
            atomicAdd(
                &new_recon[kprev*lab.nx*lab.ny + jprev*lab.nx + iprev], 
                a * flat[pixel_Idx(lab)] * exp_ray_sum);
            a = 0;
        }
    }
}


__device__ void initialize_ray(
    long long int *i,
    long long int *j,
    long long int *k,
    float *cosb,
    float *sinb,
    double *tm, double *dt,
    float *ray_versor,
    float *px, float *py, float *pz, float *beta,
    struct Lab lab)
{
    double t;

    *cosb = __cosf(beta[beta_Idx(lab)]);
    *sinb = __sinf(beta[beta_Idx(lab)]);
    apply_rotations(
        tm,
        ray_versor,
        px[pixel_Idx(lab)], py[pixel_Idx(lab)], pz[pixel_Idx(lab)],
        *cosb, *sinb,
        lab);
    *dt = 2*radius(lab) / (lab.n_ray_points + 1);
    t = -radius(lab);
    set_phantom_idxs(
        xray(t, ray_versor, lab, *cosb, *sinb), 
        yray(t, ray_versor, lab, *cosb, *sinb), 
        zray(t, ray_versor, lab.sz), 
        i, j, k, 
        lab);
}


__device__ bool ray_intersects_recon(float t, float *ray_versor, float cosb, float sinb, struct Lab lab) {
    if (radius(lab) < norm3df(xray(t, ray_versor, lab, cosb, sinb)-lab.x0, yray(t, ray_versor, lab, cosb, sinb)-lab.y0, zray(t, ray_versor, lab.sz)-lab.z0)) {
        return false;
    } else {
        return true;
    }
}

__device__ bool still_inside_the_same_voxel(
    long long int i, 
    long long int j,
    long long int k, 
    long long int iprev,
    long long int jprev, 
    long long int kprev) 
{
    if (i != iprev)
        return false;
    if (j != jprev) 
        return false;
    if (k != kprev) 
        return false;
    return true;
}



__device__ size_t tomo_Idx() 
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}

__device__ size_t pixel_Idx(struct Lab lab) 
{
    return tomo_Idx() % lab.ndetc;
}

__device__ size_t beta_Idx(struct Lab lab) 
{
    return tomo_Idx() / lab.ndetc;
}

__device__ float radius(struct Lab lab)
{
    return norm3df(lab.Lx, lab.Ly, lab.Lz);
}

__device__ double xray(double t, float ray_versor[3], struct Lab lab, float cosb, float sinb)
{
    return (cosb*lab.sx - sinb*lab.sy) + (t * ray_versor[0]);
}

__device__ double yray(double t, float ray_versor[3], struct Lab lab, float cosb, float sinb)
{
    return (sinb*lab.sx + cosb*lab.sy) + (t * ray_versor[1]);
}

__device__ double zray(double t, float ray_versor[3], float sz)
{
    return sz + (t * ray_versor[2]);
}

__device__ void apply_rotations(
    double *tm,
    float ray_versor[3],
    float detector_x, 
    float detector_y, 
    float detector_z, 
    float cosb,
    float sinb,
    struct Lab lab) 
{
    double dist, rot_sx, rot_sy;
    rot_sx = cosb*lab.sx - sinb*lab.sy;
    rot_sy = sinb*lab.sx + cosb*lab.sy;
    ray_versor[0] = (cosb*detector_x - sinb*detector_y) - rot_sx;
    ray_versor[1] = (sinb*detector_x + cosb*detector_y) - rot_sy;
    ray_versor[2] = detector_z - lab.sz;
    dist = norm3df(ray_versor[0], ray_versor[1], ray_versor[2]);
    ray_versor[0] = ray_versor[0] / (dist);
    ray_versor[1] = ray_versor[1] / (dist);
    ray_versor[2] = ray_versor[2] / (dist);
    *tm = - (rot_sx*ray_versor[0] + rot_sy*ray_versor[1] + lab.sz*ray_versor[2]);
}

__device__ void set_tomo_idxs(long long int n, int *i_detector, short int *m, struct Lab lab)
{
    *m = n / lab.ndetc;
    *i_detector = n % lab.ndetc;
}

__device__ void set_phantom_idxs(float x, float y, float z, long long int *i, long long int *j, long long int *k, struct Lab lab)
{
    float dx, dy, dz;

    dx = 2*lab.Lx / (lab.nx); 
    dy = 2*lab.Ly / (lab.ny);
    dz = 2*lab.Lz / (lab.nz);
    *i = __float2int_rn((x-lab.x0)/dx - 0.5) + lab.nx/2;
    *j = __float2int_rn((y-lab.y0)/dy - 0.5) + lab.ny/2;
    *k = __float2int_rn((z-lab.z0)/dz - 0.5) + lab.nz/2;
}

__device__ bool not_inside_phantom(long long int i, long long int j, long long int k, struct Lab lab)
{
    if (i < 0) {
        return true;
    }
    if (i >= lab.nx) {
        return true;
    }
    if (j < 0) {
        return true;
    }
    if (j >= lab.ny) {
        return true;
    }
    if (k < 0) {
        return true;
    }
    if (k >= lab.nz) {
        return true;
    }
    return false;
}