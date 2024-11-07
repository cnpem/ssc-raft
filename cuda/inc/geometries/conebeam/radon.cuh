#include <string>

#define NUM_THREADS 1024

enum Optimize {
    OPT_OFF, // sem otimizar o caminho do raio.
    OPT_ON, // com otimização mas sem suposições sobre o phantom;
};

struct Lab {  
    float x, y, z; // phantom size in each direction.
    float x0, y0, z0;
    int nx, ny, nz; // number of phantom points in each direction.
    float sx, sy, sz; // x-ray source position (x, y, z).
    int nbeta; // number of angles.
    int n_detector; // number of detector pixels.
    int n_ray_points; // number of integration points in the x-ray path.
};

// cbradon functions.
extern "C"{
int cbradon(
    struct Lab lab, 
    float* px, 
    float *py, 
    float *pz, 
    float *beta, 
    float *sample, 
    float *tomo, 
    int gpu_id);

int cbradon_MultiGPU(
        int* gpus, int ngpus,
        struct Lab lab,
        float* px, float *py, float *pz,
        float *beta,
        float *sample,
        float *tomo);
}

// Ray integral function.
int ray_integral(
    struct Lab lab, 
    float *px, 
    float *py, 
    float *pz, 
    float *beta, 
    float *phantom, 
    float *tomo, 
    enum Optimize optimize_path,
    int gpu_id);

// Kernels (cuda __global__ functions).
__global__ void kernel_OPT_OFF(float *px, float *py, float *pz, float *beta, float *phantom, float* tomo, struct Lab lab);
__global__ void kernel_OPT_ON(float *px, float *py, float *pz, float *beta, float *phantom, float* tomo, struct Lab lab);

// Cuda __device__ functions.
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
__device__ void set_tomo_idxs(long long int n, long int *i_detector, short int *m, struct Lab lab);
__device__ void set_phantom_idxs(float x, float y, float z, long long int *i, long long int *j, long long int *k, struct Lab lab);
__device__ bool not_inside_phantom(long long int i, long long int j, long long int k, struct Lab lab);

// Utilitary functions.
void malloc_float_on_gpu(float **cuda_pointer, size_t N);
void copy_float_array_from_cpu_to_gpu(float *cuda_pointer, float *pointer, long long int N);
void copy_float_array_from_gpu_to_cpu(float *pointer, float *cuda_pointer, long long int N);
