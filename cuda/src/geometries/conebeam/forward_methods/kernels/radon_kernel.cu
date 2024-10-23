#include <iostream>
#include <stdexcept>
#include <chrono>
#include "geometries/conebeam/radon.cuh"

/*
    FALTA:
        CHECAR SE OS INPUTS DO USUÁRIO ESTÃO DE ACORDO COM O ESPERADO PELO PROGRAMA:
            lab.n(x, y, z) deve ser par, etc;
            verificar se nenhum nx, nbeta etc vai dar overflow (melhor fazer isso em python pq quando chegar aqui já vai ter o overflow);

        ESCOLHER SE VAMOS TER NH E NV PARA O DETECTOR OU SÓ N_DETECTOR; (na struct lab).

        Na função set_phantomidxs falta considerar o shift do phantom.
*/

int ray_integral(
    struct Lab lab,
    float *px, 
    float *py, 
    float *pz, 
    float *beta, 
    float *phantom, 
    float *tomo,  
    enum Optimize optimize_path, 
    int gpu_id)
{
    size_t  N_phantom, N_tomo;
    float *cuda_tomo = NULL, *cuda_phantom = NULL;
    float *cuda_px = NULL, *cuda_py = NULL, *cuda_pz = NULL, *cuda_beta = NULL;
    size_t num_blocks;
    //cudaStream_t s;
    
    cudaSetDevice(gpu_id);
    //cudaStreamCreate(&s); 

    N_phantom = (size_t)lab.nx * (size_t)lab.ny * (size_t)lab.nz;
    N_tomo = (size_t)lab.n_detector * (size_t)lab.nbeta;
    num_blocks  = N_tomo / NUM_THREADS + (N_tomo % NUM_THREADS == 0 ? 0:1);
    std::cout << "nbeta = " << lab.nbeta << "\n";
    std::cout << "Number of threads per block: " << NUM_THREADS << "\n";
    std::cout << "Number of blocks:            " << num_blocks << "\n";

    malloc_float_on_gpu(&cuda_tomo, N_tomo);
    malloc_float_on_gpu(&cuda_phantom, N_phantom);
    malloc_float_on_gpu(&cuda_px, lab.n_detector);
    malloc_float_on_gpu(&cuda_py, lab.n_detector);
    malloc_float_on_gpu(&cuda_pz, lab.n_detector);
    malloc_float_on_gpu(&cuda_beta, lab.nbeta);

    copy_float_array_from_cpu_to_gpu(cuda_phantom, phantom, N_phantom);
    copy_float_array_from_cpu_to_gpu(cuda_px, px, lab.n_detector);
    copy_float_array_from_cpu_to_gpu(cuda_py, py, lab.n_detector);
    copy_float_array_from_cpu_to_gpu(cuda_pz, pz, lab.n_detector);
    copy_float_array_from_cpu_to_gpu(cuda_beta, beta, lab.nbeta);

    switch (optimize_path) {
        case OPT_OFF:
            std::cout << "Running kernel...\n";
            kernel_OPT_OFF<<<num_blocks, NUM_THREADS>>>(cuda_px, cuda_py, cuda_pz, cuda_beta, cuda_phantom, cuda_tomo, lab);
            break;
        case OPT_ON:
            std::cout << "Running safely optimized kernel...\n";
            kernel_OPT_ON<<<num_blocks, NUM_THREADS>>>(cuda_px, cuda_py, cuda_pz, cuda_beta, cuda_phantom, cuda_tomo, lab);
            break;
        default:
            throw std::invalid_argument("Invalid argument. Variable 'optimize_path' must be of type 'enum OPTIMIZE'.");
    }

    copy_float_array_from_gpu_to_cpu(tomo, cuda_tomo, N_tomo);

    cudaFree(cuda_tomo);
    cudaFree(cuda_phantom);
    cudaFree(cuda_px);
    cudaFree(cuda_py);
    cudaFree(cuda_pz);
    cudaFree(cuda_beta);

    cudaDeviceSynchronize(); //cudaDeviceSynchronize();
    std::cout << cudaGetErrorString(cudaGetLastError()) << "\n\n";

    return 0;
}

void malloc_float_on_gpu(float **cuda_pointer, size_t N) {
    cudaMalloc(cuda_pointer, N * sizeof(float));
    if (*cuda_pointer == nullptr) {
        std::cout << "Failed to allocate memory on GPU.\n";
        std::cerr << "Failed to allocate memory on GPU.\n";
	    throw std::bad_alloc();
    }
}

void copy_float_array_from_cpu_to_gpu(float *cuda_pointer, float *pointer, long long int N) {
    cudaMemcpy(cuda_pointer, pointer, N * sizeof(float), cudaMemcpyHostToDevice);
}

void copy_float_array_from_gpu_to_cpu(float *pointer, float *cuda_pointer, long long int N) {
    cudaMemcpy(pointer, cuda_pointer, N * sizeof(float), cudaMemcpyDeviceToHost);
}


__global__ void kernel_OPT_OFF(float *px, float *py, float *pz, float *beta, float *phantom, float* tomo, struct Lab lab)
{
    long long int i, j, k; // índices do raio na coordenada top. 
    float cosb, sinb; 
    double t, dt; // t é a variável de parametrização do raio de luz.
    float ray_versor[3]; // versor com a direção do raio.
    float ray_sum = 0; // soma.

    cosb = __cosf(beta[beta_Idx(lab)]);
    sinb = __sinf(beta[beta_Idx(lab)]);
    apply_rotations(&t, ray_versor, px[pixel_Idx(lab)], py[pixel_Idx(lab)], pz[pixel_Idx(lab)], cosb, sinb, lab);
    if (radius(lab) < norm3df(xray(t, ray_versor, lab, cosb, sinb)-lab.x0, yray(t, ray_versor, lab, cosb, sinb)-lab.y0, zray(t, ray_versor, lab.sz)-lab.z0)) {
        tomo[tomo_Idx()] = 0;
        return;
    }
    dt = 2*radius(lab) / (lab.n_ray_points + 1); // +1 para ficar simétrico em torno de tm e pegar até o último pedacito, dado que n_ray_points vai ser par para pessoas normais.
    t -= radius(lab);
    for (int it = 0; it < lab.n_ray_points + 1; it++) {
        t += dt;
        set_phantom_idxs(xray(t, ray_versor, lab, cosb, sinb), yray(t, ray_versor, lab, cosb, sinb), zray(t, ray_versor, lab.sz), &i, &j, &k, lab);
        if (not_inside_phantom(i, j, k, lab)) {
            continue;
        }
        ray_sum += phantom[k*lab.nx*lab.ny + j*lab.nx + i];
    }
    tomo[tomo_Idx()] = ray_sum*dt; // como o nosso vetor direção da reta é normalizado, uma variação dt na variável de parametrização t também implica em uma variação dt na nossa reta.
}

__global__ void kernel_OPT_ON(float *px, float *py, float *pz, float *beta, float *phantom, float* tomo, struct Lab lab)
{
    size_t n = blockDim.x * blockIdx.x + threadIdx.x; // acho melhor trocar a variável pela conta explícita quando precisar para economizar memória.
    int i_detector;
    short int m; // índices no tomograma.
    long long int i, j, k; // índices do raio na coordenada top. 
    float cosb, sinb; 
    double dist, tm, t, dt; // t é a variável de parametrização do raio de luz.
    float ray_versor[3]; // versor com a direção do raio.
    double ray_sum = 0; // soma.
    bool was_inside_phantom = false;

    set_tomo_idxs(n, &i_detector, &m, lab);
    cosb = __cosf(beta[m]);
    sinb = __sinf(beta[m]);
    apply_rotations(&tm, ray_versor, px[i_detector], py[i_detector], pz[i_detector], cosb, sinb, lab);
    dist = norm3df(2*lab.x, 2*lab.y, 2*lab.z); // falta adicionar shift phantom. 
    // trocar o seguint if por funcao:
    if (dist/2 < norm3df(xray(tm, ray_versor, lab, cosb, sinb)-lab.x0, yray(tm, ray_versor, lab, cosb, sinb)-lab.y0, zray(tm, ray_versor, lab.sz)-lab.z0)) {
        tomo[n] = 0;
        return;
    }
    dt = dist / (lab.n_ray_points + 1); // +1 para ficar simétrico em torno de tm e pegar até o último pedacito, dado que n_ray_points vai ser par para pessoas normais.
    t = tm - dist/2;
    for (int it = 0; it < lab.n_ray_points + 1; it++) {
        t += dt;
        set_phantom_idxs(xray(t, ray_versor, lab, cosb, sinb), yray(t, ray_versor, lab, cosb, sinb), zray(t, ray_versor, lab.sz), &i, &j, &k, lab);
        if (not_inside_phantom(i, j, k, lab)) {
            if (was_inside_phantom) { 
                break; // porque phantoms são cubinhos convexos; uma vez fora depois de dentro, nunca mais dentro.
            } 
            else {
                continue;
            }
        }
        ray_sum += phantom[k*lab.nx*lab.ny + j*lab.nx + i];
        was_inside_phantom = true; 
    }
    tomo[n] = ray_sum*dt; // como o nosso vetor direção da reta é normalizado, uma variação dt na variável de parametrização t também implica em uma variação dt na nossa reta.
}


inline __device__ size_t tomo_Idx() 
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}

inline __device__ size_t pixel_Idx(struct Lab lab) 
{
    return tomo_Idx() % lab.n_detector;
}

inline __device__ size_t beta_Idx(struct Lab lab) 
{
    return tomo_Idx() / lab.n_detector;
}

inline __device__ float radius(struct Lab lab)
{
    return norm3df(lab.x, lab.y, lab.z);
}

inline __device__ double xray(double t, float ray_versor[3], struct Lab lab, float cosb, float sinb) 
{
    return (cosb*lab.sx - sinb*lab.sy) + (t * ray_versor[0]);
}

inline __device__ double yray(double t, float ray_versor[3], struct Lab lab, float cosb, float sinb)
{
    return (sinb*lab.sx + cosb*lab.sy) + (t * ray_versor[1]);
}

inline __device__ double zray(double t, float ray_versor[3], float sz)
{
    return sz + (t * ray_versor[2]);
}

inline __device__ void apply_rotations(
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

inline __device__ void set_tomo_idxs(long long int n, int *i_detector, short int *m, struct Lab lab) 
{
    *m = n / lab.n_detector;
    *i_detector = n % lab.n_detector;
}

inline __device__ void set_phantom_idxs(float x, float y, float z, long long int *i, long long int *j, long long int *k, struct Lab lab) 
{
    float dx, dy, dz; // usar variaveis locais do kernel ao inves pra não ter que calcular toda vez.

    dx = 2*lab.x / (lab.nx); 
    dy = 2*lab.y / (lab.ny);
    dz = 2*lab.z / (lab.nz);
    *i = __float2int_rn((x-lab.x0) / dx) + lab.nx/2; // arrumar para evitar overflow; no mínimo provar que não vai acontecer overflow não importa a entrada.
    *j = __float2int_rn((y-lab.y0) / dy) + lab.ny/2; // arrumar para evitar overflow; no mínimo provar que não vai acontecer overflow não importa a entrada.
    *k = __float2int_rn((z-lab.z0) / dz) + lab.nz/2; // arrumar para evitar overflow; no mínimo provar que não vai acontecer overflow não importa a entrada.
}

inline __device__ bool not_inside_phantom(long long int i, long long int j, long long int k, struct Lab lab) // criar uma not_inside usando floats x, y, z para evitar overflow dos indices i, j, k.
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
