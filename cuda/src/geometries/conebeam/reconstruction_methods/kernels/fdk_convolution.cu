#include <stdio.h>
#include <math.h>
#include <cufft.h>
#include "geometries/conebeam/fdk.hpp"

// fazer funções para malloc e copia para não poluir os códigos.
// MUDA O NOME DA CALC_CONV PARA FILTER_BY_CONV
// TROCAR FUNCOES MATEMATICAS POR SUAS RESPECTIVAS FLOATS!!


// __host__ void calc_conv(Lab lab, float* proj, Process process){ // CRIAR NOVO TIPO Process SÓ COM O NECESSARIO (as variaveis que for usar). quando for usar, criar Process process_filter e process_backproj.
//     long long int N = process.n_filter; // trocar por process.n_filter
//     float* Q;
//     float* c_proj;

//     int n_threads = NUM_THREADS;
//     long int n_blocks  = N/n_threads + (N % n_threads == 0 ? 0:1);

//     cudaMalloc(&Q, N * sizeof(float));     
//     cudaDeviceSynchronize();

//     cudaMalloc(&c_proj, N * sizeof(float));
//     cudaDeviceSynchronize();

//     cudaDeviceSynchronize();
//     cudaMemcpy(c_proj, proj + (process.i*process.n_filter), N * sizeof(float), cudaMemcpyHostToDevice); // trocar proj por proj + process.i*process.n_filter; trocar cuda_proj por cuda_proj + process.i*process.n_filter.
    
//     calc_Q<<<n_blocks, n_threads>>>(Q, c_proj, lab, process);
//     cudaDeviceSynchronize();

//     cudaMemcpy(proj + (process.i*process.n_filter), Q, N * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();

//     printf(cudaGetErrorString(cudaGetLastError()));
//     printf("\n");

//     cudaFree(c_proj);
//     cudaFree(Q);
    
// }

extern "C"{
void filtering_conv(Lab lab, float* proj, float* Q, Process process) {
    long long int N = process.n_filter; 
    int n_threads = NUM_THREADS;
    long int n_blocks  = N/n_threads + (N % n_threads == 0 ? 0:1);

    printf("\n Filtering Convolution: Process = %d \n", process.i_gpu);

    cudaSetDevice(process.i_gpu);

    clock_t b_begin = clock();

    calc_Q<<<n_blocks, n_threads>>>(Q, proj, lab, process);

    cudaDeviceSynchronize(); 
    printf(cudaGetErrorString(cudaGetLastError()));
    printf("\n");

    clock_t b_end = clock();
    printf("Time filtering_conv: Gpu %d ---- %f \n",process.i_gpu, double(b_end - b_begin)/CLOCKS_PER_SEC);
   

}}

extern "C"{
__global__ void calc_Q(float* Q, float* proj, Lab lab, Process process) {
    long long int n = blockDim.x * blockIdx.x + threadIdx.x;
    int i,j,k;
    float X, Z;
    float w_max = 1 / (2*lab.dh); // conferir!

    set_filter_idxs_conv(n, &k, &j, &i, lab, process);

    Z = (i + process.i*process.z_filter)*lab.dv - lab.v;
    X = k*lab.dh - lab.h;

    Q[n] = calc_convolution(&proj[i*lab.nbeta*lab.nh + j*lab.nh], lab, X, Z, w_max);
}}


extern "C"{
__device__ float calc_convolution(float* projection, Lab lab, float X, float Z, float w_max) {
    float conv = 0;
    float xi, cos_thetai;
    float D2 = lab.Dsd*lab.Dsd;

    for(int i = 0; i < lab.nh; i++) {
        xi = i*lab.dh - lab.h;
        cos_thetai = lab.D/sqrt(D2 + xi*xi + Z*Z);
        conv += projection[i] * cos_thetai * gx(X - xi, w_max);
    }
    return conv*lab.dx;
}}

extern "C"{
__forceinline__ __device__ float gx(float x, float w_max) { 
    float k = 2*M_PI*w_max*x;
    if(fabs(x) < EPSILON_GX) {
        return w_max*w_max;
    }
    return 2 *w_max*w_max* (k*sin(k)+cos(k)-1) / (k*k);
}}

extern "C"{
__device__ float integral_of_gx(float a, float b, float w_max) {
    float int_gx_ka, int_gx_kb;
    float ka = 2*M_PI*w_max*a;
    float kb = 2*M_PI*w_max*b;
    int_gx_ka = (1-cos(ka)) / ka;
    int_gx_kb = (1-cos(kb)) / kb;
    if(fabs(a) < EPSILON_GX) {
        int_gx_ka = 0;
    }
    if(fabs(b) < EPSILON_GX) {
        int_gx_kb = 0;
    }
    return w_max*(int_gx_kb - int_gx_ka) / M_PI;
}}

extern "C"{
__device__ void set_filter_idxs_conv(long long int n, int* i, int*j, int* k, Lab lab, Process process) {
    long int nij, rem_ij;
    nij = lab.nh*lab.nbeta;
    *k = (n) / nij;
    rem_ij = (n) % nij;
    *j = rem_ij / lab.nh;
    *i = rem_ij % lab.nh;
}}