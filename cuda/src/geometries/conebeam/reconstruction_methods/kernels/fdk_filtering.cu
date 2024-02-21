#include "include.hpp"
#include "geometries/conebeam/fdk.hpp"
#include "common/types.hpp"

extern "C"{
void copy_gpu_filter_fft(Lab lab, float* proj, float** c_proj, cufftComplex** c_signal, float** W, Process process) {
    // long long int N = process.n_filter;
    long long int Npad = process.n_filter_pad;

    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    // printf(cudaGetErrorString(cudaGetLastError()));
    // printf("\n");

    //cudaMalloc(c_signal, sizeof(cufftComplex)*N);
    cudaMalloc(c_signal, sizeof(cufftComplex)*Npad);

    cudaMalloc(c_proj, process.n_filter * sizeof(float));    
    cudaMemcpy(*c_proj, &proj[process.idx_filter], process.n_filter * sizeof(float), cudaMemcpyHostToDevice);

    // cudaMalloc(W, lab.nh * sizeof(float));
    cudaMalloc(W, (lab.nph) * sizeof(float));

    printf("Filter number: %d \n",lab.filter_type);
    switch (lab.filter_type){
        case 0:
            // No filter Applied
            break;
        case 1:
            // Gaussian
            filt_Gaussian<<< 1, 1>>>(lab, *W);
            break;
        case 2:
            // Lorentz
            filt_Lorentz<<< 1, 1>>>(lab, *W);
            break;
        case 3:
            // Cosine
            filt_Cosine<<< 1, 1>>>(lab, *W);
            break;
        case 4:
            // Rectangle
            filt_Rectangle<<< 1, 1>>>(lab, *W);
            break;
        case 5:
            // Hann
            filt_Hann<<< 1, 1>>>(lab, *W);
            break;
        case 6:
            // Hamming
            filt_Hamming<<< 1, 1>>>(lab, *W);
            break;
        case 7:
            // Ramp
            filt_Ramp<<< 1, 1>>>(lab, *W);
            break;
        default:
            // Ramp
            filt_Ramp<<< 1, 1>>>(lab, *W);

    }

    // Normalize kernel by maximum value
    // cublasHandle_t handle = NULL;
    // cublasCreate(&handle);
    // cublasStatus_t stat;
    // int max;

    // int n_threads = NUM_THREADS;
    // long long int n_blocks  = N/n_threads + (N % n_threads == 0 ? 0:1);

    // stat = cublasIsamax(handle, lab.nph, *W, 1, &max);

    // if (stat != CUBLAS_STATUS_SUCCESS)
    //     printf("Cublas Max failed\n");

    // float maximum;
    // HANDLE_ERROR(cudaMemcpy(&maximum, *W + max, sizeof(float), cudaMemcpyDeviceToHost));
    // Normalize<<<n_blocks, n_threads>>>(*W, maximum, lab.nph, 1);

    cudaDeviceSynchronize(); 
    // printf(cudaGetErrorString(cudaGetLastError()));
    // printf("\n");

 
    printf("GPU memory allocated...\n");

    clock_t end = clock();
    printf("Time copy_to_gpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);

    // cublasDestroy(handle);
}}

extern "C"{
void copy_cpu_filter_fft(float* proj, float* c_proj, cufftComplex* c_signal, float* c_W,  Process process) {
    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    // printf(cudaGetErrorString(cudaGetLastError()));
    // printf("\n");

    long long int N = process.n_filter;   
    cudaMemcpy(&proj[process.idx_filter], c_proj, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_proj);
    cudaFree(c_signal);
    cudaFree(c_W);

    clock_t end = clock();
    printf("Time copy_to_cpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_gpu_filter_conv(Lab lab, float* proj, float** c_proj, float** c_Q, Process process) {
    long long int N = process.n_filter;

    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    // printf(cudaGetErrorString(cudaGetLastError()));
    // printf("\n");

    cudaMalloc(c_Q, sizeof(float)*N);

    cudaMalloc(c_proj, process.n_filter * sizeof(float));    
    cudaMemcpy(*c_proj, &proj[process.idx_filter], process.n_filter * sizeof(float), cudaMemcpyHostToDevice);
 
    printf("GPU memory allocated...\n");

    cudaDeviceSynchronize(); 
    // printf(cudaGetErrorString(cudaGetLastError()));
    // printf("\n");


    clock_t end = clock();
    printf("Time copy_to_gpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}

extern "C"{
void copy_cpu_filter_conv(float* proj, float* c_proj, float* c_Q, Process process) {
    clock_t begin = clock();
    cudaSetDevice(process.i_gpu);

    cudaDeviceSynchronize(); 
    // printf(cudaGetErrorString(cudaGetLastError()));
    // printf("\n");

    long long int N = process.n_filter;   //lab.nbeta * lab.nv * lab.nh;
    cudaMemcpy(&proj[process.idx_filter], c_Q, N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(c_proj);
    cudaFree(c_Q);

    clock_t end = clock();
    printf("Time copy_to_cpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}
