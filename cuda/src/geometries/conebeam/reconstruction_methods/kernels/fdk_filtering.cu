#include "common/logerror.hpp"
#include "common/opt.hpp"
#include "geometries/conebeam/fdk.hpp"

extern "C"{
void copy_gpu_filter_fft(Lab lab, 
float* proj, float** c_proj, 
cufftComplex** c_signal, float** W, 
Process process) 
{
    long long int Npad = process.n_filter_pad;

    // clock_t begin = clock();
    HANDLE_ERROR(cudaSetDevice(process.i_gpu));

    float *c_tomo = opt::allocGPU<float>((size_t)process.n_filter);/* Alloc GPU sinograms to receive from CPU (temporary ptr)*/

    HANDLE_ERROR(cudaMalloc(c_signal, sizeof(cufftComplex)*Npad)); /* Alloc GPU filter signal padded */
    HANDLE_ERROR(cudaMalloc(W, (lab.nph) * sizeof(float))); /* Alloc GPU filter kernel padded */
    HANDLE_ERROR(cudaMalloc(c_proj, process.n_filter_pad * sizeof(float))); /* Alloc GPU sinograms padded */ 

    /* Copy CPU sinograms to GPU (temporary ptr) */
    HANDLE_ERROR(cudaMemcpy(c_tomo, &proj[process.idx_filter], process.n_filter * sizeof(float), cudaMemcpyHostToDevice));

    /* Extended domain padding of sinograms */
    /* Projection GPUs padded Grd and Blocks */
    dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 TomogridBlock( (int)ceil( lab.nph          / TPBX ) + 1,
                        (int)ceil( lab.nbeta        / TPBY ) + 1,
                        (int)ceil( process.z_filter / TPBZ ) + 1);
    
    /* Copy GPU sinograms to padded GPU sinograms *c_proj*/
    opt::paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(c_tomo, *c_proj, 
                                                        dim3(lab.nh, lab.nbeta, process.z_filter),
                                                        dim3(lab.padh, 0, 0));
    
    // printf("Filter number: %d \n",lab.filter_type);
    switch (lab.filter_type){
        case 0:
            // No filter Applied
            break;
        case 1:
            // Gaussian
            filt_Gaussian<<<1,1>>>(lab, *W);
            break;
        case 2:
            // Lorentz
            filt_Lorentz<<<1,1>>>(lab, *W);
            break;
        case 3:
            // Cosine
            filt_Cosine<<<1,1>>>(lab, *W);
            break;
        case 4:
            // Rectangle
            filt_Rectangle<<<1,1>>>(lab, *W);
            break;
        case 5:
            // Hann
            filt_Hann<<<1,1>>>(lab, *W);
            break;
        case 6:
            // Hamming
            filt_Hamming<<<1,1>>>(lab, *W);
            break;
        case 7:
            // Ramp
            filt_Ramp<<<1,1>>>(lab, *W);
            break;
        default:
            // Ramp
            filt_Ramp<<<1,1>>>(lab, *W);
    }

    HANDLE_ERROR(cudaFree(c_tomo));

    // clock_t end = clock();
    // printf("Time copy_to_gpu filter: Gpu %d \n",process.i);
}}

extern "C"{
void copy_cpu_filter_fft(Lab lab, float* proj, float* c_proj, cufftComplex* c_signal, float* c_W,  Process process) {
    // clock_t begin = clock();
    HANDLE_ERROR(cudaSetDevice(process.i_gpu));

    HANDLE_ERROR(cudaDeviceSynchronize()); 

    long long int N = process.n_filter; 

    float *c_tomo = opt::allocGPU<float>((size_t)process.n_filter);/* Alloc GPU sinograms to receive from CPU (temporary ptr)*/

    /* Extended domain padding of sinograms */
    /* Projection GPUs padded Grd and Blocks */
    dim3 TomothreadsPerBlock(TPBX,TPBY,TPBZ);
    dim3 TomogridBlock( (int)ceil( lab.nph          / TPBX ) + 1,
                        (int)ceil( lab.nbeta        / TPBY ) + 1,
                        (int)ceil( process.z_filter / TPBZ ) + 1);
    
    /* Copy GPU sinograms to padded GPU sinograms *c_proj*/
    opt::remove_paddR2R<<<TomogridBlock,TomothreadsPerBlock>>>(c_proj, c_tomo,
                                                        dim3(lab.nh, lab.nbeta, process.z_filter),
                                                        dim3(lab.padh, 0, 0));

    HANDLE_ERROR(cudaMemcpy(&proj[process.idx_filter], c_tomo, N*sizeof(float), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(c_tomo));
    HANDLE_ERROR(cudaFree(c_proj));
    HANDLE_ERROR(cudaFree(c_signal));
    HANDLE_ERROR(cudaFree(c_W));

    // clock_t end = clock();
    // printf("Time copy_to_cpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
    // printf("Time copy_to_cpu Filter: Gpu %d \n",process.i);

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
 
    // printf("GPU memory allocated...\n");

    cudaDeviceSynchronize(); 
    // printf(cudaGetErrorString(cudaGetLastError()));
    // printf("\n");


    clock_t end = clock();
    // printf("Time copy_to_gpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
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
    // printf("Time copy_to_cpu: Gpu %d ---- %f \n",process.i, double(end - begin)/CLOCKS_PER_SEC);
}}
