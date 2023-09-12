// #include "../../../inc/include.h"
// #include "../../../inc/common/kernel_operators.hpp"
// #include "../../../inc/common/complex.hpp"
// #include "../../../inc/common/types.hpp"
// #include "../../../inc/common/operations.hpp"
// #include "../../../inc/common/logerror.hpp"

// #include <iostream>  
// #include <fstream>   

// using namespace std;

// #define BLOCK_WIDTH 16 
// #define BLOCK_HEIGHT 16

// /*******************/
// /* iDivUp FUNCTION */
// /*******************/
// int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

// /********************/
// /* CUDA ERROR CHECK */
// /********************/
// #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
// inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
// {
//     if (code != cudaSuccess) 
//     {
//         fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//         if (abort) exit(code);
//     }
// }

// /**********************************************/
// /* KERNEL WITH OPTIMIZED USE OF SHARED MEMORY */
// /**********************************************/
// __global__ void Optimized_Kernel_Function_shared(unsigned short *Input_Image, unsigned short *Output_Image, int Image_Width, int Image_Height)
// {
//     const int tx_l = threadIdx.x;                           // --- Local thread x index
//     const int ty_l = threadIdx.y;                           // --- Local thread y index

//     const int tx_g = blockIdx.x * blockDim.x + tx_l;        // --- Global thread x index
//     const int ty_g = blockIdx.y * blockDim.y + ty_l;        // --- Global thread y index

//     __shared__ unsigned short smem[BLOCK_WIDTH+2][BLOCK_HEIGHT+2];

//     // --- Fill the shared memory border with zeros
//     if (tx_l == 0)                      smem[tx_l]  [ty_l+1]    = 0;    // --- left border
//     else if (tx_l == BLOCK_WIDTH-1)     smem[tx_l+2][ty_l+1]    = 0;    // --- right border
//     if (ty_l == 0) {                    smem[tx_l+1][ty_l]      = 0;    // --- upper border
//         if (tx_l == 0)                  smem[tx_l]  [ty_l]      = 0;    // --- top-left corner
//         else if (tx_l == BLOCK_WIDTH-1) smem[tx_l+2][ty_l]      = 0;    // --- top-right corner
//         }   else if (ty_l == BLOCK_HEIGHT-1) {smem[tx_l+1][ty_l+2]  = 0;    // --- bottom border
//         if (tx_l == 0)                  smem[tx_l]  [ty_l+2]    = 0;    // --- bottom-left corder
//         else if (tx_l == BLOCK_WIDTH-1) smem[tx_l+2][ty_l+2]    = 0;    // --- bottom-right corner
//     }

//     // --- Fill shared memory
//                                                                     smem[tx_l+1][ty_l+1] =                           Input_Image[ty_g*Image_Width + tx_g];      // --- center
//     if ((tx_l == 0)&&((tx_g > 0)))                                      smem[tx_l]  [ty_l+1] = Input_Image[ty_g*Image_Width + tx_g-1];      // --- left border
//     else if ((tx_l == BLOCK_WIDTH-1)&&(tx_g < Image_Width - 1))         smem[tx_l+2][ty_l+1] = Input_Image[ty_g*Image_Width + tx_g+1];      // --- right border
//     if ((ty_l == 0)&&(ty_g > 0)) {                                      smem[tx_l+1][ty_l]   = Input_Image[(ty_g-1)*Image_Width + tx_g];    // --- upper border
//             if ((tx_l == 0)&&((tx_g > 0)))                                  smem[tx_l]  [ty_l]   = Input_Image[(ty_g-1)*Image_Width + tx_g-1];  // --- top-left corner
//             else if ((tx_l == BLOCK_WIDTH-1)&&(tx_g < Image_Width - 1))     smem[tx_l+2][ty_l]   = Input_Image[(ty_g-1)*Image_Width + tx_g+1];  // --- top-right corner
//          } else if ((ty_l == BLOCK_HEIGHT-1)&&(ty_g < Image_Height - 1)) {  smem[tx_l+1][ty_l+2] = Input_Image[(ty_g+1)*Image_Width + tx_g];    // --- bottom border
//          if ((tx_l == 0)&&((tx_g > 0)))                                 smem[tx_l]  [ty_l+2] = Input_Image[(ty_g-1)*Image_Width + tx_g-1];  // --- bottom-left corder
//         else if ((tx_l == BLOCK_WIDTH-1)&&(tx_g < Image_Width - 1))     smem[tx_l+2][ty_l+2] = Input_Image[(ty_g+1)*Image_Width + tx_g+1];  // --- bottom-right corner
//     }
//     __syncthreads();

//     // --- Pull the 3x3 window in a local array
//     unsigned short v[9] = { smem[tx_l][ty_l],   smem[tx_l+1][ty_l],     smem[tx_l+2][ty_l],
//                             smem[tx_l][ty_l+1], smem[tx_l+1][ty_l+1],   smem[tx_l+2][ty_l+1],
//                             smem[tx_l][ty_l+2], smem[tx_l+1][ty_l+2],   smem[tx_l+2][ty_l+2] };    

//     // --- Bubble-sort
//     for (int i = 0; i < 5; i++) {
//         for (int j = i + 1; j < 9; j++) {
//             if (v[i] > v[j]) { // swap?
//                 unsigned short tmp = v[i];
//                 v[i] = v[j];
//                 v[j] = tmp;
//             }
//          }
//     }

//     // --- Pick the middle one
//     Output_Image[ty_g*Image_Width + tx_g] = v[4];
// }

// /********/
// /* MAIN */
// /********/
// int main()
// {
//     const int Image_Width = 1580;
//     const int Image_Height = 1050;

//     // --- Open data file
//     ifstream is;         is.open("C:\\Users\\user\\Documents\\Project\\Median_Filter\\Release\\Image_To_Be_Filtered.raw", ios::binary );

//     // --- Get file length
//     is.seekg(0, ios::end);
//     int dataLength = is.tellg();
//     is.seekg(0, ios::beg);

//     // --- Read data from file and close file
//     unsigned short* Input_Image_Host = new unsigned short[dataLength * sizeof(char) / sizeof(unsigned short)];
//     is.read((char*)Input_Image_Host,dataLength);
//     is.close();

//     // --- CUDA warm up
//     unsigned short *forFirstCudaMalloc; gpuErrchk(cudaMalloc((void**)&forFirstCudaMalloc, dataLength * sizeof(unsigned short)));
//     gpuErrchk(cudaFree(forFirstCudaMalloc));

//     // --- Allocate host and device memory spaces 
//     unsigned short *Output_Image_Host = (unsigned short *)malloc(dataLength);
//     unsigned short *Input_Image; gpuErrchk(cudaMalloc( (void**)&Input_Image, dataLength * sizeof(unsigned short))); 
//     unsigned short *Output_Image; gpuErrchk(cudaMalloc((void**)&Output_Image, dataLength * sizeof(unsigned short))); 

//     // --- Copy data from host to device
//     gpuErrchk(cudaMemcpy(Input_Image, Input_Image_Host, dataLength, cudaMemcpyHostToDevice));// copying Host Data To Device Memory For Filtering

//     // --- Grid and block sizes
//     const dim3 grid (iDivUp(Image_Width, BLOCK_WIDTH), iDivUp(Image_Height, BLOCK_HEIGHT), 1);      
//     const dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT, 1); 

//     /**********************************************/
//     /* KERNEL WITH OPTIMIZED USE OF SHARED MEMORY */
//     /**********************************************/

//     cudaFuncSetCacheConfig(Optimized_Kernel_Function_shared, cudaFuncCachePreferShared);
//     Optimized_Kernel_Function_shared<<<grid,block>>>(Input_Image, Output_Image, Image_Width, Image_Height);
//     gpuErrchk(cudaPeekAtLastError());
//     gpuErrchk(cudaDeviceSynchronize());

//     // --- Copy results back to the host
//     gpuErrchk(cudaMemcpy(Output_Image_Host, Output_Image, dataLength, cudaMemcpyDeviceToHost));

//     // --- Open results file, write results and close the file
//     ofstream of2;         of2.open("C:\\Users\\angelo\\Documents\\Project\\Median_Filter\\Release\\Filtered_Image.raw",  ios::binary);
//     of2.write((char*)Output_Image_Host, dataLength);
//     of2.close();

//     cout << "\n Press Any Key To Exit..!!";
//     gpuErrchk(cudaFree(Input_Image));

//     delete Input_Image_Host;
//     delete Output_Image_Host;

//     return 0;
// }