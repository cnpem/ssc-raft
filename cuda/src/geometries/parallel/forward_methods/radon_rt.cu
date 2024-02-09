// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "common/configs.hpp"
#include "common/operations.hpp"
#include "geometries/parallel/radon.hpp"


extern "C"{

    __global__ void Radon_RT_version_Pyraft_Giov(float* projections, float* phantom, float *angles, dim3 proj_size, dim3 phantom_size)
    {
        /* dim3 proj_size    = (rays,angles,slices) */
        /* dim3 phantom_size = (   x,     y,     z) */
        int i = blockIdx.x*blockDim.x + threadIdx.x; /* rays */
        int j = blockIdx.y*blockDim.y + threadIdx.y; /* angles */
        int k = blockIdx.z*blockDim.z + threadIdx.z; /* slices */
        const size_t total_proj_size = proj_size.x * proj_size.y * proj_size.z;
        const size_t index           = opt::getIndex3d(proj_size);
        
        if ( index >= total_proj_size ) return;
        
        float ct = (float)proj_size.x/2.0f;
        float cos_t = cosf(angles[j] + 1E-5f);
        float sin_t = sinf(angles[j] + 1E-5f);

        float x = ct - 2.0f*ct*cos_t + (i - (float)proj_size.x/2.0f)*sin_t;
        float y = ct - 2.0f*ct*sin_t - (i - (float)proj_size.x/2.0f)*cos_t;

        float tx0 = -x/cos_t;
        float tx1 = (proj_size.x-x)/cos_t;

        float ty0 = -y/sin_t;
        float ty1 = (proj_size.x-y)/sin_t;

        float d1 = fmaxf(fminf(tx0,tx1), fminf(ty0,ty1));
        int d2 = int(ceil(fminf(fmaxf(tx0,tx1), fmaxf(ty0,ty1)) - d1)+0.1f);

        x += d1*cos_t;
        y += d1*sin_t;
        
        float radon = 0;
        for(int s=0; s<d2; s++)
        {
            radon += phantom[(int(y+0.5f)*proj_size.x + int(x+0.5f))%(proj_size.x*proj_size.x) + blockIdx.z*(proj_size.x*proj_size.x)];

            x += cos_t;
            y += sin_t;
        }

        projections[k*size_t(proj_size.x*proj_size.y) + proj_size.x*j + i] = radon;
    }

}

extern "C" {
    __global__ void Radon_RT_version_sscRadon(
                    float *projections, float *phantom, float *angles,
                    dim3 proj_size, dim3 phantom_size, 
                    float ax, float ay)
    {
        /* dim3 proj_size    = (rays,angles,slices) */
        /* dim3 phantom_size = (   x,     y,     z) */
        int i = blockIdx.x*blockDim.x + threadIdx.x; /* rays */
        int j = blockIdx.y*blockDim.y + threadIdx.y; /* angles */
        int k = blockIdx.z*blockDim.z + threadIdx.z; /* slices */
        
        const size_t index = IND(i,j,k,proj_size.x,proj_size.y);
        
        if ( (i >= proj_size.x) || (j >= proj_size.y) || (k >= proj_size.z) ) return;

        size_t ind; int indx, indy;
        float s, x, y, linesum, ctheta, stheta, theta, t;  

        float dx = 2.0f*ax/(phantom_size.x-1);
        float dy = 2.0f*ay/(phantom_size.y-1);

        ctheta = cosf(angles[j]);
        stheta = sinf(angles[j]);

        t = - ax + i * dx; 
        
        linesum = 0.0f;
        for( int indray = 0; indray < phantom_size.y; indray++ ){
            s = - ay + indray * dy;

            x = t * ctheta - s * stheta;
            y = t * stheta + s * ctheta;
            
            indx = (int) ((x + 1)/dx);
            indy = (int) ((y + 1)/dy);	 

            if ((indx >= 0) & (indx < phantom_size.x) & (indy >= 0) & (indy < phantom_size.y) ){
                
                ind      = IND(indx,indy,k,phantom_size.x,phantom_size.y);
                linesum += phantom[ind];
            }
        } 
        projections[index] = linesum * dy;	    
    }
}

// extern "C" {
//     void _radonp_gpu(float* h_output, float* h_input, float *angles, dim3 sizeImage, int nrays, int nangles, int device, int blocksize, float a)
//     {
//         cudaSetDevice(device);
        
//         float *d_output, *d_input, *d_angles;
        
//         // Allocate GPU buffers for the output sinogram
//         cudaMalloc(&d_output, (size_t)nrays       *     nangles * blocksize * sizeof(float) );
//         cudaMalloc(&d_input , (size_t)sizeImage.x * sizeImage.y * blocksize * sizeof(float) );
//         cudaMalloc(&d_angles,                                       nangles * sizeof(float) );
        
//         cudaMemcpy(d_input, h_input, (size_t)sizeImage.x * sizeImage.y * blocksize * sizeof(float), cudaMemcpyHostToDevice);
        
//         cudaMemcpy(d_angles, angles, nangles * sizeof(float), cudaMemcpyHostToDevice);		

//         dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
//         dim3 gridBlock((int)ceil((nrays)/threadsPerBlock.x)+1,
//             (int)ceil((nangles)/threadsPerBlock.y)+1,
//             (int)ceil(blocksize/threadsPerBlock.z)+1);
        
//         Radon_RT_version_sscRadon<<<gridBlock, threadsPerBlock>>>(d_output, d_input, d_angles, sizeImage, nrays, nangles, blocksize, a, a);
        
//         // cudaDeviceSynchronize();
        
//         // Copy output vector from GPU buffer to host memory.
//         cudaMemcpy(h_output, d_output, (size_t)nrays * nangles * blocksize * sizeof(float), cudaMemcpyDeviceToHost);
        
//         //cudaFree(d_input);
//         cudaFree(d_output);
//         cudaFree(d_input);
//         cudaFree(d_angles);
//         cudaDeviceReset();

//         return;
//     }
// }


// //----------------------
// // Radon RT
// //----------------------

// extern "C"{   

//     void RadonGPU(float* h_output, float* h_input, float *angles, 
//         dim3 sizeImage, int nrays, int nangles, int device, int nslices, float a)
//     {
//         int b;
//         int blocksize = min(nslices,64); // Herança do Giovanni -> Mudar

// 		int nblock = (int)ceil( (float) nslices / blocksize );
//         int ptr = 0, subblock;

//         for(b = 0; b < nblock; b++){
            
//             subblock   = min(nslices - ptr, blocksize);

//             _radonp_gpu(
//                         h_output + (size_t)ptr *       nrays *     nangles, 
//                         h_input  + (size_t)ptr * sizeImage.x * sizeImage.y, 
//                         angles, 
//                         sizeImage, nrays, nangles, device, subblock, 
//                         a
//                         );

//             /* Update pointer */
// 			ptr = ptr + subblock;

//         }

//         cudaDeviceSynchronize();

//     }

//     void RadonMultiGPU(int* gpus, int ngpus, float* h_output, float* h_input, float *angles, 
//         int sizeImagex, int sizeImagey, int nrays, int nangles, int nslices, float a)
//     {
//         int t;
// 		int blockgpu = (nslices + ngpus - 1) / ngpus;
// 		int ptr = 0, subblock;
        
//         std::vector<std::future<void>> threads;

//         if ( ngpus == 1 ){
            
//             RadonGPU(h_output, h_input, angles, dim3(sizeImagex,sizeImagey,nslices), nrays, nangles, gpus[0], nslices, a);
        
//         }else{

//             for(t = 0; t < ngpus; t++){ 
                
//                 subblock   = min(nslices - ptr, blockgpu);

//                 threads.push_back(std::async( std::launch::async, RadonGPU, 
//                                                 h_output + (size_t)ptr *      nrays *    nangles, 
//                                                 h_input  + (size_t)ptr * sizeImagex * sizeImagey, 
//                                                 angles, 
//                                                 dim3(sizeImagex,sizeImagey,subblock), 
//                                                 nrays, nangles, 
//                                                 gpus[t], 
//                                                 subblock, 
//                                                 a
//                                                 ));

//                 /* Update pointer */
//                 ptr = ptr + subblock;
//             }
        
//             for(auto& t : threads)
//                 t.get();
//         }
//     }

// }
