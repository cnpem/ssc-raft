// Authors: Gilberto Martinez, Eduardo X Miqueles, Giovanni Baraldi, Paola Ferraz

#include "common/configs.hpp"
#include "common/opt.hpp"
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
        const size_t index           = proj_size.y * proj_size.x * k + proj_size.x * j + i;
        
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
            radon += phantom[(int(y+0.5f)*phantom_size.x + int(x+0.5f))%(phantom_size.x*phantom_size.y) + blockIdx.z*(phantom_size.x*phantom_size.y)];

            x += cos_t;
            y += sin_t;
        }

        projections[k*size_t(proj_size.x*proj_size.y) + proj_size.x*j + i] = radon;
    }

}

extern "C" {
    __global__ void kernel_radon_ss_block(float *projections, 
    float *phantom, float *angles,
    dim3 proj_size, dim3 phantom_size, 
    float ax, float ay)
    {
        int tx = blockIdx.x * blockDim.x + threadIdx.x;
        int ty = blockIdx.y * blockDim.y + threadIdx.y;
        int tz = blockIdx.z * blockDim.z + threadIdx.z;

        float cos, sin, tan, cond, t, x, y, sum;
        int Y, X;

        /* Projection data sizes */
        int nrays    = proj_size.x;
        int nangles  = proj_size.y;
        int nslices  = proj_size.z;
        
        /* Phantom data sizes */
        int sizeImagex  = phantom_size.x;
        int sizeImagey  = phantom_size.y;
        int sizeImagez  = phantom_size.z;

        float tmin = sqrtf(ax * ax + ay * ay);
        float dt   = 2.0f*tmin/(nrays-1);

        float dx = 2.0f*ax/(sizeImagex-1);
        float dy = 2.0f*ay/(sizeImagey-1);

        if ( (tx < nrays) && (ty < nangles) && (tz < nslices) )
            {
            cond = sqrtf(2.0)/2.0;
    
            __sincosf(angles[ty], &sin, &cos);
                    
            if( sin < cond ){
                tan = sin/cos;
                sum = 0;
            
                for(Y = 0; Y < sizeImagey; Y++){
                    y = - ay + Y * dy;
                    t = - tmin + tx * dt;
                    
                    x = (t/cos - y*tan); 
                    X = (float)((x + ax)/dx);

                    // sum += tex2D<float>(texImg, X + 0.5f, Y + 0.5f + (tz*Nangles));
                    sum += phantom[IND((int)(X + 0.5f),(int)(Y + 0.5f + tz * nangles),tz,sizeImagex,sizeImagey)];
                }
            
                // output[ tz*nrays*Nangles + ty*Nrays + tx] = dxy*sum/fabsf(cos);
                projections[ IND(tx,ty,tz,nrays,nangles) ] = dy * sum / fabsf(cos);


            }else{

                tan = sin/cos;
                sum = 0;

                for(X = 0; X < sizeImagex; X++){
                    x = - ax + X * dx;
                    
                    t = - tmin + tx * dt;
                    y = (t/sin - x/tan);
                    Y = (float)((y + ay)/dy);
                    
                    // sum += tex2D<float>(texImg, X + 0.5f, Y + 0.5f + (tz*Nangles));
                    sum += phantom[ IND((int)(X + 0.5f),(int)(Y + 0.5f + tz * nangles),tz,sizeImagex,sizeImagey) ];

                }

                // output[ tz*Nrays*Nangles + ty*Nrays + tx] = dxy*sum/fabsf(sin);
                projections[ IND(tx,ty,tz,nrays,nangles) ] = dx * sum / fabsf(sin);
            }
        }
    }		

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
        
        size_t index = IND(i,j,k,proj_size.x,proj_size.y);
        
        if ( (i >= proj_size.x) || (j >= proj_size.y) || (k >= proj_size.z) ) return;

        int indx, indy;
        float s, x, y, linesum, ctheta, stheta, t;  

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
            
                linesum += phantom[IND(indx,indy,k,phantom_size.x,phantom_size.y)];
            }
        } 
        projections[index] = linesum * dy;	    
    }
}

extern "C" {
    void getRadonRT(float* projection, float* obj, float *angles, 
    dim3 tomo_size, dim3 obj_size, float ax, float ay)
    {
        int nrays   = tomo_size.x;
        int nangles = tomo_size.y;
        int nslices = tomo_size.z;

        dim3 threadsPerBlock(TPBX,TPBY,TPBZ);
        dim3 gridBlock((int)ceil( nrays   / threadsPerBlock.x ) + 1,
                       (int)ceil( nangles / threadsPerBlock.y ) + 1,
                       (int)ceil( nslices / threadsPerBlock.z ) + 1);
        
        Radon_RT_version_sscRadon<<<gridBlock, threadsPerBlock>>>(  projection, obj, angles, 
                                                                    tomo_size, obj_size, 
                                                                    ax, ay);

    }
}

//----------------------
// Radon RT
//----------------------

extern "C"{   

    void getRadonRTGPU(float* projection, float* obj, float *angles, 
    dim3 tomo_size, dim3 obj_size, float ax, float ay, int ngpu)
    {
        HANDLE_ERROR(cudaSetDevice(ngpu));

        int nrays   = tomo_size.x;
        int nangles = tomo_size.y;
        int nslices = tomo_size.z;

        int b;
        int blocksize = min(nslices,64); // Herança do Giovanni -> Mudar

		int nblock = (int)ceil( (float) nslices / blocksize );
        int ptr = 0, subblock;

        // Allocate GPU buffers for the output sinogram
        float *dproj   = opt::allocGPU<float>((size_t)     nrays *    nangles * blocksize);
        float *dobj    = opt::allocGPU<float>((size_t)obj_size.x * obj_size.y * blocksize);
        float *dangles = opt::allocGPU<float>( nangles );
        
        opt::CPUToGPU<float>(angles, dangles, nangles);	

        for(b = 0; b < nblock; b++){
            
            subblock   = min(nslices - ptr, blocksize);

            opt::CPUToGPU<float>(   obj + (size_t)ptr * obj_size.x * obj_size.y, 
                                    dobj, (size_t)obj_size.x * obj_size.y * subblock);

            getRadonRT(dproj, dobj, dangles, 
                        dim3(nrays, nangles, subblock), 
                        dim3(obj_size.x, obj_size.y, subblock), 
                        ax, ay);

            opt::GPUToCPU<float>(   projection + (size_t)ptr * nrays * nangles, 
                                    dproj, (size_t)nrays * nangles * subblock);

            /* Update pointer */
			ptr = ptr + subblock;
        }

        HANDLE_ERROR(cudaDeviceSynchronize());        

        HANDLE_ERROR(cudaFree(dproj));
        HANDLE_ERROR(cudaFree(dobj));
        HANDLE_ERROR(cudaFree(dangles));
        // cudaDeviceReset();
    }

    void getRadonRTMultiGPU(int* gpus, int ngpus, 
        float* projection, float* obj, float *angles, 
        int sizeImagex, int sizeImagey, 
        int nrays, int nangles, int nslices, 
        float ax, float ay)
    {
        int t;
		int blockgpu = (nslices + ngpus - 1) / ngpus;
		int ptr = 0, subblock;
        
        std::vector<std::future<void>> threads;
        threads.reserve(ngpus);

        if ( ngpus == 1 ){
            
            getRadonRTGPU( projection, obj, angles,
                        dim3(nrays, nangles, nslices), 
                        dim3(sizeImagex,sizeImagey,nslices),  
                        ax, ay, gpus[0]);
        
        }else{

            for(t = 0; t < ngpus; t++){ 
                
                subblock   = min(nslices - ptr, blockgpu);

                threads.push_back(std::async( std::launch::async, getRadonRTGPU, 
                                                projection + (size_t)ptr *      nrays *    nangles, 
                                                obj        + (size_t)ptr * sizeImagex * sizeImagey, 
                                                angles, 
                                                dim3(nrays, nangles, subblock), 
                                                dim3(sizeImagex,sizeImagey,subblock), 
                                                ax,ay,
                                                gpus[t]
                                                ));

                /* Update pointer */
                ptr = ptr + subblock;
            }
        
            for(auto& t : threads)
                t.get();
        }
    }

}
