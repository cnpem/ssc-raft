#include "../../inc/include.h"
#include "../../inc/common/complex.hpp"
#include "../../inc/common/types.hpp"
#include "../../inc/common/operations.hpp"
#include "../../inc/common/logerror.hpp"

#define PI 3.141592653589793238462643383279502884

extern "C"{
    __global__ void cone_rebinning_kernel(float *dtomo, float *dctomo, PAR param)
    {
        size_t i = threadIdx.x + blockIdx.x * blockDim.x;
        size_t j = threadIdx.y + blockIdx.y * blockDim.y;
        size_t k = threadIdx.z + blockIdx.z * blockDim.z;
        size_t n = param.Nx * param.Ny * param.Nz;

        size_t index, th_ind, voxel, a_ind, b_ind;
        float bt_coord, th_coord, t_coord, r_coord; 
        float a_coord, b_coord;

        if ( i >= param.Nx || j >= param.Ny || k >= param.Nz ) return;
        
        float dt  = param.pixelDetx / param.magnx; /* X */
        float dr  = param.pixelDety / param.magny; /* Slices */
        float dbt = 2.0f * PI       / param.Nz   ; /* Angles */
        
        /* Conebeam pixel coordinates */
        t_coord  = dt  * (float)i - 0.5 * param.Nx * dt; 
        r_coord  = dr  * (float)j - 0.5 * param.Ny * dr;
        bt_coord = dbt * (float)k; 

        a_coord  = ( ( param.d1x + param.d2x ) * t_coord ) / sqrtf( ( ( param.d1x + param.d2x ) * ( param.d1x + param.d2x ) ) + ( t_coord * t_coord ) );
        b_coord  = ( ( param.d1y + param.d2y ) * r_coord ) / sqrtf( ( ( param.d1y + param.d2y ) * ( param.d1y + param.d2y ) ) + ( r_coord * r_coord ) );
        
        th_coord = bt_coord - atanf( t_coord / ( param.d1x + param.d2x ) );
        
        /* Parallel pixel Indexes */
        a_ind  = (size_t)round( ( a_coord  + 0.5 * param.Nx * dt ) / dt  );
        b_ind  = (size_t)round( ( b_coord  + 0.5 * param.Ny * dr ) / dr  );
        th_ind = (size_t)round( ( th_coord                       ) / dbt ) % param.Nz;
        
        index = param.Nx * ( k      * param.Ny + i     ) + j    ; /* Cone index */
        voxel = param.Nx * ( th_ind * param.Ny + a_ind ) + b_ind; /* Parallel index */

        // if ( voxel < n )
        dtomo[voxel] = dctomo[index];
    }
}

// extern "C"{
//     __global__ void cone_rebinning_kernel2(float *dtomo, float *dctomo, PAR param)
//     {
//         size_t i = threadIdx.x + blockIdx.x * blockDim.x;
//         size_t j = threadIdx.y + blockIdx.y * blockDim.y;
//         size_t k; //= threadIdx.z;
//         size_t n = param.Nx * param.Ny * param.Nz;

//         size_t index, th_ind, voxel, a_ind, b_ind;
//         float bt_coord, th_coord, t_coord, r_coord; 
//         float aux, aux2, auxt, auxr, a_coord, b_coord;

//         if ( i > param.Nx || j > param.Ny ) return;
        
//         /* Step sizes */
//         float dt  = ( 2.0f * param.Dt ) / ( param.Nx ); /* X */
//         float dr  = ( 2.0f * param.Dr ) / ( param.Ny ); /* Slices */
//         float dbt = ( 2.0f * M_PI     ) / ( param.Nz ); /* Angles */

//         for ( k = 0; k < param.Nz; k++){
//             /* Conebeam pixel coordinates */
//             t_coord  = dt  * (float)i - param.Dt; 
//             r_coord  = dr  * (float)j - param.Dr;
//             bt_coord = dbt * (float)k; 

//             /* Parallel beam pixel Coordinates */
//             aux  = t_coord - param.ct - param.st - param.rt;
//             aux2 = r_coord - param.cr - param.sr - param.rr;
//             auxt = ( param.zt * param.zt ) + ( aux  * aux  );
//             auxr = ( param.zt * param.zt ) + ( aux2 * aux2 );

//             a_coord  = ( ( ( t_coord - param.ct - param.st ) * param.zt ) / sqrtf( auxt ) ) + param.rt + param.st;
//             b_coord  = ( ( ( r_coord - param.cr - param.sr ) * param.zt ) / sqrtf( auxr ) ) + param.rr + param.sr;
//             th_coord = bt_coord + atanf( aux / param.zt );

//             /* Parallel pixel Indexes */
//             a_ind  = (size_t)ceil( ( a_coord  + param.Dt ) / dt  ) % param.Nx;
//             b_ind  = (size_t)ceil( ( b_coord  + param.Dr ) / dr  ) % param.Ny;
//             th_ind = (size_t)ceil( ( th_coord            ) / dbt ) % param.Nz;
            
//             index = param.Nx * ( k      * param.Ny + i      ) + j;     /* Conebeam index */
//             voxel = param.Nx * ( th_ind * param.Ny + a_ind  ) + b_ind; /* Parallel beam index */

//             if ( voxel < n )
//                 dtomo[voxel] = dctomo[index];
//         }
//     }
// }

extern "C"{
    void cone_rebinning_cpu(float *tomo, float *conetomo, PAR param)
    {
        size_t index, th_ind, i, j, k, voxel, a_ind, b_ind;
        float bt_coord, th_coord, t_coord, r_coord; 
        float a_coord, b_coord;
    
        float dt  = param.pixelDetx / param.magnx; /* X */
        float dr  = param.pixelDety / param.magny; /* Slices */
        float dbt = 2.0f * PI       / param.Nz   ; /* Angles */

        size_t n = param.Nx * param.Ny * param.Nz;

        for ( i = 0; i < param.Nx; i++){
            for ( j = 0; j < param.Ny; j++){
                for ( k = 0; k < param.Nz; k++){
                                    
                    /* Conebeam pixel coordinates */
                    t_coord  = dt  * (float)i - 0.5 * param.Nx * dt; 
                    r_coord  = dr  * (float)j - 0.5 * param.Ny * dr;
                    bt_coord = dbt * (float)k; 

                    a_coord  = ( ( param.d1x + param.d2x ) * t_coord ) / sqrt( ( ( param.d1x + param.d2x ) * ( param.d1x + param.d2x ) ) + ( t_coord * t_coord ) );
                    b_coord  = ( ( param.d1y + param.d2y ) * r_coord ) / sqrt( ( ( param.d1y + param.d2y ) * ( param.d1y + param.d2y ) ) + ( r_coord * r_coord ) );
                    
                    th_coord = bt_coord - atan( t_coord / ( param.d1x + param.d2x ) );
                    
                    /* Parallel pixel Indexes */
                    a_ind  = (size_t)round( ( a_coord  + 0.5 * param.Nx * dt ) / dt  );
                    b_ind  = (size_t)round( ( b_coord  + 0.5 * param.Ny * dr ) / dr  );
                    th_ind = (size_t)round( ( th_coord                       ) / dbt ) % param.Nz;
                    
                    index = param.Nx * ( k      * param.Ny + i     ) + j    ; /* Cone index */
                    voxel = param.Nx * ( th_ind * param.Ny + a_ind ) + b_ind; /* Parallel index */

                    if ( voxel < n )
                        tomo[voxel] = conetomo[index];
                
                }
            }
        }
    }

    // void cone_rebinning_cpu2(float *tomo, float *conetomo, PAR param) 
    // {
    //     size_t index, th_ind, i, j, k, voxel, a_ind, b_ind;
    //     float bt_coord, th_coord, t_coord, r_coord; 
    //     float aux, aux2, auxt, auxr, a_coord, b_coord;
    
    //     float dt  = ( 2.0f * param.Dt ) / ( param.Nx ); /* X */
    //     float dr  = ( 2.0f * param.Dr ) / ( param.Ny ); /* Slices */
    //     float dbt = ( 2.0f * PI       ) / ( param.Nz ); /* Angles */

    //     size_t n = param.Nx * param.Ny * param.Nz;
    //     float m  = param.zt / param.z2;

    //     for ( i = 0; i < param.Nx; i++){
    //         for ( j = 0; j < param.Ny; j++){
    //             for ( k = 0; k < param.Nz; k++){
                                    
    //                 /* Conebeam pixel coordinates */
    //                 t_coord  = dt  * (float)i - param.Dt; 
    //                 r_coord  = dr  * (float)j - param.Dr;
    //                 bt_coord = dbt * (float)k; 

    //                 /* Parallel beam pixel Coordinates */
    //                 aux  = t_coord - param.ct - param.st - param.rt;
    //                 aux2 = r_coord - param.cr - param.sr - param.rr;
    //                 auxt = ( param.z1 * param.z1 ) + ( aux  * aux  );
    //                 auxr = ( param.z1 * param.z1 ) + ( aux2 * aux2 );

    //                 a_coord  = ( ( ( t_coord - param.ct - param.st ) * param.z1 * m) / sqrt( auxt ) ) + param.rt + param.st;
    //                 b_coord  = ( ( ( r_coord - param.cr - param.sr ) * param.z1 * m) / sqrt( auxr ) ) + param.rr + param.sr;
    //                 th_coord = bt_coord + atan( aux / param.z1 );
                    
    //                 /* Parallel pixel Indexes */
    //                 a_ind  = (size_t)ceil( ( a_coord  + param.Lt ) / dt  ) % param.Nx;
    //                 b_ind  = (size_t)ceil( ( b_coord  + param.Lr ) / dr  ) % param.Ny;
    //                 th_ind = (size_t)ceil( ( th_coord            ) / dbt ) % param.Nz;
                    
    //                 index = param.Nx * ( k      * param.Ny + i     ) + j;
    //                 voxel = param.Nx * ( th_ind * param.Ny + a_ind ) + b_ind;

    //                 if ( voxel < n )
    //                     tomo[voxel] = conetomo[index];
                
    //             }
    //         }
    //     }
    // }
}