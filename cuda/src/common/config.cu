#include "common/configs.hpp"
#include "common/types.hpp"

extern "C"{

    // void setTomoParameters(CFG *configs, int nrays, int nangles, int nslices, 
    // int padx, int pady, int padz, int blocksize)
    // {
    //     /* Set Tomogram variables */
    //     configs->tomo.size = dim3(nrays,nangles,nslices);  

    //     /* Set padding */
        
    //     /* Pad is the integer number such that the total padding is = ( pad + 1 ) * dimension 
    //     Example: 
    //         - Data have dimension on x-axis of nx = 2048;
    //         - The padx = 1;
    //         - The new dimension is nx_pad = nx * (1 + padx) = 4096
    //     */
    //     configs->tomo.pad = dim3(padx, pady, padz); 

    //     /* Padsize is the final dimension with padding. 
    //     Example:
    //         - Data have dimension on x-axis of nx = 2048 and padx = 1
    //         - padsizex = nx_pad = nx * (1 + padx) = 4096
    //         - See Pad example above. 
    //     */
    //     // configs->tomo.padsize = dim3(configs->tomo.size.x * ( 1 + configs->tomo.pad.x ),configs->tomo.size.y * ( 1 + configs->tomo.pad.y ),configs->tomo.size.z);

    //     /* GPU blocksize */
    //     configs->blocksize = blocksize;
    // }

    // void setObjParameters(CFG *configs, int nx, int ny, int nz, 
    // int padx, int pady, int padz)
    // {
    //     /* Set Detector variables */
    //     configs->obj.size = dim3(nx,ny,nz);  

    //     /* Set padding */
        
    //     /* Pad is the integer number such that the total padding is = ( pad + 1 ) * dimension 
    //     Example: 
    //         - Data have dimension on x-axis of nx = 2048;
    //         - The padx = 1;
    //         - The new dimension is nx_pad = nx * (1 + padx) = 4096
    //     */
    //     configs->obj.pad = dim3(padx, pady, padz); 

    //     /* Padsize is the final dimension with padding. 
    //     Example:
    //         - Data have dimension on x-axis of nx = 2048 and padx = 1
    //         - padsizex = nx_pad = nx * (1 + padx) = 4096
    //         - See Pad example above. 
    //     */
    //     // configs->obj.padsize = dim3(configs->obj.size.x * ( 1 + configs->obj.pad.x ),configs->obj.size.y * ( 1 + configs->obj.pad.y ),configs->obj.size.z);

    // }

    // void setGeometryParameters(CFG *configs, float detector_pixel_x_meters, float detector_pixel_y_meters, 
    //     float energy_eV, float z2_x_meters, float z2_y_meters, float magnitude_x, float magnitude_y)
    // {
    //     /* Set Geometry */
    //     configs->geometry.detector_pixel_x = detector_pixel_x_meters;
    //     configs->geometry.detector_pixel_y = detector_pixel_y_meters;
    //     configs->geometry.energy           = energy_eV;
    //     configs->geometry.z2x              = z2_x_meters;
    //     configs->geometry.z2y              = z2_y_meters;
    //     configs->geometry.magnitude_x      = magnitude_x;
    //     configs->geometry.magnitude_y      = magnitude_y;
    //     configs->geometry.wavelength       = ( ( plank * vc ) / configs->geometry.energy );

    //     configs->geometry.obj_pixel_x      = configs->geometry.detector_pixel_x / configs->geometry.magnitude_x;
    //     configs->geometry.obj_pixel_y      = configs->geometry.detector_pixel_y / configs->geometry.magnitude_y;

    //     configs->geometry.z2x             /= configs->geometry.magnitude_x;
    //     configs->geometry.z2y             /= configs->geometry.magnitude_y; 
    // }

}
