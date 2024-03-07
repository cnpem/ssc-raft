#include "common/configs.hpp"
#include "common/logerror.hpp"
#include "geometries/parallel/em.hpp"
#include "geometries/parallel/radon.hpp"
#include "geometries/parallel/bst.hpp"

extern "C"{
    void setEMRTParameters(CFG *configs, float *parameters_float, int *parameters_int)
    {
        /* Set Geometry */
        configs->geometry.geometry = PARALLEL;
        
        /* Set Tomogram (or detector) variables */
        configs->tomo.size     = dim3(parameters_int[0],parameters_int[1],parameters_int[2]);  

        /* Set padding */
        
        /* Pad is the integer number such that the total padding is = ( pad + 1 ) * dimension 
        Example: 
            - Data have dimension on x-axis of nx = 2048;
            - The padx = 1;
            - The new dimension is nx_pad = nx * (1 + padx) = 4096
        */
        configs->tomo.pad      = dim3(parameters_int[4],0,0); //dim3(parameters_int[4],parameters_int[5],parameters_int[6]);

        /* Padsize is the final dimension with padding. 
        Example:
            - Data have dimension on x-axis of nx = 2048 and padx = 1
            - padsizex = nx_pad = nx * (1 + padx) = 4096
            - See Pad example above. 
        */
        configs->tomo.padsize  = configs->tomo.size;
        
        /* Number of flats */
        configs->numflats      = 1; // parameters_int[7];

        /* Set Reconstruction variables */
        configs->obj.size      = dim3(parameters_int[3],parameters_int[3],configs->tomo.size.z); 

        configs->em_iterations = parameters_int[8];

    }

    void printEMRTParameters(CFG *configs)
    {
        printf("Tomo size: %d, %d, %d \n",configs->tomo.size.x,configs->tomo.size.y,configs->tomo.size.z);
        printf("Tomo Pad: %d, %d, %d \n",configs->tomo.pad.x,configs->tomo.pad.y,configs->tomo.pad.z);
        printf("Tomo Padsize: %d, %d, %d \n",configs->tomo.padsize.x,configs->tomo.padsize.y,configs->tomo.padsize.z);
        printf("Recon size: %d, %d, %d \n",configs->obj.size.x,configs->obj.size.y,configs->obj.size.z);
        printf("Nflats: %d \n", configs->numflats);
        printf("EM iterations: %d \n", configs->em_iterations);
    }

    void setEMFQParameters(CFG *configs, float *parameters_float, int *parameters_int)
    {
        /* Set Geometry */
        configs->geometry.geometry = PARALLEL;

        /* Detector pixel size [m] */
        configs->geometry.detector_pixel_x    = parameters_float[0];
        configs->geometry.detector_pixel_y    = parameters_float[1];
        
        /* Set Tomogram (or detector) variables */
        configs->tomo.size         = dim3(parameters_int[0],parameters_int[1],parameters_int[2]);  

        /* Set padding */
        
        /* Pad is the integer number such that the total padding is = ( pad + 1 ) * dimension 
        Example: 
            - Data have dimension on x-axis of nx = 2048;
            - The padx = 1;
            - The new dimension is nx_pad = nx * (1 + padx) = 4096
        */
        configs->tomo.pad      = dim3(parameters_int[4],0,0); //dim3(parameters_int[4],parameters_int[5],parameters_int[6]);

        /* Padsize is the final dimension with padding. 
        Example:
            - Data have dimension on x-axis of nx = 2048 and padx = 1
            - padsizex = nx_pad = nx * (1 + padx) = 4096
            - See Pad example above. 
        */
        configs->tomo.padsize  = dim3(configs->tomo.size.x * ( 1 + configs->tomo.pad.x),configs->tomo.size.y,configs->tomo.size.z);
        
        /* Number of flats */
        configs->numflats      = 1; // parameters_int[7];

        /* Set Reconstruction variables */
        configs->obj.size          = dim3(parameters_int[3],parameters_int[3],configs->tomo.size.z); 

        /* EM Frequency parameters */
        configs->em_iterations     = parameters_int[8];
        configs->interpolation     = parameters_int[9];
    
        /* Total variation regularization parameter */
        configs->reconstruction_tv = 0.0f; // parameters_float[2];

    }

    void printEMFQParameters(CFG *configs)
    {
        printf("Tomo size: %d, %d, %d \n",configs->tomo.size.x,configs->tomo.size.y,configs->tomo.size.z);
        printf("Tomo Pad: %d, %d, %d \n",configs->tomo.pad.x,configs->tomo.pad.y,configs->tomo.pad.z);
        printf("Tomo Padsize: %d, %d, %d \n",configs->tomo.padsize.x,configs->tomo.padsize.y,configs->tomo.padsize.z);
        printf("Recon size: %d, %d, %d \n",configs->obj.size.x,configs->obj.size.y,configs->obj.size.z);
        printf("Nflats: %d \n", configs->numflats);
        printf("EM FQ iterations: %d \n", configs->em_iterations);
        printf("EM FQ interpolation: %d \n", configs->interpolation);
        printf("EM FQ detector pixel size [m]: (%e, %e) \n", configs->geometry.detector_pixel_x,configs->geometry.detector_pixel_y);
        printf("EM FQ total variation parameter: %e \n", configs->reconstruction_tv);

    }

    void setFBPParameters(CFG *configs, float *parameters_float, int *parameters_int)
    {
        /* Set Geometry */
        configs->geometry.geometry = PARALLEL;

        /* Set Tomogram (or detector) variables */
        configs->tomo.size     = dim3(parameters_int[0],parameters_int[1],parameters_int[2]);  

        /* Set padding */
        
        /* Pad is the integer number such that the total padding is = ( pad + 1 ) * dimension 
        Example: 
            - Data have dimension on x-axis of nx = 2048;
            - The padx = 1;
            - The new dimension is nx_pad = nx * (1 + padx) = 4096
        */
        configs->tomo.pad      = dim3(parameters_int[4],0,0); //dim3(parameters_int[4],parameters_int[5],parameters_int[6]);

        /* Padsize is the final dimension with padding. 
        Example:
            - Data have dimension on x-axis of nx = 2048 and padx = 1
            - padsizex = nx_pad = nx * (1 + padx) = 4096
            - See Pad example above. 
        */
        configs->tomo.padsize  = dim3(configs->tomo.size.x * ( 1 + configs->tomo.pad.x),configs->tomo.size.y,configs->tomo.size.z);
        
        /* Compute memory in bytes of a single slice for Tomogram and its padded version for FFT */
        configs->tomo.lenght_memory_bytes    = static_cast<float>(sizeof(float)) *    configs->tomo.size.x;
        configs->tomo.width_memory_bytes      = static_cast<float>(sizeof(float)) *    configs->tomo.size.y;

        configs->tomo.slice_memory_bytes      = configs->tomo.lenght_memory_bytes * configs->tomo.width_memory_bytes;
        configs->tomo.slice_padd_memory_bytes = static_cast<float>(sizeof(float)) * configs->tomo.padsize.x * configs->tomo.padsize.y;

        /* Set Reconstruction variables */
        configs->obj.size      = dim3(parameters_int[3],parameters_int[3],configs->tomo.size.z); 

        /* Compute memory in bytes of a single slice for Tomogram */
        configs->obj.slice_memory_bytes = static_cast<float>(sizeof(float)) * configs->obj.size.x * configs->obj.size.y;

        /* Set magnitude [(z1+z2)/z1] according to the beam geometry (Parallel) */
        configs->geometry.magnitude_x        = 1.0;
        configs->geometry.magnitude_y        = 1.0;
                
        /* Set Reconstruction method variables */
        configs->reconstruction_filter_type  = parameters_int[7];  /* Reconstruction Filter type */
        configs->rotation_axis_offset        = parameters_int[8];  /* Rotation Axis offset */
        
        configs->reconstruction_paganin_reg  = 4.0f * float(M_PI) * float(M_PI) * parameters_float[0]; /* Reconstruction Filter regularization parameter */
        configs->reconstruction_reg          = parameters_float[1]; /* General regularization parameter */
    
        /* Compute total memory used of FBP method on a singles slice */
        configs->total_required_mem_per_slice = (size_t) (
                configs->tomo.slice_memory_bytes      + // Tomo slice
                configs->obj.slice_memory_bytes       + // Reconstructed object slice
                configs->tomo.slice_padd_memory_bytes + // FFT slice
                configs->tomo.lenght_memory_bytes     + // FBP filter kernel
                configs->tomo.width_memory_bytes        // angles
                );      
    
    }
    
    void printFBPParameters(CFG *configs)
    {
        printf("Tomo size: %d, %d, %d \n",configs->tomo.size.x,configs->tomo.size.y,configs->tomo.size.z);
        printf("Tomo Pad: %d, %d, %d \n",configs->tomo.pad.x,configs->tomo.pad.y,configs->tomo.pad.z);
        printf("Tomo Padsize: %d, %d, %d \n",configs->tomo.padsize.x,configs->tomo.padsize.y,configs->tomo.padsize.z);
        printf("Recon size: %d, %d, %d \n",configs->obj.size.x,configs->obj.size.y,configs->obj.size.z);
        printf("FBP filter: %d \n", configs->reconstruction_filter_type);
        printf("FBP regularization: %e \n", configs->reconstruction_reg);
        printf("FBP rot axis offset: %d \n", configs->rotation_axis_offset);
        printf("FBP regularization Paganin: %e \n", configs->reconstruction_paganin_reg);
    }
}