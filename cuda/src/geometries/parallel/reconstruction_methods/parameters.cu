#include "common/configs.hpp"
#include "common/logerror.hpp"
#include "geometries/parallel/em.hpp"
#include "geometries/parallel/radon.hpp"
#include "geometries/parallel/bst.hpp"


void setEMParameters(CFG *configs, float *parameters_float, int *parameters_int)
{
    /* Set Geometry */
    configs->geometry.geometry = PARALLEL;
    
    /* Set Tomogram (or detector) variables */
    configs->tomo.size         = dim3(parameters_int[0],parameters_int[1],parameters_int[2]);  

    /* Set padding */
    configs->tomo.padsize      = configs->tomo.size;  

    /* Set Reconstruction variables */
    configs->obj.size          = dim3(parameters_int[3],parameters_int[3],configs->tomo.size.z); 

    configs->em_iterations     = parameters_int[4];

}

void setFBPParameters(CFG *configs, float *parameters_float, int *parameters_int)
{
    /* Set Geometry */
    configs->geometry.geometry           = PARALLEL;

    /* Set Tomogram (or detector) variables */
    configs->tomo.size                   = dim3(parameters_int[0],parameters_int[1],parameters_int[2]); 

    /* Set Padding */
    configs->tomo.pad                    = dim3(parameters_int[4],0,0); 

    int npadx                            = configs->tomo.size.x * ( 1 + 2 * configs->tomo.pad.x ); 
    int npady                            = configs->tomo.size.y; 
    int npadz                            = configs->tomo.size.z; 

    configs->tomo.padsize                = dim3(npadx,npady,npadz); 

    /* Set Reconstruction variables */
    configs->obj.size                    = dim3(parameters_int[3],parameters_int[3],configs->tomo.size.z); 
    
    /* Set magnitude [(z1+z2)/z1] according to the beam geometry (Parallel) */
    configs->geometry.magnitude_x        = 1.0;
    configs->geometry.magnitude_y        = 1.0;
            
    /* Set Reconstruction method variables */
    configs->reconstruction_filter_type  = parameters_int[6];  /* Reconstruction Filter type */
    configs->rotation_axis_offset        = parameters_int[7];  /* Rotation Axis offset */
    
    configs->reconstruction_paganin_reg  = 4.0f * float(M_PI) * float(M_PI) * ( parameters_float[0] == 0.0 ? 0.0 : parameters_float[0] ); /* Reconstruction Filter regularization parameter */
    configs->reconstruction_reg          = parameters_float[1]; /* General regularization parameter */

}