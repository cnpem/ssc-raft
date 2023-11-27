#include "../../inc/include.h"


extern "C"{

	void setGPUParameters(GPU *gpus_parameters, dim3 size_pad, int ngpus, int *gpus)
	{
		/* GPUs */
		gpus_parameters->ngpus = ngpus;
		gpus_parameters->gpus  = gpus;

		/* Initialize Device sizes variables */
		int Nsx = 16;
		int Nsy = 16; 
		int Nsz = 1;

		gpus_parameters->BT  = dim3(Nsx,Nsy,Nsz);
        const int bx         = ( size_pad.x + Nsx - 1 ) / Nsx;	
		const int by         = ( size_pad.y + Nsy - 1 ) / Nsy;
		const int bz         = ( size_pad.z + Nsz - 1 ) / Nsz;
		gpus_parameters->Grd = dim3(bx,by,bz);
	}

    void setReconstructionParameters(CFG *configs, float *parameters_float, int *parameters_int, int *flags)
	{

		/* Set Geometry */
		configs->geometry = parameters_int[0];

		/* Set Reconstruction variables */
		configs->nx = parameters_int[1]; 
		configs->ny = parameters_int[2]; 
		configs->nz = parameters_int[3];

		configs->x  = parameters_float[0];
		configs->y  = parameters_float[1];
		configs->z  = parameters_float[2];
		configs->dx = parameters_float[3]; 
		configs->dy = parameters_float[4];
		configs->dz = parameters_float[5];
		
		/* Set Tomogram (or detector) variables (h for horizontal (nrays) and v for vertical (nslices)) */
		configs->nrays   = parameters_int[4]; 
		configs->nslices = parameters_int[5]; 
		configs->nangles = parameters_int[6];

		configs->h       = parameters_float[6];
		configs->v       = parameters_float[7];
		configs->dh      = parameters_float[8];
		configs->dv      = parameters_float[9];
		configs->dangles = parameters_float[10];

		/* Set Padding */
		configs->padx        = parameters_int[7]; 
		configs->pady        = parameters_int[8];
		configs->padz        = parameters_int[9];

		configs->pad_nrays   = configs->nrays   * ( 1 + 2 * configs->padx ); 
		configs->pad_nangles = configs->nangles * ( 1 + 2 * configs->pady ); 
		configs->pad_nslices = configs->nslices * ( 1 + 2 * configs->padz );

		/* Set General reconstruction variables*/
		configs->pixelDetx        = parameters_float[11];
		configs->pixelDety        = parameters_float[12];
		configs->detector_pixel_x = parameters_float[11];
		configs->detector_pixel_y = parameters_float[12];
		configs->energy           = parameters_float[13];

		configs->lambda           = ( plank * vc          ) / configs->energy;
		configs->wave             = ( 2.0   * float(M_PI) ) / configs->lambda;

		configs->z1x              = parameters_float[14];
		configs->z1y              = parameters_float[15];
		configs->z2x              = parameters_float[16];
		configs->z2y              = parameters_float[17];

		configs->z1               = configs->z1x; 
		configs->z2               = configs->z2x; 
		configs->z12              = configs->z1x + configs->z2x;

		/* Set magnitude [(z1+z2)/z1] according to the beam geometry */
		switch (configs->geometry){
			case 0: /* Parallel */	
				configs->magnx       = 1.0;
				configs->magny       = 1.0;
				configs->magnitude_x = 1.0;
				configs->magnitude_y = 1.0;
				break;
			case 1: /* Conebeam */
				configs->magnx       = ( configs->z1x + configs->z2x ) / configs->z1x;
				configs->magny       = ( configs->z1y + configs->z2y ) / configs->z1y;

				configs->magnitude_x = ( configs->z1x + configs->z2x ) / configs->z1x;
				configs->magnitude_y = ( configs->z1y + configs->z2y ) / configs->z1y;
				break;
			case 2: /* Fanbeam */		
				configs->magnx       = ( configs->z1x + configs->z2x ) / configs->z1x;
				configs->magny       = 1.0;
				configs->magnitude_x = ( configs->z1x + configs->z2x ) / configs->z1x;
				configs->magnitude_y = 1.0;
				break;
			default:
				printf("Parallel case as default! \n");
				break;
		}

		/* Set General variables */

		/* Set Bool variables - Pipeline */
		configs->do_flat_dark_correction = flags[0]; 
		configs->flat_dark_do_log        = flags[1];
		configs->do_phase_filter         = flags[2];
		configs->do_rings                = flags[3];
		configs->do_rotation_offset      = flags[4];
		configs->do_alignment            = flags[5];
		configs->do_reconstruction       = flags[6];

		/* Set Flat/Dark Correction */
		configs->numflats  = parameters_int[10];

		/* Set Phase Filter */
		configs->phase_filter_type           = parameters_int[11]; /* Phase Filter type */
		configs->phase_filter_regularization = parameters_float[18]; /* Phase Filter regularization parameter */

		/* Set Rings */
		configs->rings_block  = parameters_int[12];
		configs->rings_lambda = parameters_float[19];

		/* Set Rotation Axis Correction */
		configs->rotation_axis_offset = parameters_int[13];

		/* Set Reconstruction method variables */
		configs->reconstruction_method                = parameters_int[14];
		configs->reconstruction_filter_type           = parameters_int[15];   /* Reconstruction Filter type */
		configs->reconstruction_filter_regularization = parameters_float[20]; /* Reconstruction Filter regularization parameter */
		configs->reconstruction_regularization        = parameters_float[21]; /* General regularization parameter */

		/* Set Slices */
		configs->reconstruction_start_slice = parameters_int[16]; // Slices: start slice = reconstruction_start_slice, end slice = reconstruction_end_slice
		configs->reconstruction_end_slice   = parameters_int[17]; 
		configs->tomogram_start_slice       = parameters_int[18]; // Slices: start slice = tomogram_start_slice, end slice = tomogram_end_slice
		configs->tomogram_end_slice         = parameters_int[19];

		/* Paralell */

		/* Set FBP */

		/* Set BST */

		/* Set EM RT */
		configs->em_iterations = parameters_int[20];

		/* Set EM FST */

		/* Conical */

		/* Set FDK */

		/* Set EM Conical */

	}

}