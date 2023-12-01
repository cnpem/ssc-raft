#include "../../inc/include.h"
#include "../../inc/common/config.h"


extern "C"{

	void setGPUParameters(GPU *gpus_parameters, dim3 size_pad, int ngpus, int *gpus)
	{
		/* GPUs */
		gpus_parameters->ngpus = ngpus;
		gpus_parameters->gpus  = gpus;

		/* Initialize Device sizes variables */
		int Nsx                = 16;
		int Nsy                = 16; 
		int Nsz                = 1;

		gpus_parameters->BT    = dim3(Nsx,Nsy,Nsz);
        const int bx           = ( size_pad.x + Nsx - 1 ) / Nsx;	
		const int by           = ( size_pad.y + Nsy - 1 ) / Nsy;
		const int bz           = ( size_pad.z + Nsz - 1 ) / Nsz;
		gpus_parameters->Grd   = dim3(bx,by,bz);
	}

    void setReconstructionParameters(CFG *configs, float *parameters_float, int *parameters_int, int *flags)
	{
		/* Set Geometry */
		configs->geometry.geometry         = parameters_int[0];

		/* Set Reconstruction variables */
		configs->recon.size                = dim3(parameters_int[1],parameters_int[2],parameters_int[3]); 

		configs->recon.x                   = parameters_float[0];
		configs->recon.y                   = parameters_float[1];
		configs->recon.z                   = parameters_float[2];
		configs->recon.dx                  = parameters_float[3]; 
		configs->recon.dy                  = parameters_float[4];
		configs->recon.dz                  = parameters_float[5];
		
		/* Set Tomogram (or detector) variables (h for horizontal (nrays) and v for vertical (nslices)) */
		configs->tomo.size                 = dim3(parameters_int[4],parameters_int[5],parameters_int[6]); 

		configs->tomo.nrays                = parameters_float[6];
		configs->tomo.nangles              = parameters_float[7];
		configs->tomo.nslices              = parameters_float[8];

		configs->tomo.x                    = parameters_float[9];
		configs->tomo.y                    = parameters_float[10];
		configs->tomo.z                    = parameters_float[11];
		configs->tomo.dx                   = parameters_float[12];
		configs->tomo.dy                   = parameters_float[13];
		configs->tomo.dz                   = parameters_float[14];


		/* Set Padding */
		configs->tomo.pad                  = dim3(parameters_int[7],parameters_int[8],parameters_int[9]); 

		int npadx                          = configs->tomo.size.x * ( 1 + 2 * configs->tomo.pad.x ); 
		int npady                          = configs->tomo.size.y * ( 1 + 2 * configs->tomo.pad.y ); 
		int npadz                          = configs->tomo.size.z * ( 1 + 2 * configs->tomo.pad.z );

		configs->tomo.npad                 = dim3(npadx,npady,npadz); 

		/* Set General reconstruction variables*/
		configs->geometry.detector_pixel_x = parameters_float[15];
		configs->geometry.detector_pixel_y = parameters_float[16];
		configs->geometry.energy           = parameters_float[17];
		configs->geometry.z1x              = parameters_float[18];
		configs->geometry.z1y              = parameters_float[19];
		configs->geometry.z2x              = parameters_float[20];
		configs->geometry.z2y              = parameters_float[21];

		/* Set wavelenght (lambda) and wavenumber (wave) */
		configs->geometry.lambda           = ( plank * vc          ) / configs->geometry.energy;
		configs->geometry.wave             = ( 2.0   * float(M_PI) ) / configs->geometry.lambda;

		/* Set magnitude [(z1+z2)/z1] according to the beam geometry */
		switch (configs->geometry.geometry){
			case 0: /* Parallel */	
				configs->geometry.magnitude_x = 1.0;
				configs->geometry.magnitude_y = 1.0;
				break;
			case 1: /* Conebeam */
				configs->geometry.magnitude_x = ( configs->geometry.z1x + configs->geometry.z2x ) / configs->geometry.z1x;
				configs->geometry.magnitude_y = ( configs->geometry.z1y + configs->geometry.z2y ) / configs->geometry.z1y;
				break;
			case 2: /* Fanbeam */		
				configs->geometry.magnitude_x = ( configs->geometry.z1x + configs->geometry.z2x ) / configs->geometry.z1x;
				configs->geometry.magnitude_y = 1.0;
				break;
			default:
				printf("Parallel case as default! \n");
				configs->geometry.magnitude_x = 1.0;
				configs->geometry.magnitude_y = 1.0;
				break;
		}

		/* Set General variables */

		/* Set Bool variables - Pipeline */
		configs->flags.do_flat_dark_correction  = flags[0]; 
		configs->flags.do_flat_dark_log         = flags[1];
		configs->flags.do_phase_filter          = flags[2];
		configs->flags.do_rings                 = flags[3];
		configs->flags.do_rotation_offset       = flags[4];
		configs->flags.do_alignment             = flags[5];
		configs->flags.do_reconstruction        = flags[6];

		/* Set Flat/Dark Correction */
		configs->numflats                       = parameters_int[10];

		/* Set Phase Filter */
		configs->phase_filter_type              = parameters_int[11]; /* Phase Filter type */
		configs->phase_filter_reg               = parameters_float[19]; /* Phase Filter regularization parameter */

		/* Set Rings */
		configs->rings_block                    = parameters_int[12];
		configs->rings_lambda                   = parameters_float[22];

		/* Set Rotation Axis Correction */
		configs->rotation_axis_offset           = parameters_int[13];

		/* Set Reconstruction method variables */
		configs->reconstruction_method          = parameters_int[14];
		configs->reconstruction_filter_type     = parameters_int[15];   /* Reconstruction Filter type */

		float paganin_reg                       = ( parameters_float[23] == 0.0 ? 0.0 : parameters_float[23] ); /* Reconstruction Filter regularization parameter */
		configs->reconstruction_paganin_reg     = 4.0f * float(M_PI) * float(M_PI) * configs->geometry.z2x * paganin_reg * configs->geometry.lambda  / ( configs->geometry.magnitude_x );
		configs->reconstruction_reg             = parameters_float[24]; /* General regularization parameter */

		/* Set Slices on Reconstruction and on Tomogram */
		/* For Parallel and Fanbeam geometry, 
		Slices on reconstrucion are the SAME as the slices on tomogram.
		For Conebeam geometry, 
		Slices on reconstrucion are DIFFERENT as the slices on tomogram. 
		*/
		configs->recon.start_slice              = parameters_int[16]; /* Slices: start slice on reconstruction */  
		configs->recon.end_slice                = parameters_int[17]; /* Slices: end slice on reconstruction */ 

		configs->tomo.start_slice               = parameters_int[18]; /* Slices: start slice on tomogram */ 
		configs->tomo.end_slice                 = parameters_int[19]; /* Slices: end slice on tomogram */ 

		/* Paralell */

		/* Set FBP */

		/* Set BST */

		/* Set EM RT */
		configs->em_iterations                  = parameters_int[20];

		/* Set EM FST */

		/* Conical */

		/* Set FDK */

		/* Set EM Conical */

	}

	void setPhaseFilterParameters(GEO *geometry, DIM *tomo, float *parameters_float, int *parameters_int)
	{
		/* Set Geometry */
		geometry->geometry = parameters_int[0];
		
		/* Set Tomogram (or detector) variables (h for horizontal (nrays) and v for vertical (nslices)) */
		tomo->size         = dim3(parameters_int[1],parameters_int[2],parameters_int[3]); 

		tomo->x            = parameters_float[0];
		tomo->y            = parameters_float[1];
		tomo->z            = parameters_float[2];
		tomo->dx           = parameters_float[3];
		tomo->dy           = parameters_float[4];
		tomo->dz           = parameters_float[5];

		/* Set Padding */
		tomo->pad          = dim3(parameters_int[4],parameters_int[5],parameters_int[6]); 

		int npadx          = tomo->size.x * ( 1 + 2 * tomo->pad.x ); 
		int npady          = tomo->size.y * ( 1 + 2 * tomo->pad.y ); 
		int npadz          = tomo->size.z * ( 1 + 2 * tomo->pad.z );

		tomo->npad         = dim3(npadx,npady,npadz); 

		/* Set General reconstruction variables*/
		geometry->energy   = parameters_float[6];

		geometry->lambda   = ( plank * vc          ) / geometry->energy;
		geometry->wave     = ( 2.0   * float(M_PI) ) / geometry->lambda;

		geometry->z1x      = parameters_float[7];
		geometry->z1y      = parameters_float[8];
		geometry->z2x      = parameters_float[9];
		geometry->z2y      = parameters_float[10];

		/* Set magnitude [(z1+z2)/z1] according to the beam geometry */
		switch (geometry->geometry){
			case 0: /* Parallel */	
				geometry->magnitude_x = 1.0;
				geometry->magnitude_y = 1.0;
				break;
			case 1: /* Conebeam */
				geometry->magnitude_x = ( geometry->z1x + geometry->z2x ) / geometry->z1x;
				geometry->magnitude_y = ( geometry->z1y + geometry->z2y ) / geometry->z1y;
				break;
			case 2: /* Fanbeam */		
				geometry->magnitude_x = ( geometry->z1x + geometry->z2x ) / geometry->z1x;
				geometry->magnitude_y = 1.0;
				break;
			default:
				printf("Parallel case as default! \n");
				geometry->magnitude_x = 1.0;
				geometry->magnitude_y = 1.0;
				break;
		}

	}

}