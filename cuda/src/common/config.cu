#include "common/configs.hpp"
#include "common/types.hpp"

extern "C"{

	void printGPUParameters(GPU *gpus_parameters)
	{
		printf("BT: %d, %d, %d \n", gpus_parameters->BT.x, gpus_parameters->BT.y, gpus_parameters->BT.z);
		printf("GD: %d, %d, %d \n", gpus_parameters->Grd.x, gpus_parameters->Grd.y, gpus_parameters->Grd.z);
	}

	void setGPUParameters(GPU *gpus_parameters, dim3 size, int ngpus, int *gpus)
	{
		/* Initialize Device sizes variables */
		int Nsx                = TPBX;
		int Nsy                = TPBY;
		int Nsz                = TPBZ;

		gpus_parameters->BT    = dim3(Nsx,Nsy,Nsz);
		const int bx           = ( size.x + Nsx - 1 ) / Nsx;
		const int by           = ( size.y + Nsy - 1 ) / Nsy;
		const int bz           = ( size.z + Nsz - 1 ) / Nsz;
		gpus_parameters->Grd   = dim3(bx,by,bz);

        gpus_parameters->gpus = gpus;
        gpus_parameters->ngpus = ngpus;
    }

    void setReconstructionParameters(CFG *configs, float *parameters_float, int *parameters_int, int *flags)
	{
		/* Set Geometry */
		// TODO: remove this parameter from ctypes wrapper?
        //configs->geometry.geometry        = parameters_int[0];

		/* Set Reconstruction variables */
		configs->obj.size                 = dim3(parameters_int[0],parameters_int[1],parameters_int[2]);

		//configs->obj.Lx                   = parameters_float[0];
		//configs->obj.Ly                   = parameters_float[1];
		//configs->obj.Lz                   = parameters_float[2];
		//configs->obj.dx                   = parameters_float[3];
		//configs->obj.dy                   = parameters_float[4];
		//configs->obj.dz                   = parameters_float[5];

		/* Set Tomogram (or detector) variables (h for horizontal (nrays) and v for vertical (nslices)) */
		configs->tomo.size                 = dim3(parameters_int[3],parameters_int[4],parameters_int[5]);

		//configs->tomo.Lx                   = parameters_float[9];
		//configs->tomo.Ly                   = parameters_float[10];
		//configs->tomo.Lz                   = parameters_float[11];
		//configs->tomo.dx                   = parameters_float[12];
		//configs->tomo.dy                   = parameters_float[13];
		//configs->tomo.dz                   = parameters_float[14];

		/* Set Padding */
		configs->tomo.pad                  = dim3(parameters_int[6],parameters_int[7],parameters_int[8]);

		//int npadx                          = configs->tomo.size.x * ( 1 + 2 * configs->tomo.pad.x );
		//int npady                          = configs->tomo.size.y * ( 1 + 2 * configs->tomo.pad.y );
		//int npadz                          = configs->tomo.size.z * ( 1 + 2 * configs->tomo.pad.z );

		//configs->tomo.padsize              = dim3(npadx,npady,npadz);

		/* Set General reconstruction variables*/
		configs->geometry.detector_pixel_x = parameters_float[0];
		configs->geometry.detector_pixel_y = parameters_float[1];
		configs->geometry.energy           = parameters_float[2];
		configs->geometry.z1x              = parameters_float[3];
		configs->geometry.z1y              = parameters_float[4];
		configs->geometry.z2x              = parameters_float[5];
		configs->geometry.z2y              = parameters_float[6];

        configs->geometry.magnitude_x = parameters_float[7];
        configs->geometry.magnitude_y = parameters_float[8];

		/* Set wavelenght and wavenumber */
		configs->geometry.wavelength       = ( plank * vc          ) / configs->geometry.energy;
		configs->geometry.wavenumber       = ( 2.0   * float(M_PI) ) / configs->geometry.wavelength;


		/* Set General variables */

		/* Set Bool variables - Pipeline */
		configs->flags.do_flat_dark_correction  = flags[0];
		configs->flags.do_flat_dark_log         = flags[1];
		configs->flags.do_phase_filter          = flags[2];
		configs->flags.do_rings                 = flags[3];
		configs->flags.do_rotation              = flags[4];
		configs->flags.do_rotation_correction   = flags[5];
		configs->flags.do_reconstruction        = flags[6];
        configs->flags.do_rotation_auto_offset = false;

		/* Set Flat/Dark Correction */
		configs->numflats                       = parameters_int[9];

		/* Set Phase Retrieval */
		configs->phase_type                     = parameters_int[10]; /* Phase method type */
		configs->beta_delta                     = parameters_float[9]; /* Phase regularization parameter */

		/* Set Rings */
		configs->rings_block                    = parameters_int[11];
		configs->rings_lambda                   = parameters_float[10];

		/* Set Rotation Axis Correction */
		configs->rotation_axis_offset           = parameters_int[12];

		/* Set Reconstruction method variables */
		configs->reconstruction_method          = parameters_int[13];
		configs->reconstruction_filter_type     = parameters_int[14];   /* Reconstruction Filter type */

		configs->reconstruction_paganin         = configs->geometry.wavelength * configs->geometry.z2x * float(M_PI) * (configs->beta_delta == 0.0f ? 0.0f: (1.0f / configs->beta_delta) ); /* Reconstruction Filter regularization parameter */
		configs->reconstruction_reg             = parameters_float[11]; /* General regularization parameter */

		/* Set Slices on Reconstruction and on Tomogram */
		/* For Parallel and Fanbeam geometry,
		Slices on reconstruction are the SAME as the slices on tomogram.
		For Conebeam geometry,
		Slices on reconstrucion are DIFFERENT as the slices on tomogram.
		*/
		configs->obj.zslice0               = parameters_int[15]; /* Slices: start slice on reconstruction */
		configs->obj.zslice1               = parameters_int[16]; /* Slices: end slice on reconstruction */

		configs->tomo.zslice0              = parameters_int[17]; /* Slices: start slice on tomogram */
		configs->tomo.zslice1              = parameters_int[18]; /* Slices: end slice on tomogram */

		/* Paralell */

		/* Set FBP */

		/* Set BST */

		/* Set EM RT */
		configs->em_iterations                  = parameters_int[19];

		/* Set EM FST */

		/* Conical */

		/* Set FDK */

		/* Set EM Conical */

	}

}
