Examples
========

We refer to the Scientific Computing Group (GCC) page `https://gcc.lnls.br/ <https://gcc.lnls.br/>`_ for a more complete sscRaft usage examples and tutorials. 
Link - `https://gcc.lnls.br/wiki/docs/ssc-raft/ <https://gcc.lnls.br/wiki/docs/ssc-raft/>`_.

FDK
***

* FDK (Feldkamp, Davis and Kress) Reconstruction Algorithm

The FDK Reconstruction Algorithm is a popular method for three-dimensional reconstruction from cone-beam projections. 
It was developed in 1984 by Feldkamp, Davis and Kress as a practical geometry adaptation of existing analytical Filtered Backprojection strategies for reconstruction.

This reconstruction method consists of:

- Filtering conical projections with Fourier Transforms
- Backprojecting to sample reconstructions

* sscRaft version 2.1.0 example:

	.. code-block:: python
		
			import numpy as np
			import h5py
			import sscRaft

			in_path = 'path_to_input_HDF5_file'
			in_name = 'name_input_HDF5_file'

			# Load data, flat and dark
			# Mogno HDF5 example:

			data = h5py.File(in_path + in_name, "r")["scan"]["detector"]["data"][:].astype(np.float32)
			flat = h5py.File(in_path + in_name, "r")["scan"]["detector"]["flats"][:].astype(np.float32)
			dark = h5py.File(in_path + in_name, "r")["scan"]["detector"]["darks"][:].astype(np.float32)

			# Dictionary
			experiment = {}
			experiment['z1[m]'] = 2103*1e-3 # Source-Sample distance (float). Defaults to 500*1e-3
			experiment['z1+z2[m]'] = 2530.08*1e-3 # Source-Detector distance (float). Defauts to 1.0
			experiment['detectorPixel[m]'] = 3.61*1e-6 # Detector pixel size (float). Defaults to 1.44*1e-6
			experiment['reconSize'] = 2048 # Recons dimension (cube) (int). Defaults to tomogram shape[0]
			experiment['gpu'] = [0,1,2,3] # List of GPUs (int list). Defaults to [0]
			experiment['rings'] = (True,2) # Rings parameters: (bool,int) = (Apply ring or not, ring blocks: recommended 2 or 4). Defaults to (True,2)
			experiment['normalize'] = (True,True) # Flat normalization: (bool,bool) = (Do normalization, use -log). Defaults to (True,True)
			experiment['padding'] = 800 # Zero pad for fiter: (int). Defaults to 0.
			experiment['shift'] = (True,0) # Rotation shift: (bool,int) = (Use automatic find_rotation function, shift value - can be negative). Defaults to (True,0)
			experiment['detector type'] = 'pco' # Choose detector used: (string) = 'pco', 'mobpix', 'pimegaSi' or 'pimegaCdTe'. Defaults to 'pco'

			recon = sscRaft.reconstruction_fdk(experiment, data, flat, dark)

* sscRaft version 2.0.1 example:

	.. code-block:: python

			import numpy as np
			import h5py
			import sscRaft

			in_path = 'path_to_input_HDF5_file'
			in_name = 'name_input_HDF5_file'

			# Load data, flat and dark
			# Mogno HDF5 example:

			data = h5py.File(in_path + in_name, "r")["scan"]["detector"]["data"][:].astype(np.float32)
			flat = h5py.File(in_path + in_name, "r")["scan"]["detector"]["flats"][:].astype(np.float32)[0,:,:]
			dark = h5py.File(in_path + in_name, "r")["scan"]["detector"]["darks"][:].astype(np.float32)[0,:,:]

			# Dictionary
			experiment = {}
			experiment['z1'] = 2103*1e-3
			experiment['z2'] = 2530.08*1e-3
			experiment['pixel'] = 3.61*1e-6
			experiment['n'] = 2048
			experiment['gpus'] = np.array([0,1,2,3])
			experiment['apply_rings'] = True
			experiment['normalize'] = True
			experiment['padding'] = 800

			recon = sscRaft.reconstruction_fdk(experiment, data, flat, dark)


Save HDF5 with metadata
***********************

How to save a numpy array in HDF5 format with metadata from a dictionary together with the ssc-reft version, for a reconstruct data from FDK.

	.. code-block:: python

			import sscRaft
			import h5py

			in_path = 'path_to_input_HDF5_file'
			in_name = 'name_input_HDF5_file'

			# Load data, flat and dark
			# set dictionary
			experiment = {}

			# set sscRaft dictionary as previous examples

			recon = sscRaft.reconstruction_fdk(experiment, data, flat, dark)

			out_path = 'path_to_output_HDF5_file'
			out_name = 'name_output_HDF5_file'
			ext = '.hdf5'

			# Add more parameters on dictionary if necessary to save in output file
			experiment['Input file'] = in_path + in_name
			experiment['Energy [KeV]'] = '22 and 39'

			# Append an existing HDF5 file
			outfile = h5py.File(out_path + out_name + ext,'a')
			# Or create a new HDF5 file
			# outfile = h5py.File(out_path + out_name + ext,'w')


			try:
					# Call function to save the metadata from dictionary 'experiment' with the software 'sscRaft' and its version 'sscRaft.__version__'
					sscRaft.Metadata_hdf5(outputFileHDF5 = outfile, dic = experiment, software = 'sscRaft', version = sscRaft.__version__)
			except:
					print("Error! Cannot save metadata in HDF5 output file.")
					pass

			# Save reconstruction to HDF5 output file
			outfile.create_dataset('recon', data = recon)

EM/TV
*****

Expectation Maximization with total variation using a parallel tomogram as an input: 

	.. code-block:: python

			import numpy
			import matplotlib.pyplot as plt
			import time
			
			from sscPhantom import mario
			import sscRaft 

			start = time.time()
			
			dic = {'gpu': [0,1,2,3], 'blocksize':16, 'nangles': 309}

			tomop = radon.tomogram(phantom, dic, 'parallel')

			elapsed = time.time() - start
			
			print('Elapsed time for parallel tomogram (sec):', elapsed )

			#########
			sino  = numpy.copy(tomop)
			nangles = 309
			recsize = 510

			dic = {'gpu': [0,1,2,3], 'blocksize':16, 'nangles': nangles, 'niterations': [20,1,1], 
				'regularization': 1,  'epsilon': 1e-15, 'method': 'EM/TV'}

			start = time.time()

			output, rad = sscRaft.emfs( sino, dic )

			elapsed = time.time() - start

			print('Elapsed time for parallel EM recon (sec):', elapsed )


CAT
***

EM/TV from real ptychographic data restored using package ``ssc-cdi``. After a full
ptychographic 3D reconstruction, we obtain a sequence of parallel sinograms, which
can be considered approximate Radon transforms. A 3D inversion follows using the
code below:

	.. code-block:: python

		import sscRaft 
		from sscRadon import radon
		import numpy
		import time

		mdata = numpy.load(<my_data.npy>)

		## preprocessing measured data
		
		nproc = 144

		start = time.time()
		tmp = radon.get_wiggle( new, "vertical", nproc, ref )
		print('Elapsed wiggle vertical:',time.time()-start)
		
		start = time.time()
		tmp = radon.get_wiggle( tmp, "horizontal", nproc, ref)
		print('Elapsed wiggle horizontal:',time.time()-start)

		data = numpy.copy(tmp)
		
		###
		
		sino = numpy.swapaxes( data, 0, 1)
		nangles = sino.shape[1]
		recsize = sino.shape[2]
		
		dic = {'gpu': [0,1,2,3], 'blocksize':20, 'nangles': nangles, 'niterations': [20,1,1], 
		'regularization': 1,  'epsilon': 1e-15, 'method': 'tEM'}

		start = time.time()

		output, rad = sscRaft.emfs( sino, dic )
		
		elapsed = time.time() - start
		
		print('Elapsed time for parallel EM recon (sec):', elapsed )

  Note that ``sino`` is a transposition from ``data`` in order to use ``ssc-raft`` usual axis order
  :math:`slice \times angles \times rays` 


REBINNING
*********

Conebeam tomogram rebinning to parallel tomogram: 

	.. code-block:: python

		from sscRaft import rebinning as rb
		import numpy
		import matplotlib.pyplot as plt
		import time

		ConeData = numpy.load(<my_data.npy>)

		dic = {}  # Declare Dictionary

		dic['Distances'] = (2,1) # (z1, z2) Distances source/sample (z1) and sample/detector (z2) 
		dic['Poni'] = (0.,0) # Tuple PONI (point of incidence) of central ray at detector (cx,cy)
		dic['DetectorSize'] = (1,1) # Tuple of detector size (Dx,Dy), where the size interval is [-Dx,Dx], [-Dy,Dy]
		dic['ParDectSize'] = dic['DetectorSize'] # Tuple of parallel detector size (Lx,Ly), where the size interval is [-Lx,Lx], [-Ly,Ly]
		dic['ShiftPhantom'] = (0,0) # Tuple of phantom shift (sx,sy)
		dic['ShiftRotation'] = (0,0) # Tuple of rotation center shift (rx,ry)

		dic['Type'] = 'cpu' # String ('cpu','gpu','py') of function type - cpu, gpu, python, respectively - used to compute tomogram (3D). Defauts to 'cpu'.
		dic['gpus'] = [0] # List of GPU devices used for computation. GPU function uses only ONE GPU.

		start_ = time.time()

		RebinningData = rb.conebeam_rebinning_to_parallel(ConeData, dic)

		elapsed = time.time() - start

		print('Elapsed time for a rebinning with', dic['Type'], 'function is', elapsed, '(sec)' )

	Note that ``ConeData`` and ``RebinningData`` need an axis order :math:`angles \times slices \times rays` 