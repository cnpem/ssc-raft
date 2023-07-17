Examples
========

We refer to the Scientific Computing Group (GCC) page `https://gcc.lnls.br/ <https://gcc.lnls.br/>`_ for a more complete sscRaft usage examples and tutorials. 
Link - `https://gcc.lnls.br/wiki/docs/ssc-raft/ <https://gcc.lnls.br/wiki/docs/ssc-raft/>`_.


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