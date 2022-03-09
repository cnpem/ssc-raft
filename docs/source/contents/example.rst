Examples
========

EM/TV
*****

* Expectation Maximization with total variation using a parallel tomogram as an input: 

 .. code-block:: python

		 import numpy
		 import matplotlib.pyplot as plt
		 import time
		 
		 from sscPhantom import mario
		 from sscRaft import parallel
    
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

		 output, rad = parallel.emfs( sino, dic )

		 elapsed = time.time() - start

		 print('Elapsed time for parallel EM recon (sec):', elapsed )


CAT
***

* EM/TV from real ptychographic data restored using package ``ssc-cdi`. After a full
  ptychographic 3D reconstruction, we obtain a sequence of parallel sinograms, which
  can be considered approximate Radon transforms. A 3D inversion follows using the
  code below:

  .. code-block:: python

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

		  output, rad = parallel.emfs( sino, dic )
		  
		  elapsed = time.time() - start
		  
		  print('Elapsed time for parallel EM recon (sec):', elapsed )

  Note that ``sino`` is a transposition from ``data`` in order to use ``ssc-raft`` usual axis order
  :math:`slice \times angles \times rays` 
