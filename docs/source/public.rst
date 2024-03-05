Public
======

***************
Getting started
***************

This page documents the ssc-raft module from Sirius Scientific Computing Group. 

***************
Code structure
***************

This module is implemented in ``CUDA`` with an interface in ``Python3``.

Function call
***************

The ssc-raft algorithm is called as ``sscRaft`` module in ``Python3`` 

.. code-block:: python
    
   import sscRaft

All the functions in sscRaft are called by passing numpy array arguments and a dictionary. 

.. code-block:: python

    '''set dictionary and arrays:

    dic = {}
    ...

    input1 = ...
    input2 = ...
    ...
    '''

    output = sscRaft.function_name(input1, input2, dic)

Example of usage of the filtered backprojection (by ray-tracing) algorithm (FBP):

.. code-block:: python

    import sscRaft
    import numpy

    out_path = 'path_to_output_HDF5_file'
    out_name = 'name_output_HDF5_file'

    '''Load or compute tomogram
    tomogram = ...

    Set Dictionary
    '''
    dic = {}
    dic['gpu']                    = [0,1]
    dic['filter']                 = 'lorentz'
    dic['paganin regularization'] = 1e-3
    dic['padding']                = 2
    dic['angles[rad]']            = numpy.linspace(0, np.pi, tomogram.shape[1])

    recon = sscRaft.fbp(tomogram, dic)

.. - :ref:`io`
.. - :ref:`correction`
.. - :ref:`alignment`
.. - :ref:`phasefilter`
.. - :ref:`rings`
.. - :ref:`reconstruction`
.. - :ref:`forward`
.. - :ref:`pipelines`

sscRaft 
*******

Documentation for the sscRaft ``PYTHON`` functions.

.. toctree::

    io
    correction
    alignment
    rings
    reconstruction
    forward
    pipelines




