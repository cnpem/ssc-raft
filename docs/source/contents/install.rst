Installation
============

PIP
***

One can install the latest version of ``sscRaft`` directly from our ``pip server`` by:

.. code-block:: bash

    pip config --user set global.extra-index-url http://gcc.lnls.br:3128/simple/
    pip config --user set global.trusted-host gcc.lnls.br

    pip install sscRaft==version

Where ``version`` is the version number the ``sscRaft``

.. code-block:: bash

    pip install sscRaft==2.1.0


Or manually download it from the `package <http://gcc.lnls.br:3128/packages/>`_ list.

******
GITLAB
******

The package can be cloned from CNPEM's gitlab and installed locally with your user:

.. code-block:: bash

    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft
    make clean
    make    


The ``<version>`` is the version of the ``sscRaft`` to be installed. Example, to install version 2.1.0

.. code-block:: bash
    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v2.1.0 --single-branch
    cd ssc-raft
    make clean
    make    



REQUIREMENTS
************

This package uses ``C``, ``C++``, ``CUDA``, ``CUBLAS``, ``CUFFT``, ``PTHREADS`` 
and ``PYHTON 3``. For completeness, the following python packages are used within 
this package:

.. code-block:: python 

        import ctypes
        import numpy
        import sys
        import os
        import gc
        import uuid
        import SharedArray 
        import warnings
        import matplotlib
        import logger
        import time
        import h5py
        import json
        import multiprocessing


MEMORY
******

Be careful using GPU functions due to memory allocation.

UNINSTALL
*********

To uninstall ``sscRaft`` use the command

.. code-block:: bash

    pip uninstall sscRaft -y
