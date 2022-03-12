Install
========

GITLAB
******

The package can be cloned from CNPEM's gitlab and hence install locally with your user:

.. code-block:: bash

   git clone https://gitlab.cnpem.br/GCC/ssc-raft.git

   python3 setup.py install --user --cuda


.. note:: Flag --cuda is mandatory for users with a Nvidia/GPU.

REQUIREMENTS
************

For completeness, the following packages are used within this package:

.. code-block:: python 

        import ctypes
        from ctypes import c_float as float32
        from ctypes import c_int as int32
        from ctypes import c_int as int32
        from ctypes import POINTER
        from ctypes import c_void_p  as void_p
        from ctypes import c_size_t as size_t

        import numpy
        import sys
        import gc
        import uuid
        import SharedArray as sa

        import warnings

MEMORY
******

Be careful using GPU functions due to memory allocation.
