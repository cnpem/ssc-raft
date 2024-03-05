Installation
============

This package uses ``C``, ``C++``, ``CUDA`` and ``PYTHON3``. See bellow for full requirements.

The library sscRaft can be installed with either ``pip`` or ``git``.

GIT
***

One can clone our [gitlab](https://gitlab.cnpem.br/) repository and install with the following steps

.. code-block:: bash

    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft 
    make clean && make


The ``<version>`` is the version of the ``sscRaft`` to be installed. Example, to install version 2.2.8

.. code-block:: bash

    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v2.2.8 --single-branch
    cd ssc-raft 
    make clean && make


PIP
***

One can install the latest version of sscRaft directly from our ``pip server``

.. code-block:: bash

    pip install sscRaft==version --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple


Where ``version`` is the version number of the ``sscRaft``

.. code-block:: bash

    pip install sscRaft==2.2.8 --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple


MEMORY
******

Be careful using GPU functions due to memory allocation.

Requirements
************

The prerequisite for installing ``ssc-raft`` module ``sscRaft`` is ``C``, ``C++``, 
``CUDA`` and ``Python 3``.  

The following ``CUDA`` modules are used:

- ``CUBLAS``
- ``CUFFT``
- ``PTHREADS``
- ``CMAKE>=3.11``

The following ``Python3`` modules are used:

- ``skbuild>=0.17.0``
- ``setuptools>=64.0.0``
- ``numpy>=3.8.0``
- ``skimage >=0.19.3``
- ``scipy``
- ``matplotlib``
- ``logging``
- ``warning``
- ``sys``
- ``os``
- ``pathlib``
- ``inspect``
- ``SharredArray``
- ``ctypes``
- ``uuid``
- ``time``
- ``h5py``
- ``json``
- ``multiprocessing``
- ``click==8.0.4``
- ``colorama==0.4.5``
- ``rich==12.6.0``
- ``mdurl==0.1.0``
- ``Pygments==2.14.0``
- ``shellingham==1.5.0``
- ``typer==0.9.0``
- ``typing_extensions==4.1.1``

UNINSTALL
*********

To uninstall ``sscRaft`` use the command

.. code-block:: bash

    pip uninstall sscRaft -y
    