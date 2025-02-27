Install
=======

This package uses ``C``, ``C++``, ``CUDA`` and ``Python3``. 
Before installation, you will need the following packages installed:

* ``CUDA >= 10.0.0``
* ``C``
* ``C++`` 
* ``Python >= 3.8.0``
* ``PIP``
* ``libcurl4-openssl-dev``

This package supports nvidia ``GPUs`` with capabilities ``7.0`` or superior and a compiler with support to ``c++17``.

See bellow for build requirements and dependencies.

The library sscRaft can be installed from the source code at `zenodo website <https://zenodo.org/>`_, by our public github or by ``pip``/ ``gitlab``
if inside the CNPEM network. More information on the ``sscRaft`` package on 
`sscRaft website <https://gcc.lnls.br/wiki/docs/ssc-raft/>`_
available inside the CNPEM network.

Documentation
*************

The ``sscRaft`` package information can be found on the `sscRaft website <https://gcc.lnls.br/wiki/docs/ssc-raft/>`_ inside the CNEPM network.
Also, the `HTML` documentation can be found in the source directory `./docs/build/index.html` and can be opened in your preferred brownser.


GITHUB
******

The latest package version can be cloned from our public `github <https://github.com/cnpem/ssc-raft/>`_ repository and installed locally with:

.. code-block:: bash

    git clone https://github.com/cnpem/ssc-raft.git
    cd ssc-raft 
    make clean && make

To install a specific version (``<version>``), one can use:

.. code-block:: bash

    git clone  https://github.com/cnpem/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft 
    make clean && make


Example, to install version 3.1.1:

.. code-block:: bash

    git clone https://github.com/cnpem/ssc-raft.git --branch v3.1.1 --single-branch
    cd ssc-raft 
    make clean && make


Source code from Zenodo
***********************

The source code can be downloaded from `zenodo website <https://zenodo.org/>`_ under the 
DOI: `10.5281/zenodo.10988342 <https://doi.org/10.5281/zenodo.10988342>`_.

After download the ``ssc-raft-v<version>.tar.gz`` with the source files, one can decompress by

.. code-block:: bash

    tar -xvf ssc-raft-v<version>.tar.gz


To compile the source files, enter the follwing command inside the folder

.. code-block:: bash

    make clean && make


PIP
***

One can install the latest version of sscRaft directly from our ``pip server`` inside the CNPEM network.

.. warning::

    This installation option is available only inside the CNPEM network.

.. code-block:: bash

    pip install sscRaft==version --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple


Where ``version`` is the version number of the ``sscRaft``. Example:

.. code-block:: bash

    pip install sscRaft==3.0.0 --index-url https://gitlab.cnpem.br/api/v4/projects/1978/packages/pypi/simple


GITLAB
******

.. warning::

    For this installation option is available only inside the CNPEM network.

The latest package version can be cloned from CNPEM's `gitlab <https://gitlab.cnpem.br/>`_ and installed locally with:

.. code-block:: bash

    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git
    cd ssc-raft 
    make clean && make

To install a specific version (``<version>``), one can use:

.. code-block:: bash

    git clone  https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v<version> --single-branch
    cd ssc-raft 
    make clean && make


Example, to install version 3.0.3:

.. code-block:: bash

    git clone https://gitlab.cnpem.br/GCC/ssc-raft.git --branch v3.0.3 --single-branch
    cd ssc-raft 
    make clean && make


Memory
******

Be careful using GPU functions due to memory allocation.

Requirements
************

Before installation, you will need to have the following packages installed:

* ``CUDA >= 10.0.0``
* ``C``
* ``C++`` 
* ``Python >= 3.8.0``
* ``PIP``
* ``libcurl4-openssl-dev``

The build requirements are:

* ``CUBLAS``
* ``CUFFT``
* ``PTHREADS``
* ``CMAKE>=3.10``
* ``scikit-build>=0.17.0``
* ``setuptools>=64.0.0``

The ``Python3`` dependencies are:

* ``numpy``
* ``scikit-image >=0.19.3``
* ``scipy``
* ``matplotlib``
* ``SharedArray``
* ``uuid``
* ``h5py``

Uninstall
*********

To uninstall ``sscRaft`` use the command

.. code-block:: bash

    pip uninstall sscRaft
    