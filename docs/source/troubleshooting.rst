Troubleshooting
===============

Build and runtime problems
**************************

Build fails with ``cicc: not found`` (micromamba inside a Singularity/Apptainer container)
-------------------------------------------------------------------------------------------

.. note::

    This problem was observed **only** in a ``micromamba`` (conda) environment running inside a
    ``Singularity``/``Apptainer`` container (with ``--fakeroot`` write permission).
    Installations using a regular CUDA toolkit were not affected.

**Symptom.** ``pip install .`` fails during the CMake configuration, while identifying the CUDA compiler:

.. code-block:: text

    CMake Error at .../CMakeDetermineCompilerId.cmake:966 (message):
      Compiling the CUDA compiler identification source file "CMakeCUDACompilerId.cu" failed.
      Compiler: /opt/conda/envs/<env-name>/bin/nvcc
      ...
      sh: 1: /opt/conda/envs/<env-name>/bin/../targets/x86_64-linux/nvvm/bin/cicc: not found

**Cause.** CMake picked the ``nvcc`` installed in the conda/micromamba environment (for example because
``CUDACXX`` points to it). That environment only has the ``nvcc`` driver and lacks the NVVM front end
(``cicc``), which is shipped in separate conda packages. The container also provides a complete CUDA toolkit
(for example ``/usr/local/cuda-12.5``), which is already in ``PATH`` and ``LD_LIBRARY_PATH``.

**Solution.** Use the CUDA toolkit of the container instead of the one from the environment, and remove any
previous build cache:

.. code-block:: bash

    echo $CUDACXX                                  # check what is forcing the conda nvcc
    ls /usr/local/cuda-12.5/nvvm/bin/cicc          # check that the system toolkit is complete

    unset CUDACXX
    export CUDACXX=/usr/local/cuda-12.5/bin/nvcc
    export CUDA_HOME=/usr/local/cuda-12.5

    rm -rf _skbuild
    pip install . -v --no-cache-dir 2>&1 | tee build.log

Adjust the CUDA path to the version available in your container. Do not mix the toolkit of the container
with the one from the conda environment in the same build.

**Alternative.** Complete the toolkit inside the environment (the ``cufft`` and ``cublas`` libraries must
also be available in it):

.. code-block:: bash

    micromamba install -n <env-name> -c nvidia cuda-nvcc-impl cuda-nvvm-impl cuda-cudart-dev cuda-libraries-dev

Package name shown as ``UNKNOWN`` / missing Python modules (``micromamba`` environment)
------------------------------------------------------------------------------------------

**Symptom.** The build finishes, or fails, with the package named ``UNKNOWN-0.0.0`` instead of ``sscRaft``,
or ``pip`` tries to download Python modules that should already be in the environment.

**Cause.** The package name and version are defined in the ``[project]`` table of ``pyproject.toml``, which is only
read by ``setuptools>=61``. With an older ``setuptools`` (or with ``packaging`` missing/outdated) the table is
ignored and the name falls back to ``UNKNOWN``. The build and runtime modules must therefore be present, with
compatible versions, before ``pip install .`` is called.

The table below compares the modules with ``requirements.txt`` (which mirrors ``pyproject.toml``):

.. list-table::
   :header-rows: 1

   * - Module
     - In ``requirements.txt``
     - Role
   * - ``setuptools``
     - yes, ``>=61.0`` (was ``>=60.0``, which does not read ``[project]``)
     - build
   * - ``packaging``
     - added
     - build (used by ``setuptools``/``scikit-build``)
   * - ``tqdm``
     - yes
     - runtime
   * - ``cycler``
     - added
     - runtime (dependency of ``matplotlib``)
   * - ``pyparsing``
     - added
     - runtime (dependency of ``matplotlib``)
   * - ``python-dateutil``
     - added
     - runtime (dependency of ``matplotlib``)

**Solution.** Install these modules with ``micromamba``, from the ``conda-forge`` channel, and **not** with ``pip``,
so that ``conda`` keeps control of the environment:

.. code-block:: bash

    micromamba install -n <env-name> -c conda-forge --override-channels \
        "setuptools>=61" packaging cycler pyparsing python-dateutil tqdm

    micromamba activate <env-name>
    python -c "import setuptools; print(setuptools.__version__)"    # must be >= 61

    rm -rf _skbuild
    pip install . -v --no-cache-dir --no-build-isolation

``--no-build-isolation`` makes ``pip`` use the ``setuptools`` installed above instead of downloading its own copy.
In that case the other build modules (``scikit-build``, ``cmake``, ``ninja``, ``wheel``) must also be installed in the
environment, also from ``conda-forge``:

.. code-block:: bash

    micromamba install -n <env-name> -c conda-forge --override-channels \
        scikit-build "cmake>=3.24" ninja wheel

Runtime error ``the provided PTX was compiled with an unsupported toolchain``
------------------------------------------------------------------------------

**Symptom.** The library builds, but the first call to a CUDA kernel fails at run time:

.. code-block:: text

    283 in .../cuda/src/processing/excentric_tomo.cu: Cuda error: the provided PTX was compiled with an unsupported toolchain.

This is ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``. It is reported by the first ``cudaGetLastError()`` after a kernel launch,
so the line number only indicates where the error was caught, not a problem in that kernel.

**Cause.** The library was compiled only for the architecture ``sm_70`` (the default of ``CMakeLists.txt``)
and the GPU in use has a different compute capability (for example, an A100 is ``sm_80``). A GPU cannot run
SASS code of another architecture, so the driver compiles the embedded PTX at load time (JIT). If this PTX
was generated by a ``nvcc`` newer than the installed driver supports, the driver refuses it.

The maximum PTX version accepted depends on the driver, not on the installed toolkit. For example, driver
``570.x`` supports up to CUDA 12.8, so PTX generated by ``nvcc`` 12.9 is rejected. Check the supported
version in the ``CUDA Version`` field of ``nvidia-smi``.

**Solution.** Compile native code for the GPU of the machine where the code runs, so no JIT is needed.
First, get the compute capability and the driver version:

.. code-block:: bash

    nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv

Then rebuild with the architecture of the GPU (``8.0`` is ``80``, ``9.0`` is ``90``, etc.). The suffix ``-real``
generates only native code (SASS), without PTX:

.. code-block:: bash

    export CUDACXX=/usr/local/cuda-12.5/bin/nvcc     # a toolkit supported by the driver
    export CUDA_HOME=/usr/local/cuda-12.5
    export CMAKE_CUDA_ARCHITECTURES=80-real          # example: A100

    pip uninstall -y sscRaft
    rm -rf _skbuild
    pip install . -v --no-cache-dir

The default architecture (``70``) is only used when ``CMAKE_CUDA_ARCHITECTURES`` is not defined. To support
more than one GPU model, use a list, for example ``CMAKE_CUDA_ARCHITECTURES="70-real;80-real"``.

**Check.** Confirm that the installed library has code for your GPU. If ``sm_80`` (or your architecture)
does not appear, the library in use is not the one just compiled (old installation or build cache):

.. code-block:: bash

    python -c "import sscRaft, os; print(os.path.dirname(sscRaft.__file__))"
    cuobjdump --list-elf <path-printed-above>/lib/libraft.so

General tips
------------

* Always remove the ``_skbuild`` directory after changing the CUDA compiler, otherwise the old compiler path
  stays in the CMake cache.
* Use ``pip install . -v`` to see the complete CMake output; the real error is only shown there.
* If the package name appears as ``UNKNOWN`` in the build, check that the ``pip`` build environment uses
  ``setuptools>=61`` (the ``[project]`` table of ``pyproject.toml`` is ignored by older versions) and that
  the CMake step finished without errors.
