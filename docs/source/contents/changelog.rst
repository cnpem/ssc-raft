Changelog
=========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[2.2.2] - 2023-10-20
--------------------
Added
~~~~~
- New function in Python for a reconstruction pipeline for Mogno beamline
- New function in Python for a compute rotation axis deviation ONLY
- New dictionary entries 

Changed
~~~~~~~
- Pipeline for Mogno beamline has the rotation axis correction done right before the ``FDK``

Corretions
~~~~~~~~~~
- Rotation Axis function ``correct_rotation_axis360()`` in ``rotationaxis.py`` is corrected for negative deviations
- Phase filter CUDA padding is corrected
- ``FDK`` processes setting was increased.

Known Bugs
~~~~~~~~~~
- Memory issues on ``EM`` for cone-beam geometry
- Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
- Memory issues on ``FDK``: In reconstruction by slices
- ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

[2.2.1] - 2023-09-21
--------------------
Added
~~~~~
- Phase filters: "Paganin, Bronnikov, Rytov, Born" - all by frames
- Padding inside ``FDK``
- Inclusion of angles list
- ``FDK`` Reconstruction by Slices (with bugs)
- New dictionary entries 

Changed
~~~~~~~
- Padding is now done inside CUDA functions
- Metadata datasets modifications in saving 
- Rotation Axis function ``correct_rotation_axis360()`` in ``rotationaxis.py``: set ``padding = 0`` variable 
- ``FDK`` receives an angles list

Corretions
~~~~~~~~~~
- The ``FDK`` resconstruction multiplication factor of ``2`` related to filtering computed by Fourier Transform is corrected.

Known Bugs
~~~~~~~~~~
- Memory issues on ``EM`` for cone-beam geometry
- Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
- Memory issues on ``FDK``: In reconstruction by slices
- ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs
- Rotation Axis function with bug for negative deviations
- Phase filter with bug on CUDA Padding

[2.2.0] - 2023-07-17
--------------------
Added
~~~~~
- Function for Mogno beamline reconstruction in cone-beam geometry
- New dictionary entries 
- Added ``EM`` for cone-beam geometry
- Parallel ``EM`` now accepts a list of nonregular angles as input
- Documentation page updated! New examples of usage in documentation page

Changed
~~~~~~~
- Metadata datasets modifications in saving 
- Dictionary entries for ``correct_projections()`` function in ``flatdark.py``: removed ``frames info``
- Internal structure changed

Corretions
~~~~~~~~~~
- Reconstruction parallel method ``EM`` bug with use of multiprocessing (python) together with other GPU functions.

Bugs
~~~~~~~~~~
- Memory issues on ``EM`` for cone-beam geometry
- The ``FDK`` resconstruction is returning a multiplication factor of ``2`` related to filtering computed by Fourier Transform. This factor changes a little when the filtering is computed by direct convolution
- ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

[2.1.4] - 2023-02-24
--------------------
Added
~~~~~
- New dictionary entries for ``normalization`` entry in ``FDK`` pipeline
- New dictionary entries for ``correct_projections()`` function in ``flatdark.py`` 
- New examples of usage documentation page

Changed
~~~~~~~
- Metadata datasets modifications in saving 

Corretions
~~~~~~~~~~
- Linear interpolation correction bug in ``flatdark.cu`` - now parallelize over slices
- Reconstruction parallel method ``EM`` bug in blocksize = (1 or data.shape) and ngpus = 1


[2.1.3] - 2023-02-15
--------------------
Corretions
~~~~~~~~~~
- Temporary correction in a bug in frame corrections to detect outlier values in sinogram

[2.1.2] - 2023-02-09
--------------------
Corretions
~~~~~~~~~~
- Fixed rings bug  in ``filtering.cu`` on ``cuda.src.geometries.gc.fdk``

[2.1.1] - 2023-02-06
--------------------
Corretions
~~~~~~~~~~
- Fixed minor bug in ``__init__.py`` on ``cuda.src.geometries.gp.reconstruction``

[2.1.0] - 2023-02-02
--------------------
Added
~~~~~
- Dictionary new entries for conical reconstruction functions
- Cuda MultiGPU normalization function for linear interpolation between flat before and after
- Rings by blocks added; dictionary parameter added
- New examples of usage documentation page

Changed
~~~~~~~
- Dictionary entries name conical reconstruction functions
- Python normalization function name

Corretions
~~~~~~~~~~
- Fixed minor bug in normalization - now parallelize over angles

[2.0.1] - 2023-01-24
--------------------
Added
~~~~~
- Automatic correction of rotation shift for conical rays

[2.0.0] - 2023-01-24
--------------------
Added
~~~~~
- FDK for conical rays
- Added rings correction to FDK source code
- Added normalization of flat and dark to FDK
- Added padding to FDK
- Save metadata and version to HDF5 file

Changed
~~~~~~~
- Internal organization folders

[1.0.3] to [1.0.0] - previous releases
--------------------------------------
Added
~~~~~
- Raft for parallel rays 

Changed
~~~~~
- Internal structure
