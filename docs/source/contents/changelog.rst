Changelog
=========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.


[2.1.4] - 2023-02-15
--------------------
Added
~~~~~
- New dictionary entries for ``normalization`` entry in ``FDK`` pipeline
- New dictionary entries for ``correct_projections()`` function in ``flatdark.py`` 
- New examples of usage documentation page

Corretions
~~~~~~~~~~
- Fixed a bug in linear interpolation correction in ``flatdark.cu`` - now parallelize over slices

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
