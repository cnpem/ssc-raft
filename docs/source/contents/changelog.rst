Changelog
=========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_ and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.


[2.1.0] - 2023-02-01
--------------------
Added
~~~~~
- Dictionary new entries
- Cuda normalization function for linear interpolation between flat before and after
- Rings by blocks added

Changed
~~~~~~~
- Dictionary entries name changed
- Python normalization function name

Corretions
~~~~~~~~~~
- Fixed minor bug in normalization


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