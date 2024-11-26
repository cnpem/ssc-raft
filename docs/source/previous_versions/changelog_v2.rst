.. _logV2:

Version 2.3.0 - 2024-07-03
--------------------------
* Added

  - Radon Conebeam function
  - New requirements
  - CLI on pyproject

* Changed

  - Changed requirements versions

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

Version 2.2.12 - 2024-05-29
---------------------------

* Changed

  - Changed CC and VMF alignment methods to perform alignment iteratively from binned to original sinograms
  - Minor changes to IR alignemnt. Also performs alignment iteratively now, but it is not yet compatible with wrapped sinogram.

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

Version 2.2.11 - 2024-05-17
---------------------------
* Added

  - Added workaround to reduce sinogram with odd number of pixels in XY to an (even,even) shape so it works with sscRaft filters in FBP.

* Changed

  - Bug fix for Iterative reprojection (IR) alignment 

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

Version 2.2.10 - 2024-04-15
---------------------------
* Added

  - Iterative reprojection (IR) alignment module added
  - Parallel implementation for parts of cross-correlation alignment

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

Version 2.2.9 - 2024-03-26
--------------------------
* Added

  - Alignment: function for shifting frames using scipy in parallel 

* Changed

  - Divided cross-correlation alignment in two steps for ease of use with auxiliary plots

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

Version 2.2.8 - 2024-02-23
--------------------------
* Added

  - Added dictionary entry ``angles[rad]`` on ``EM``

* Changed

  - Fixed ``FBP`` slices bug!
  - Removed ``z1[m]``, ``z2[m]``, ``z1+z2[m]``, ``detectorPixel[m]`` and ``energy[eV]`` from ``FPB`` 
  - Removed ``z1[m]``, ``z2[m]``, ``z1+z2[m]``, ``detectorPixel[m]`` and ``energy[eV]`` from paganin regularization on ``FBP``
  - Changed dictionary entry  ``angles`` to ``angles[rad]`` on ``EM``
  - Changed dictionary entry  ``angles`` to ``angles[rad]`` on ``EM``
  - Changed ``radon.py`` location on folders

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs


Version 2.2.7 - 2024-02-19
--------------------------
* Added

  - Added sinogram alignment module (Cross Correlation and Vertical Mass Fluctuation, see paper 10.1364/oe.27.036637) that were previously part of ssc-cdi
  - Added Radon Ray Tracing Multi GPU functions with angles list as argument
  - Added Python EM Frequency function

* Changed

  - Fixed Dictionary entry ``TempPath`` on ``sscRaft.pipelines.mogno.mogno.Read_TomoFlatDark()`` returning bug if missing 
  - Fixed Error return bug on ``sscRaft.pipelines.mogno.mogno.Read_TomoFlatDark()`` if data cannot be found

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs
  - ``FBP`` bugs: repetition of slices, sum of different slices

Version 2.2.6 - 2024-01-23
--------------------------
* Added

  - Added ``numpy.flip()`` for ``FBP`` method return on Mogno pipeline for standardization.

* Changed

  - Fixed dictionary default logging print on ``rafttypes.py``
  - Fixed return on reconstruction methods in the case of wrong method selected for the Mogno pipeline

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs


Version 2.2.5 - 2024-01-04
--------------------------
* Added

  - Paganin in ``FBP`` CUDA function
  - New filters in ``FBP`` CUDA function: ``hamming``, ``hann`` and ``ramp``
  - CLI for Mogno pipeline: Added slices for ``FBP`` parallel reconstruction
  - Default dictionary values

* Changed

  - Fixed documentation
  - Added correct instalation instructions
  - Compilation by CMake

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs


Version 2.2.4 - 2023-12-22
--------------------------
* Added

  - New functions on Mogno pipeline in ``mogno.py``
  - New python pipeline functions as input the ndarray of data, flat and dark: ``get_reconstruction()``
  - CLI for Mogno pipeline: ``get_recon`` on ``ssc_raft_cli.py`` for data, flat and dark on different hdf5 files
  - CLI for Mogno pipeline: ``mogno_recon`` on ``ssc_raft_cli.py`` for data, flat and dark on same hdf5 files
  - Mogno pipeline now has the option to use ``FBP`` parallel reconstruction
  - Mogno pipeline now has the option to automatically find the rotation axis deviation for measures in 180 degrees

* Changed

  - Mogno pipeline functions now needs now to pass the ``dic['uselog'] = True or False`` parameter for Flat/Dark correction
  - Small changes in Mogno pipeline functions in ``mogno.py``
  - Function ``phase_filters()`` on ``phase_filters.py``: now receives [angles,slices,rays] ndarray (tomogram) as argument (previous [slices,angles,rays])
  - Function ``phase_filters()`` on ``phase_filters.py``: now returns [angles,slices,rays] ndarray (tomogram) (previous [slices,angles,rays])

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

* Removed

  - Mogno pipeline function option to use ``phase_filters()`` function on projections - Paganin is done inside ``FDK`` as in version 2.2.3


Version 2.2.3 - 2023-11-09
--------------------------
* Added

  - New dictionary entries 
  - Paganin filter on ``FDK``
  - New functions on Mogno pipeline in ``mogno.py``

* Changed

  - Dictionary entries 
  - Mogno pipeline function ``reconstruction_mogno()`` in ``mogno.py``

* Corretions

  - Memory issues on ``FDK``: illegal memmory access on backprojection

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

* Removed

  - Mogno pipeline function ``preprocessing_mogno()`` in ``mogno.py``

Version 2.2.2 - 2023-10-20
--------------------------
* Added

  - New function in Python for a reconstruction pipeline for Mogno beamline
  - New function in Python for a compute rotation axis deviation ONLY
  - New dictionary entries 

* Changed

  - Pipeline for Mogno beamline has the rotation axis correction done right before the ``FDK``

* Corretions

  - Rotation Axis function ``correct_rotation_axis360()`` in ``rotationaxis.py`` is corrected for negative deviations
  - Phase filter CUDA padding is corrected
  - ``FDK`` processes setting was increased.

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

Version 2.2.1 - 2023-09-21
--------------------------
* Added

  - Phase filters: "Paganin, Bronnikov, Rytov, Born" - all by frames
  - Padding inside ``FDK``
  - Inclusion of angles list
  - ``FDK`` Reconstruction by Slices (with bugs)
  - New dictionary entries 

* Changed

  - Padding is now done inside CUDA functions
  - Metadata datasets modifications in saving 
  - Rotation Axis function ``correct_rotation_axis360()`` in ``rotationaxis.py``: set ``padding = 0`` variable 
  - ``FDK`` receives an angles list

* Corretions

  - The ``FDK`` resconstruction multiplication factor of ``2`` related to filtering computed by Fourier Transform is corrected.

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices
  - ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs
  - Rotation Axis function with bug for negative deviations
  - Phase filter with bug on CUDA Padding

Version 2.2.0 - 2023-07-17
--------------------------
* Added

  - Function for Mogno beamline reconstruction in cone-beam geometry
  - New dictionary entries 
  - Added ``EM`` for cone-beam geometry
  - Parallel ``EM`` now accepts a list of nonregular angles as input
  - Documentation page updated! New examples of usage in documentation page

* Changed

  - Metadata datasets modifications in saving 
  - Dictionary entries for ``correct_projections()`` function in ``flatdark.py``: removed ``frames info``
  - Internal structure changed

* Corretions

  - Reconstruction parallel method ``EM`` bug with use of multiprocessing (python) together with other GPU functions.

* Known Bugs

  - Memory issues on ``EM`` for cone-beam geometry
  - The ``FDK`` resconstruction is returning a multiplication factor of ``2`` related to filtering computed by Fourier Transform. This factor changes a little when the filtering is computed by direct convolution
  - ``Tomo360`` (Mogninho - parallel-beam): Correction of bug for odd angle dimension and multiple GPUs

Version 2.1.4 - 2023-02-24
--------------------------
* Added

  - New dictionary entries for ``normalization`` entry in ``FDK`` pipeline
  - New dictionary entries for ``correct_projections()`` function in ``flatdark.py`` 
  - New examples of usage documentation page

* Changed

  - Metadata datasets modifications in saving 

* Corretions

  - Linear interpolation correction bug in ``flatdark.cu`` - now parallelize over slices
  - Reconstruction parallel method ``EM`` bug in blocksize = (1 or data.shape) and ngpus = 1

Version 2.1.3 - 2023-02-15
--------------------------
* Corretions

  - Temporary correction in a bug in frame corrections to detect outlier values in sinogram

Version 2.1.2 - 2023-02-09
--------------------------
* Corretions

  - Fixed rings bug  in ``filtering.cu`` on ``cuda.src.geometries.gc.fdk``

Version 2.1.1 - 2023-02-06
--------------------------
* Corretions

  - Fixed minor bug in ``__init__.py`` on ``cuda.src.geometries.gp.reconstruction``

Version 2.1.0 - 2023-02-02
--------------------------
* Added

  - Dictionary new entries for conical reconstruction functions
  - Cuda MultiGPU normalization function for linear interpolation between flat before and after
  - Rings by blocks added; dictionary parameter added
  - New examples of usage documentation page

* Changed

  - Dictionary entries name conical reconstruction functions
  - Python normalization function name

* Corretions

  - Fixed minor bug in normalization - now parallelize over angles

Version 2.0.1 - 2023-01-24
--------------------------
* Added

-  Automatic correction of rotation shift for conical rays

Version 2.0.0 - 2023-01-24
--------------------------
* Added

  - FDK for conical rays
  - Added rings correction to FDK source code
  - Added normalization of flat and dark to FDK
  - Added padding to FDK
  - Save metadata and version to HDF5 file

* Changed

  - Internal organization folders

Version 1.0.3 to 1.0.0 - previous releases
------------------------------------------
* Added

  - Raft for parallel rays

* Changed

  - Internal structure
