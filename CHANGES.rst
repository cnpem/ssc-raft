Version 3.2.3 - 2025-04-28
--------------------------
*Changed:*
  - Fixed blocksize division for ``correct_background()`` function.
  - Fixed blocksize division for ``phase_retrieval()`` function (Paganin by frames).
  - Fixed blocksize division for ``rings()`` and ``correct_rotation_axis`` functions. 
  - Fixed ``L`` variable on ``FDK`` process division.

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Memory issues on ``EM`` for cone-beam geometry.
  - No cuda streams on ``FBP by BST``: bug.
  - A few bugs on TOMCAT's CLI rings and paganin methods call.
  - Return of ``FDK`` memory bug on backprojection function for some dimensions. Probably never fully fixed!

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.

Version 3.2.2 - 2025-04-14
--------------------------
*Changed:*
  - Return blocksize division function for power of 2. 
  - Fixed ``FDK`` process division function for slices. 

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Memory issues on ``EM`` for cone-beam geometry.
  - No cuda streams on ``FBP by BST``: bug.
  - A few bugs on TOMCAT's CLI rings and paganin methods call.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.


Version 3.2.1 - 2025-04-11
--------------------------
*Added:*
  - New blocksize division that is not power of 2. 

*Changed:*
  - Fixed rotation axis blocks: block size computations.
  - Fixed ``FDK`` process division function for slices. 

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Memory issues on ``EM`` for cone-beam geometry.
  - No cuda streams on ``FBP by BST``: bug.
  - A few bugs on TOMCAT's CLI rings and paganin methods call.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.

Version 3.2.0 - 2025-04-01
--------------------------
*Added:*
  - Added extendend domain padding for functions ``FDK``, ``FBP by RT`` and ``FBP by BST``. 
  - Added rotation axis offset function by blocks.
  - Added rotation axis correction function by blocks with subpixel correction.
  - No cuda streams on ``FBP by BST``.
  - Blocksize option added on ``FDK`` function.
  - Added rotational axis and excentric tomography methods documentation pages.
  - Added version on zenodo ``json``.

*Changed:*
  - Changed use of padding for functions ``FDK``, ``FBP by RT`` and ``FBP by BST``. Now it is on the whole data process, not only on the filter.
  - Fixed TOMCAT API pipeline for live-reconstruction.
  - Fixed memory allocation on ``FBP by BST`` and ``background_correction``.
  - Changed names for excentric tomography functions.
  - Fixed excentric tomography stitching correction function padding.
  - Fixed ``stitchExcentricTomo`` (parallel-beam) bug for odd angle dimension.
  - Some minor bugs on the ``iterative_reprojetion()`` alignment function.

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Memory issues on ``EM`` for cone-beam geometry.
  - No cuda streams on ``FBP by BST``: bug.
  - A few bugs on TOMCAT's CLI rings and paganin methods call.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.


Version 3.1.1 - 2025-02-26
--------------------------
*Added:*
  - Added GPU transpose function ``transpose_gpu()``.

*Changed:*
  - Fixed Memory issues on ``FDK``: limitation for number of processes as it is hard-coded.
  - Fixed Memory issues on ``FDK``: In reconstruction by slices. Supports only blocks divisible by 8.
  - Fixed ``iterative_reprojection()`` alignment function with ciclic imports.

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Memory issues on ``EM`` for cone-beam geometry.
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.

Version 3.1.0 - 2025-02-20
--------------------------
*Added:*
  - Added pixel size as optional input argument on parallel Radon function ``Radon_RT``.
  - Added an optimized version of conebeam Radon function.
  - Added rotation axis correction and Centersino by blocks function.
  - Added CLI for parallel pipeline with excentric tomography for SLS/TOMCAT data. 

*Changed:*
  - Fixed ``FBP`` units issues. Previously, the function returned a nondimensionalized reconstruction. This change adds dimension (in SI units) to the reconstruction [``1/m``].
  - Fixed ``FBP`` padding bug. 
  - Changed ``FBP by RT`` values to be compatible with ``FDK`` for all angles.
  - Changed ``FBP by BST`` values to be compatible with ``FDK`` for 180 degree angles.
  - Changed ``FBP`` filter R2C/C2R to add padding. 
  - Fixed parallel radon funtion ``Radon_RT`` units issues. Previously, the function returned a nondimensionalized projection. This change adds dimension (in SI units) to the projection [``1/m``] if the user use as input the pixel size.

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - Memory issues on ``EM`` for cone-beam geometry.
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded
  - Memory issues on ``FDK``: In reconstruction by slices. Supports only blocks divisible by 8.
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.


Version 3.0.3 - 2024-12-11
--------------------------
*Changed:*
  - Fixed ``FDK`` blocksize bug for some dimension sizes.
  - Fixed iterative alignment bug where the ``FBP`` method call not updated from version 2.Y.Z.

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Padding not working very well on ``FBP`` - turned-off.
  - Memory issues on ``EM`` for cone-beam geometry.
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded.
  - Memory issues on ``FDK``: In reconstruction by slices. Supports only blocks divisible by 8.
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs.
  - Problem on Paganin by slices version in the methods ``FBP by RT`` and ``FBP by BST`` where the beta/delta parameter have a difference of 1e-11 of the same parameter for ``FDK`` method.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.


Version 3.0.2 - 2024-10-24
--------------------------
*Added:*
  - Fast transpose zyx2xyz on large data.

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Padding not working very well on ``FBP`` - turned-off.
  - Memory issues on ``EM`` for cone-beam geometry.
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded.
  - Memory issues on ``FDK``: In reconstruction by slices.
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs.
  - Problem on Paganin by slices version in the methods ``FBP by RT`` and ``FBP by BST`` where the beta/delta parameter have a difference of 1e-11 of the same parameter for ``FDK`` method.
  - Iterative alignment bug: ``FBP`` method call was not updated.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.


Version 3.0.1 - 2024-10-02
--------------------------
*Added:*
  - ``correct_rotation_axis_cropped()`` function tha crops the extra padding added for rotation axis offset correction.
  
*Changed:*
  - Corrected bug on FFTShift for ``phase_retrieval()`` function (classic Paganin method).
  - Corrected bug on ``cufftPlanMany`` for ``phase_retrieval()`` function (classic Paganin method).
  - Inclusion of magnitude on on Paganin by slices version in ``FDK``.
 
*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Padding not working very well on ``FBP`` - turned-off.
  - Memory issues on ``EM`` for cone-beam geometry.
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded.
  - Memory issues on ``FDK``: In reconstruction by slices.
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs.
  - Problem on Paganin by slices version in the methods ``FBP by RT`` and ``FBP by BST`` where the beta/delta parameter have a difference of 1e-11 of the same parameter for ``FDK`` method.
  - Iterative alignment bug: ``FBP`` method call was not updated.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.

Version 3.0.0 - 2024-09-10
--------------------------
*Added:*
  - ``EM`` on Frequency domain for parallel-beam.
  - Initial guess in ``tEMRT`` and  ``eEMRT`` for parallel-beam.
  - ``BST`` reconstruction with new filters and paganin filter.
  - Radon ray tracing for parallel beam.
  - Wiggle and other methods of alignment.
  - C/C++/CUDA pipeline.
  - ``io.py`` file for io related functions.
  - ``correct_background()`` function that corrects the background (flat/dark) with data axis as ``[slices,angles,lenght]`` as input.
  - ``correct_rotation_axis()`` function to correct axis deviation.
  - Stitching 360 to 180 degrees tomography functions for parallel beam.
  - ``phase_retrieval()`` function added with Paganin method by frames.
  - Pinned memory functions for usage.
  - ``CUDA STREAMS`` added in background correction, rings and ``FBP`` by ``BST`` functions.
  - ``transpose()`` C/C++/CUDA function to change from projection space to sinogram space.
  - ``flip_x()`` C/C++/CUDA function to flip (reflect) x-axis.
 
*Changed:*
  - Source code re-factored.
  - Dictionary entries.
  - Changed dictionary all function entries from ``angles`` to ``angles[rad]`` on ``EM``.
  - ``em()`` function to support all ``EM`` related methods for parallel beam, as of now.
  - ``fbp()`` function to support all ``FBP`` related methods for parallel beam, like BST, as of now.
  - Rings and flat/dark correction functions dictionary.
  - Paganin regularization dictionary entry for slices version from ``paganin regularization`` to ``beta/delta`` and standardization for all Paganin related methods.
  - Reconstruction methods have now the possibility to receive the reconstruction volume as input.

*Known Bugs:*
  - ``BST`` works for 180 degrees only on a regular angle mesh.
  - ``BST`` angles are hardcoded and not as input.
  - Padding not working very well on ``FBP`` - turned-off.
  - Memory issues on ``EM`` for cone-beam geometry.
  - Memory issues on ``FDK``: limitation for number of processes as it is hard-coded.
  - Memory issues on ``FDK``: In reconstruction by slices.
  - Paganin slice version not working on ``FBP by RT`` and ``FBP by BST`` methods.
  - ``Tomo360`` (parallel-beam): Correction of bug for odd angle dimension and multiple GPUs.
  - Iterative alignment bug: ``FBP`` method call was not updated.

*Removed:*
  - ``em_cone()`` function.
  - ``bst()`` function.
  - ``phase_filter()`` functions.
  - CLI as of now.

*To be done:*
  - Refactoring ``FDK``.
  - Refactoring ``EM`` conebeam ray tracing.
  - Refactoring ``FST`` frequency domain forward method for parallel beam.
  - Refactoring ``RadonCONE`` ray tracing forward method for conebeam.
