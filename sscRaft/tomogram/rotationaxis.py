from ..rafttypes import *
import numpy as np

def correct_rotation_axis360(data: np.ndarray, experiment: dict) -> np.ndarray:
    """CPU (python) function: Corrects the rotation axis of a sample measured on more then 180 degrees.
    Searches for the rotation axis index in axis 2 (x variable) if necessary, or corrects over a given rotation axis index value.
    Returns the projections with rotation axis corrected.
    Works with parallel, fan and cone beam sinograms for 360 degrees projections.

    Args:
        data (ndarray): Projection tomogram. The axes are [slices, angles, lenght].
        experiment (dictionary): Dictionary with the experiment info.

    Returns:
        (ndarray, int): Rotation axis corrected tomogram (3D) with axes [slices, angles, lenght] 
        and Number of pixels representing the deviation of the center of rotation. 

    Raises:
        ValueError: If the number of angles/projections is not an even number.

    Dictionary parameters:
        *``experiment['shift']`` (Tuple, optional): (bool,int) Rotation axis automatic corrrection (is_autoRot) (`is_autoRot = True`, `value = 0`).
        *``experiment['findRotationAxis']`` (Tuple, optional): (int,int,int) For rotation axis function. Tuple (`nx_search=500`, `nx_window=500`, `nsinos=None`).
        *``experiment['padding']`` (int, optional): Number of elements for horizontal zero-padding. Defaults to 0.

    Options:
        * `nx_search` (int, optional): Width of the search. 
        If the center of rotation is not in the interval `[nx_search-nx//2; nx_search+nx//2]` this function will return a wrong result.
        Default is `nx_search=500`.
        * `nx_window` (int, optional): How much of the sinogram will be used in the axis 2.
        Default is `nx_window=500`.
        * `nsinos` (int or None, optional): Number of sinograms to average over.
        Default is None, which results in `nsinos = nslices//2`, where `nslices = tomo.shape[1]`.
        * `is_autoRot` (bool,optional): Apply the automatic rotation axis correction.
        Default is `True`.
        * `value`(int,optional): Value of the rotation axis shift for correction.
        Default is `0`.

    """

    is_autoRot   = experiment['shift'][0]
    shift        = experiment['shift'][1]
    padding      = experiment['padding']

    if is_autoRot:
        logger.info('Applying automatic rotation axis correction')

        nx_search  = experiment['findRotationAxis'][0]
        nx_window  = experiment['findRotationAxis'][1]
        nsinos     = experiment['findRotationAxis'][2]
        
        shift      = find_rotation_axis_360(np.swapaxes(data,0,1), nx_search = nx_search, nx_window = nx_window, nsinos = nsinos)

    else:
        logger.info(f'Applying given rotation axis correction deviation value: {shift}')

    if padding < 2*np.abs(shift):
        padding = 2*shift

    padd = padding - 2 * np.abs(shift)

    proj = np.zeros((data.shape[0], data.shape[1], data.shape[2]+padding))

    if(shift < 0):
        proj[:,:,padd//2 + 2*np.abs(shift):data.shape[2]+padd//2 + 2*np.abs(shift)] = data
    else:
        proj[:,:,padd//2:data.shape[2]+padd//2] = data

        logger.info(f'Corrected projection for rotation axis: new shape {proj.shape}')

    # Garbage Collector
    # lists are cleared whenever a full collection or
    # collection of the highest generation (2) is run
    # collected = gc.collect() # or gc.collect(2)
    # logger.log(DEBUG_MEM,f'Garbage collector: collected {collected} objects.')

    return proj, shift

def find_rotation_axis_360(tomo, nx_search=500, nx_window=500, nsinos=None):
    """CPU (python) function: Searches for the rotation axis index in axis 2 (rays (x) variable) of a sample measured on more then 180 degrees.
    It minimizes the symmetry error.
    Works with parallel, fan and cone beam sinograms (proof: to do).
    It is assumed the center of rotation is between two pixels. 
    It might be interesting to implement an alternative where the center is ON a pixel (better suitted for odd number of angles/projections).

    Parameters
    ----------
    tomo : 3 dimensional array_like object
        Raw tomogram.
        tomo[m, i, j] selects the value of the pixel at the i-row and j-column in the projection m.
    nx_search : int, optional
        Width of the search. 
        If the center of rotation is not in the interval [nx_search-nx//2; nx_search+nx//2] this function will return a wrong result.
        Default is nx_search=400.
    nx_window : int, optional
        How much of the sinogram will be used in the axis 2.
        Default is nx_window=400.
    nsinos : int or None, optional
        Number of sinograms to avarege over.
        Default is None, which results in nsinos = nz//2, where nz = tomo.shape[1].

    Returns
    -------
    deviation : int
        Number of pixels representing the deviation of the center of rotation from the middle of the tomogram (about the axis=2).
        The integer "deviation + nx//2" is the pixel index which minimizes the symmetry error in the tomogram.
    
    Raises
    ------
    ValueError
        If the number of angles/projections is not an even number.
    """
    ntheta, nz, nx = tomo.shape

    diff_symmetry = np.ones(2*nx_search) # começando em zero para podermos fazer um gráfico de semilogy.

    if nsinos is None:
        nsinos = nz//20
    if nsinos == 0:
        nsinos = nz
    
    for k in range(nsinos):
        for i in range(2*nx_search):
            center_idx = (nx//2)-nx_search+i

            if (ntheta%2 == 0):
                sino_esq = tomo[:ntheta//2, k+(nz//2), center_idx-nx_window : center_idx]
                sino_dir = np.flip(tomo[ntheta//2:, k+(nz//2), center_idx : center_idx+nx_window], axis=1)
            else:
                sino_esq = tomo[:ntheta//2, k+(nz//2), center_idx-nx_window : center_idx]
                sino_dir = np.flip(tomo[ntheta//2:ntheta-1, k+(nz//2), center_idx : center_idx+nx_window], axis=1)
            
            mean_sinos = np.linalg.norm(sino_esq + sino_dir)/2
            diff_symmetry[i] += np.linalg.norm(sino_esq - sino_dir) / mean_sinos
    
    deviation = np.argmin(diff_symmetry) - nx_search

    # print('Shift :', deviation)
    logger.info(f'Automatic rotation axis correction deviation: {deviation}')

    # Garbage Collector
    # lists are cleared whenever a full collection or
    # collection of the highest generation (2) is run
    # collected = gc.collect() # or gc.collect(2)
    # logger.log(DEBUG,f'Garbage collector: collected {collected} objects.')

    return deviation