from typing import Optional
from ...rafttypes import *
import numpy
import ctypes

def Centersino(frame0, frame1, flat, dark):
        """ Find the offset of a 180 tomogram to correctly align it, computed by cross correlation.
        It does the application of Flatfield :math:`I_{0}` and darkfield :math:`d_{0}` 
        on the intensity data :math:`I` inside function

        .. math:: 
            -\log( \\frac{I - d_{0}}{I_{0}} ) 

        Args:
            frame0 (ndarray): First frame of intensity data obtained by detector
            frame1 (ndarray): Last frame of intensity data obtained by detector
            flat (ndarray): Flat
            dark (ndarray): Dark

        Returns:
            (int): Offset
        """        

        nrays = frame0.shape[-1]
        nslices = frame0.shape[-2]

        frame0 = numpy.ascontiguousarray(frame0.astype(numpy.float32))
        frame0ptr = frame0.ctypes.data_as(ctypes.c_void_p)

        frame1 = numpy.ascontiguousarray(frame1.astype(numpy.float32))
        frame1ptr = frame1.ctypes.data_as(ctypes.c_void_p)

        dark = numpy.ascontiguousarray(dark.astype(numpy.float32))
        darkptr = dark.ctypes.data_as(ctypes.c_void_p)

        flat = numpy.ascontiguousarray(flat.astype(numpy.float32))
        flatptr = flat.ctypes.data_as(ctypes.c_void_p)

        offset = libraft.findcentersino(frame0ptr, frame1ptr, darkptr, flatptr, 
                                        ctypes.c_int(nrays), ctypes.c_int(nslices))
        
        return offset

def Centersino_subpixel(frame0, frame1, flat, dark):
        """ Find the subpixel offset of a 180 tomogram to correctly align it, computed by cross correlation.
        It does the application of Flatfield :math:`I_{0}` and darkfield :math:`d_{0}` 
        on the intensity data :math:`I` inside function

        .. math:: 
            -\log( \\frac{I - d_{0}}{I_{0}} ) 

        Args:
            frame0 (ndarray): First frame of intensity data obtained by detector
            frame1 (ndarray): Last frame of intensity data obtained by detector
            flat (ndarray): Flat
            dark (ndarray): Dark

        Returns:
            (float): Subpixel offset
        """        

        nrays = frame0.shape[-1]
        nslices = frame0.shape[-2]

        frame0 = numpy.ascontiguousarray(frame0.astype(numpy.float32))
        frame0ptr = frame0.ctypes.data_as(ctypes.c_void_p)

        frame1 = numpy.ascontiguousarray(frame1.astype(numpy.float32))
        frame1ptr = frame1.ctypes.data_as(ctypes.c_void_p)

        dark = numpy.ascontiguousarray(dark.astype(numpy.float32))
        darkptr = dark.ctypes.data_as(ctypes.c_void_p)

        flat = numpy.ascontiguousarray(flat.astype(numpy.float32))
        flatptr = flat.ctypes.data_as(ctypes.c_void_p)

        offset = libraft.findcentersino_subpixel(frame0ptr, frame1ptr, darkptr, flatptr, 
                                        ctypes.c_int(nrays), ctypes.c_int(nslices))
        
        return offset

def correct_rotation_axis360(data: numpy.ndarray, dic: dict) -> numpy.ndarray:
    """CPU (python) function: Corrects the rotation axis of a sample measured on more then 180 degrees.
    Searches for the rotation axis index in axis 2 (x variable) if necessary, or corrects over a given rotation axis index value.
    Returns the projections with rotation axis corrected.
    Works with parallel, fan and cone beam sinograms for 360 degrees projections.

    Args:
        data (ndarray): Projection tomogram. The axes are [slices, angles, lenght]
        dic (dictionary): Dictionary with the parameters info

    Returns:
        (ndarray, int): Rotation axis corrected tomogram (3D) with axes [slices, angles, lenght] 
        and Number of pixels representing the deviation of the center of rotation

    Raises:
        ValueError: If the number of angles/projections is not an even number.

    Dictionary parameters:

        * ``dic['shift']`` (Tuple, optional): (bool,int) Rotation axis automatic corrrection (is_autoRot) (``is_autoRot = True``, ``value = 0``)
        * ``dic['findRotationAxis']`` (Tuple, optional): (int,int,int) For rotation axis function. Tuple (``nx_search=500``, ``nx_window=500``, ``nsinos=None``)
        * ``dic['padding']`` (int, optional): Number of elements for horizontal zero-padding. Defaults to ``0``

    Options:

        * ``nx_search`` (int, optional): Width of the search. If the center of rotation is not in the interval ``[nx_search-nx//2; nx_search+nx//2]`` this function will return a wrong result. Default is ``nx_search=500``.
        * ``nx_window`` (int, optional): How much of the sinogram will be used in the axis 2. Default is ``nx_window=500``.
        * ``nsinos`` (int or None, optional): Number of sinograms to average over.Default is None, which results in ``nsinos = nslices//2``, where ``nslices = tomo.shape[1]``.
        * ``is_autoRot`` (bool,optional): Apply the automatic rotation axis correction. Default is ``True``.
        * ``value`` (int,optional): Value of the rotation axis shift for correction. Default is ``0``.

    """

    is_autoRot   = dic['shift'][0]
    shift        = dic['shift'][1]

    if is_autoRot:
        logger.info('Applying automatic rotation axis correction')

        nx_search  = dic['findRotationAxis'][0]
        nx_window  = dic['findRotationAxis'][1]
        nsinos     = dic['findRotationAxis'][2]
        
        shift      = find_rotation_axis_360(numpy.swapaxes(data,0,1), nx_search = nx_search, nx_window = nx_window, nsinos = nsinos)

        dic.update({'shift':[is_autoRot,shift]})

    else:
        logger.info(f'Applying given rotation axis correction deviation value: {shift}')

    proj = numpy.zeros((data.shape[0], data.shape[1], data.shape[2] + 2 * numpy.abs(shift)))

    if(shift < 0):
        proj[:,:,2 * numpy.abs(shift):data.shape[2] + 2 * numpy.abs(shift)] = data
    else:
        proj[:,:,0:data.shape[2]] = data

    logger.info(f'Corrected projection for rotation axis: new shape {proj.shape}')

    return proj, shift

def find_rotation_axis_360(tomo, nx_search=500, nx_window=500, nsinos=None):
    """CPU (python) function: Searches for the rotation axis index in axis 2 (rays (x) variable) of a sample measured on more than 180 degrees.
    It minimizes the symmetry error.
    Works with parallel, fan and cone beam sinograms (proof: to do).
    It is assumed the center of rotation is between two pixels. 
    It might be interesting to implement an alternative where the center is ON a pixel (better suitted for odd number of angles/projections).

    Parameters
    ----------
    tomo : 3 dimensional array_like object
        Raw tomogram in [angles,slice,lenght] axis.
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

    diff_symmetry = numpy.ones(2*nx_search) # começando em zero para podermos fazer um gráfico de semilogy.

    if nsinos is None:
        nsinos = nz//20
    if nsinos == 0:
        nsinos = nz
    
    for k in range(nsinos):
        for i in range(2*nx_search):
            center_idx = (nx//2)-nx_search+i

            if (ntheta%2 == 0):
                sino_esq = tomo[:ntheta//2, k+(nz//2), center_idx-nx_window : center_idx]
                sino_dir = numpy.flip(tomo[ntheta//2:, k+(nz//2), center_idx : center_idx+nx_window], axis=1)
            else:
                sino_esq = tomo[:ntheta//2, k+(nz//2), center_idx-nx_window : center_idx]
                sino_dir = numpy.flip(tomo[ntheta//2:ntheta-1, k+(nz//2), center_idx : center_idx+nx_window], axis=1)
            
            mean_sinos = numpy.linalg.norm(sino_esq + sino_dir)/2
            diff_symmetry[i] += numpy.linalg.norm(sino_esq - sino_dir) / mean_sinos
    
    deviation = numpy.argmin(diff_symmetry) - nx_search

    logger.info(f'Automatic rotation axis correction deviation: {deviation}')

    return deviation

def c_correct_rotation_axis(data: numpy.ndarray, deviation: int,
                            out: Optional[numpy.ndarray] = None) -> numpy.ndarray:

    if out is None:
        out = np.empty_like(data)
    else:
        if not out.flags.c_contiguous:
            raise ValueError('Output array must be contiguous')

    out = CNICE(out)
    out_ptr = out.ctypes.data_as(c_void_p)
    data = CNICE(data)
    data_ptr = data.ctypes.data_as(c_void_p)

    sizez, sizey, sizex = [ctypes.c_int(d) for d in data.shape]

    libraft.correctRotationAxis(data_ptr, out_ptr,
                                sizex, sizey, sizez,
                                ctypes.c_int(deviation))


    return out


def correct_rotation_axis(data: numpy.ndarray, deviation: int) -> numpy.ndarray:
    """Corrects the rotation axis of a data according to a deviation value defined 
    by the number of pixels translated form the center of the data.

    The deviation value, with sign, is computed through the sscRaft function ``Centersino()``.

    Args:
        data (ndarray): Projection tomogram. The axes are [slices, angles, lenght]
        deviation (int): Number of pixels representing the rotation axis deviation

    Returns:
        (ndarray): Rotation axis corrected tomogram (3D) with shape [slices, angles, 2 * deviation + lenght] 

    * CPU function
    """
    logger.info(f'Applying given rotation axis correction deviation value: {deviation}')

    deviation = - deviation # Fix centersino value

    proj = numpy.zeros((data.shape[0], data.shape[1], data.shape[2] + 2 * numpy.abs(deviation)))

    if(deviation < 0):
        proj[:,:,2 * numpy.abs(deviation):data.shape[2] + 2 * numpy.abs(deviation)] = data
    else:
        proj[:,:,0:data.shape[2]] = data

    logger.info(f'Corrected projection for rotation axis: new shape {proj.shape}')

    return proj

def correct_rotation_axis_cropped(data: numpy.ndarray, deviation: int) -> numpy.ndarray:
    """Corrects the rotation axis of a data according to a deviation value defined 
    by the number of pixels translated form the center of the data.

    The deviation value, with sign, is computed through the sscRaft function ``Centersino()``.

    Args:
        data (ndarray): Projection tomogram. The axes are [slices, angles, lenght]
        deviation (int): Number of pixels representing the rotation axis deviation

    Returns:
        (ndarray): Rotation axis corrected tomogram (3D) with shape [slices, angles, lenght] 

    * CPU function
    """
    logger.info(f'Applying given rotation axis correction deviation value: {deviation}')

    deviation = - deviation # Fix centersino value

    proj = numpy.zeros((data.shape[0], data.shape[1], data.shape[2] + 2 * numpy.abs(deviation)))

    if(deviation < 0):
        proj[:,:,2 * numpy.abs(deviation):data.shape[2] + 2 * numpy.abs(deviation)] = data
    else:
        proj[:,:,0:data.shape[2]] = data


    if deviation != 0:
        proj = proj[:,:,numpy.abs(deviation):-numpy.abs(deviation)]

    return proj


def correct_rotation_axis_subpixel(data: numpy.ndarray, axis_offset: float, gpus: list = [0], blocksize: int = 0) -> numpy.ndarray:
    """Corrects the rotation axis of a data according to a deviation value from
    the center of the data. Subpixel precision.

    The deviation value, with sign, is computed through the sscRaft function ``Centersino()``.

    Args:
        data (ndarray): Projection tomogram. The axes are [slices, angles, lenght]
        axis_offset (float): Distance in pixels representing the rotation axis deviation

    Returns:
        (ndarray): Rotation axis corrected tomogram (3D) with shape [slices, angles, lenght] 

    * CPU function
    """
    logger.info(f'Applying subpixel rotation axis correction with deviation value: {axis_offset}')

    ngpus    = len(gpus)
    gpus     = numpy.array(gpus)
    gpus     = numpy.ascontiguousarray(gpus.astype(numpy.intc))
    gpus_ptr = gpus.ctypes.data_as(ctypes.c_void_p)

    nrays    = data.shape[-1]
    nangles  = data.shape[-2]
    
    if len(data.shape) == 3:
        nslices = data.shape[ 0]
    if len(data.shape) == 2:
        nslices = 1

    data     = CNICE(data)
    data_ptr = data.ctypes.data_as(ctypes.c_void_p)

    libraft.getRotAxisCorrectionMultiGPU(gpus_ptr, ctypes.c_int(ngpus),
                                         data_ptr, c_float(axis_offset),
                                         c_int(nrays), c_int(nangles), c_int(nslices),
                                         c_int(blocksize))

    return data


def Centersino_block(frame0, frame1, flat, dark, block_of_slices):
    """ Find the offset by block of slices of a 180 tomogram 
    to correctly align it, computed by cross correlation.
    It does the application of Flatfield :math:`I_{0}` and darkfield :math:`d_{0}` 
    on the intensity data :math:`I` inside function

    .. math:: 
        -\log( \\frac{I - d_{0}}{I_{0}} ) 

    Args:
        frame0 (ndarray): First frame of intensity data obtained by detector. The axes are [slices, lenght]
        frame1 (ndarray): Last frame of intensity data obtained by detector - The axes are [slices, lenght]
        flat (ndarray): Flat. The axes are [slices, lenght]
        dark (ndarray): Dark. The axes are [slices, lenght]
        block_of_slices (int): Number of slices in a block

    Returns:
        (int list): List of offset values
    """   
    
    nslices =  frame0.shape[0]

    rotation_axis = []

    for i in range(0,nslices,block_of_slices):
    
        deviation = Centersino(frame0[i:i+block_of_slices], frame1[i:i+block_of_slices], flat[i:i+block_of_slices], dark[i:i+block_of_slices])
    
        rotation_axis.append(deviation)
        
    return rotation_axis


def correct_rotation_axis_block(data, rotation_axis_list, block_of_slices):
    """Corrects the rotation axis by blocks of a data according to a deviation value defined 
    by the number of pixels translated form the center of the data.

    The rotation_axis_list, with signs, is computed through the sscRaft function ``Centersino_block()``.

    Args:
        data (ndarray): Projection tomogram. The axes are [slices, angles, lenght]
        rotation_axis_list (int list): List of rotation axis deviation by blocks
        block_of_slices (int): Number of slices in a block

    Returns:
        (ndarray): Rotation axis corrected tomogram (3D) with shape [slices, angles, lenght] 

    * CPU function
    """
    nslices = data.shape[0]

    j = 0
    for i in range(0,nslices,block_of_slices):

        data[i:i+block_of_slices] = correct_rotation_axis_cropped(data[i:i+block_of_slices], rotation_axis_list[j])
        j = j + 1

    return data