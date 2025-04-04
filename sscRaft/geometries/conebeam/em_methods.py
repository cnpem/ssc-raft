from ...rafttypes import *
import numpy
import pathlib

def valid_data(cone_data: dict):
    ok = True
    if "size" in cone_data:
        size = cone_data["size"]
        if isinstance(size, int):
            if size > 0:
                size = numpy.array([size, size, size])
            else:
                ok = False
                warnings.warn(
                "Invalid reconstruction size encountered: it must be a positive integer.")
        elif len(size) != 3:
            ok = False
            warnings.warn(
                "Invalid reconstruction size encountered: it must be an int or a list of ints of length 3.")
        else:
            for n in size:
                if n < 1:
                    ok = False
                    warnings.warn(
                        "Invalid reconstruction size encountered: it must be a positive integer.")
    if "Energy[keV]" not in cone_data:
        ok = False
        warnings.warn("X-ray photon energy is missing from input data.")
    return ok

def _process_cone_data(tomo, flat, dark):
    """Rever o 'tomo - dark' (quando a eq. 57 do Fessler converge para a simples subtração do dark?).
    """
    if tomo.ndim != 3:
        raise ValueError("'tomo' must be a 3-dimensional array_like object. It has {} dimensions".format(tomo.ndim))
    if flat.ndim != 3:
        raise ValueError("'flat' must be a 3-dimensional array_like object. It has {} dimensions".format(flat.ndim))
    if dark.ndim != 3:
        raise ValueError("'dark' must be a 3-dimensional array_like object. It has {} dimensions".format(dark.ndim))

    flat = flat.mean(axis=0)
    dark = dark.mean(axis=0)
    flat -= dark # rever esse passo. o ideal é enviar o dark para a gpu e somar lá dentro, ao invés de subtrair aqui.
    tomo = tomo.copy() - dark

    if (tomo < 0).any():
        raise ValueError("Negative value found in tomo after dark subtraction.")
    if (flat < 0).any():
        raise ValueError("Negative value found in flat after dark subtraction.")

    if tomo.dtype != numpy.float32:
        tomo = tomo.astype(numpy.float32)
    if flat.dtype != numpy.float32:
        flat = flat.astype(numpy.float32)
    if dark.dtype != numpy.float32:
        dark = dark.astype(numpy.float32)

    return tomo, flat, dark

def _make_angs_pointer(cone_data, tomo_shape):
    try:
        angs = CNICE(cone_data["angs"], numpy.float32)
    except:
        angs = numpy.linspace(0, 2*numpy.pi, tomo_shape[0], dtype=numpy.float32)
    angs_p = angs.ctypes.data_as(ctypes.c_void_p)
    return angs_p

def _make_detector_arrays(z2, nh, nv, H, V):
    dh = 2*H/nh
    dv = 2*V/nv
    h = numpy.linspace(-H+dh/2, H-dh/2, nh)
    v = numpy.linspace(-V+dv/2, V-dv/2, nv)
    hh, vv = numpy.meshgrid(h, v)
    hh = hh.flatten()
    vv = vv.flatten()
    yy = numpy.linspace(z2, z2, nh*nv)
    return hh, yy, vv

def _make_detector_pointer(cone_data, tomo_shape):
    try:
        px = CNICE(cone_data["px"], numpy.float32)
        py = CNICE(cone_data["py"], numpy.float32)
        pz = CNICE(cone_data["pz"], numpy.float32)
    except:
        z2 = cone_data["z2"]
        pixel_size = cone_data["pixel_size"]
        nx_detc = tomo_shape[2]
        nz_detc = tomo_shape[1]
        H = nx_detc*pixel_size/2
        V = nz_detc*pixel_size/2
        px, py, pz = _make_detector_arrays(
            z2, 
            nx_detc, nz_detc, 
            H, V)
    cone_data["px"] = px
    cone_data["py"] = py
    cone_data["pz"] = pz
    px_p = px.ctypes.data_as(ctypes.c_void_p)
    py_p = py.ctypes.data_as(ctypes.c_void_p)
    pz_p = pz.ctypes.data_as(ctypes.c_void_p)
    return pz_p, py_p, px_p

def _get_rot_axis_deviation(cone_data, tomo_shape):
    rot_axis_pos = cone_data["rotation axis position"]
    deviation = -(rot_axis_pos-tomo_shape[2]/2)*cone_data["eps"]
    return deviation

def _align_exp_wrt_rot_axis_position(cone_data, tomo_shape):
    """Align virtual experimental setup with respect to rotation axis position."""
    deviation = _get_rot_axis_deviation(cone_data, tomo_shape)
    cone_data["px"] += deviation
    cone_data["source"][0] += deviation
    print("cone_data['source'] inside _align... fun after update: ", cone_data['source'])

def _make_struct_lab_pointer(cone_data, tomo_shape):
    """Build struct Lab as expected by C++/CUDA.
    """
    try:
        source = cone_data["source"]
        z1 = numpy.linalg.norm(source)
        z2 = numpy.mean(cone_data["py"].flatten()) # rever isso para um caso mais geral.
    except:
        z1, z2 = cone_data["z1"], cone_data["z2"]
        source = numpy.array([0, -z1, 0], dtype=numpy.float32)
        cone_data["source"] = source
    try:
        p0 = cone_data["p0"]
    except:
        p0 = numpy.zeros(3, dtype=numpy.float32)

    mag = (z1 + z2) / z1 # magnificação.
    eps = cone_data["pixel_size"]/mag # effective pixel size.
    cone_data["mag"] = mag
    cone_data["eps"] = eps
    # Lxy = (1/2)*tomo_shape[2]*eps # / numpy.sqrt(2)
    # Lz  = (1/2)*tomo_shape[1]*eps
    # Lz = (1/2)*nz*eps
    # Lx = (1/2)*nx*eps
    # Ly = (1/2)*ny*eps

    if "size" in cone_data:
        size = cone_data["size"]
        if isinstance(size, int):
            size = numpy.array([size, size, size])
        elif len(size) != 3:
            raise ValueError(
                "Invalid reconstruction size encountered: it must be an int or a list of ints of length 3.")
    else:
        size = numpy.array([tomo_shape[2], tomo_shape[2], tomo_shape[1]])

    if size[0] == size[1]:
        Lxy = (1/2)*size[0]*eps
        Lx = Lxy
        Ly = Lxy
        Lz  = (1/2)*size[2]*eps
    else:
        print("The reconstruction size has different number of voxels in x and y dimensions.")
        sure = input("Are you sure thats what you want? (y/n)")
        if sure == "y":
            raise NotImplementedError(
                "Sorry, this functionality is not implemented yet since it is really sample specific.")
        else:
            raise ValueError("Invalid input. Try again with the same number of voxels in x and y.")

    try:
        if cone_data["meio_sup_comp"]:
            Lx = 2*Lxy
            Ly = 2*Lxy
            size[0] *= 2
            size[1] *= 2
    except:
        Lx = Lxy
        Ly = Lxy
        warnings.warn("By default, it is assumed this reconstruction has full lateral compact support.")
    L0 = (Lx, Ly, Lz)
    cone_data["L0"] = L0
    cone_data["size"] = size

    try:
        num_of_integration_points = int(cone_data["number of integration points"])
    except:
        num_of_integration_points = int(20e3)
    
    _align_exp_wrt_rot_axis_position(cone_data, tomo_shape)
    
    lab = Lab_EM(
        Lx=L0[0], Ly=L0[1], Lz=L0[2],
        x0=p0[0], y0=p0[1], z0=p0[2],
        nx=size[0], ny=size[1], nz=size[2],
        sx=source[0], sy=source[1], sz=source[2],
        nbeta=len(cone_data["angs"]),
        ndetc=tomo_shape[1]*tomo_shape[2],
        n_ray_points=num_of_integration_points)
    
    print("lab.sx: ", lab.sx)
    print("cone_data['source']: ", cone_data['source'])

    return lab

def _make_recon_pointer(cone_data, recon_max, lab):
    try:
        recon = CNICE(cone_data["recon"], numpy.float32)
        neg_idxs = recon < 0
        if neg_idxs.any():
            recon[neg_idxs] = 1
            warnings.warn("Negative value encountered in initial-value-recon.")
        higher_than_max_idxs = recon > recon_max
        if higher_than_max_idxs.any():
            recon[higher_than_max_idxs] = recon_max
            warnings.warn("Higher than recon_max value encountered in initial-value-recon.")
    except:
        recon = numpy.ones([lab.nz, lab.ny, lab.nx], dtype=numpy.float32, order="C")
    recon_p = recon.ctypes.data_as(ctypes.c_void_p)
    return recon_p, recon


def tEM_cone(
    tomo: numpy.ndarray,
    flat: numpy.ndarray,
    dark: numpy.ndarray,
    cone_data: dict,
    niter: int,
    gpus: list,
    tv: float = 0.0,
    recon_max: float = numpy.inf,
    save_path: str = "") -> numpy.ndarray:
    """Statistical iterative reconstruction method for transmission tomography: seeks Expectation Maximization (EM).

    It is designed to solve cone-beam (i.e., point source) generated tomograms.

    The algorithm makes the following assumptions:
        1) The sample (reconstruction region) has positive values only. Two remarks about the first assumption: 
            a) it is obviously true for attenuation/absorption coefficients (imaginary part of refraction index) 
\sout{if the source is not radioactive}.
            b) for phase coefficient (real part of refraction index - 1) it might have only negative values (low 
density samples), which is not a problem since...
        2) The flat-field intensity of each pixel for a time interval dt is: 
            a) independent; and 
            b) described by the Poisson distribution (each pixel with its own mean value).
    If this hyphothesis are not met, reconstruction quality might be affected.

    Considerations about usage:
        1) Units of measurement do not matter, as long as the input data is consistent within itself (e.g., all 
variables describing length should use the same unit of measurement).
        2) It is recomended to use this function in a script called by sbatch (slurm command).
        3) It is assumed that all visible GPUs are available to be used by this function.
    
    Algorithm references:
    [1] Fessler, Jeffrey A.. "Statistical Image Reconstruction Methods for Transmission Tomography." (2000).


    Parameters
    ----------
    tomo: numpy.ndarray
        Stack of projections for different angles. Each projection pixel represents a photon count.
        3-dimensional numpy array.
    flat: numpy.ndarray
        Stack of flat-field projections. Each projection pixel represents a photon count.
        3-dimensional numpy array.
    dark: numpy.ndarray
        Stack of dark-field projections. Each projection pixel represents a photon count.
        3-dimensional numpy array.
    cone_data: dict
        Dictionary with the following expected fields:
            "z1": float
                source to sample distance.
                dimension: LENGTH.
            "z2": float
                sample to detector distance.
                dimension: LENGTH.
            "size": int or array_like
                reconstruction size. 
                If array_like: size = [nx, ny, nz].
                dimension: none.
            "pixel_size": float
                pixel width (it is assumed it is a square).
                dimension: LENGTH.
        It may also have:
            "angs": numpy.ndarray
                Must correspond to the tomography projection angles.
                The angles may NOT be equally spaced.
                dimension: RADIANS.
                Default is numpy.linspace(0, 2*numpy.pi, nangs, endpoint=False).
            "recon": numpy.ndarray
                dimension: LENGTH^-1.  
                Default is numpy.ones([nz, ny, nx]).
            "meio_sup_comp": bool
                If the data has half compact support for extended field of view.
                If set to True, the index of the "rotation axis position" must be provided (index on tomogram axis 2).
                Related to the "rotation axis position" key on this same dict.
                Default is False.
            "rotation axis position": int
                Index of rotation axis position on tomogram axis=2.
            "number of integration points": int
                Number of integration points on each ray path.
    niter: int
        Number of iterations.
        Time to complete the reconstruction increases proportionally with this parameter.
    gpus: list
        GPUs available (represented by integers).
    tv: float
        Total-Variation regularization.
        Default is 0.0 (zero).
    recon_max: float
        Maximum -physically acceptable- value admitted in the reconstruction.
        Default is numpy.inf (positive infinite).
        dimension: LENGTH^-1.
    nintegration: int
        Number of integration points on the ray path.
        Higher is more precise.
        Time to complete the reconstruction increases proportionally with this parameter.
    save_path: str or pathlib.Path
        Path to save the reconstruction.
        
    Returns
    -------
    recon: numpy.ndarray
        Returns a 3-dimensional numpy array representing the sample reconstruction.
        dimension: LENGTH^-1.

    Raises
    ------
    KeyError
        If a required key is missing from 'cone_data' dict parameter.
    ValueError
        If:
            a) fed with invalid data; or
            b) if an error in C++ is encountered. In this case, an error code will be returned from the C++ function:
                -4: CPU malloc error.
                -3: got both errors below.
                -2: cudaError or an inapropriate device (e.g., not A100, H100 or better) was detected.
                -1: invalid user input detected at C++ runtime only.
    RuntimeError
        If an error in CUDA is encountered. In this case, an error code will be returned from the C++ function:
            1: CUDA sync error (non-sticky).
            2: CUDA async error (non-sticky).
            3: CUDA sticky error.
    
    Example
    -------
    >>> import sscRaft
    >>> # ...load data to tomo, flat, dark and cone_data...
    >>> recon = sscRaft.tEM_cone(tomo, flat, dark, cone_data, ...)
    """
    if not valid_data(cone_data) or len(gpus) <= 0 or tv < 0.0 or recon_max <= 0.0:
        raise ValueError("Invalid input data. Check warnings for details.")
    ngpus: ctypes.c_int = ctypes.c_int(len(gpus))
    niter: ctypes.c_int = ctypes.c_int(int(niter))
    tv: ctypes.c_float = ctypes.c_float(tv)
    max_val: ctypes.c_float = ctypes.c_float(recon_max)
    legth_unit = cone_data["unit of measurement (length)"]
    cone_data["unit of measurement (attenuation coefficient)"] = legth_unit + "^-1"
    cone_data["tomo.shape"] = tomo.shape

    tomo, flat, dark = _process_cone_data(tomo, flat, dark)
    tomo, flat = CNICE(tomo, np.float32), CNICE(flat, np.float32)
    tomo_p = tomo.ctypes.data_as(ctypes.c_void_p)
    flat_p = flat.ctypes.data_as(ctypes.c_void_p)
    # dark_p = numpy.ascontiguousarray(dark).ctypes.data_as(ctypes.c_void_p) # to do: sum dark after ray integral.
    angs_p = _make_angs_pointer(cone_data, tomo.shape)
    pz_p, py_p, px_p = _make_detector_pointer(cone_data, tomo.shape)
    lab = _make_struct_lab_pointer(cone_data, tomo.shape)
    recon_p, recon = _make_recon_pointer(cone_data, recon_max, lab)

    dt = time.time()
    exe_status = conebeam_tEM_gpu(
        lab, 
        flat_p, px_p, py_p, pz_p, angs_p,
        recon_p, tomo_p, 
        ngpus, niter, tv, max_val)
    dt = time.time() - dt

    if exe_status != 0:
        if exe_status > 0:
            raise RuntimeError("Error code '{}' encountered in CUDA.".format(exe_status))
        else:
            raise ValueError("Error code '{}' encountered in C++.".format(exe_status))
    
    if save_path:
        try: # 'try' since we don't want to waste the recon because an error ocorred while saving...
            path = pathlib.Path(save_path)
            _save_recon(
                path, recon, cone_data,
                _create_method_dict(
                    niter, tv, max_val, cone_data["number of integration points"], 
                    dt, inspect.currentframe().f_code.co_name, sys.argv[0]))
        except:
            warnings.warn("Error ocurred while saving reconstruction.")

    return recon

def _save_recon(path: pathlib.Path, recon: numpy.ndarray, cone_dict: dict, method_dict: dict):
    ok = True
    if path.exists():
        ok = False
        path = "./recon_tEM_{}.h5".format(numpy.random.randint(0, numpy.iinfo(numpy.int64).max))
        warnings.warn(
            "Path to save reconstruction already exists as a path for another file.\n"
            + "\tSaving to path '{}' instead.".format(path))
        path = pathlib.Path(path)
        _save_recon(path, recon, cone_dict, method_dict)
    else:
        with h5py.File(path, "w") as ff:
            dset = ff.create_dataset("attenuation coefficient", data=recon.astype(numpy.float16))
            for key, val in cone_dict.items():
                try:
                    dset.attrs.create(key, val)
                except:
                    ff.create_dataset(key, data=val)
            grp = ff.create_group("method parameters")
            for key, val in method_dict.items():
                try:
                    grp.attrs.create(key, val)
                except:
                    grp.create_dataset(key, data=val)
    return ok

def _create_method_dict(
        niter, tv, max_val, num_integration_points_on_ray_path,
        time, function_name, script_name):
    dic = {}
    dic["algorithm"] = "cone beam transmission expectation maximization"
    dic["number of iterations"] = niter
    dic["total variation regularization parameter"] = tv
    dic["max reconstruction value allowed"] = max_val
    dic["number of integration points on ray path"] = num_integration_points_on_ray_path
    dic["execution time (s)"] = time
    dic["function name"] = function_name
    dic["python entry script"] = script_name
    return dic

def em_cone(tomogram, flat, dark, dic, **kwargs) -> numpy.ndarray:
    """ Expectation maximization for 3D tomographic parallel sinograms

    Args:
        tomogram: three-dimensional stack of sinograms [slices,angles,rays] 
        dic: input dictionary 
        
    Returns:
        (ndarray): stacking 3D reconstructed volume, reconstructed sinograms [z,y,x]

    Dictionary:
    
    *``dic['z1[m]']`` (float): Source-sample distance in meters. Defaults to 500e-3.
    *``dic['z1+z2[m]']`` (float): Source-detector distance in meters. Defaults to 1.0.
    *``dic['detectorPixel[m]']`` (float): Detector pixel size in meters. Defaults to 1.44e-6.
    *``dic['reconSize']`` (int): Reconstruction dimension. Defaults to data shape[0].
    *``dic['gpu']`` (list): List of gpus for processing. Defaults to [0].
    *``dic['nangles']`` (int):  Number of angles
    *``dic['angles']`` (float list):  list of angles in radians
    *``dic['niterations']`` (int,int,int,int):  Tuple. First position refers to the global number of
      iterations for the {Transmission,Emission}/EM algorithms. Second and Third positions refer to the local 
      number of iterations for EM/TV number of iterations, as indicated by Yan & Luminita`s article.
      Number of integration points.
    *``dic['regularization']`` (float):  Regularization parameter for EM/TV
    *``dic['max tolerance']`` (float):  Maximum -physically acceptable- value admitted in the reconstruction.
        Default is numpy.inf (positive infinite).
    *``dic['shift']`` (int): Tuple (value = 0). Rotation axis deviation value.
    *``dic['Energy[keV]']`` (float or list of floats): photon energy in keV.
    *``dic['save path']`` (str,optional): Path to save reconstruction

    """
    # Set default dictionary parameters:

    # dicparams = ('gpu','angles','nangles','niterations','regularization')
    # defaut    = ([0],None,tomogram.shape[1],[8,3,8,200],1)

    # SetDictionary(dic,dicparams,defaut)

    # Set parameters

    gpus                    = dic['gpu']
    niter                   = dic['niterations'][0]
    reg                     = dic['regularization']

    try:
        recon_max           = dic['max tolerance']
    except:
        recon_max           = numpy.inf

    cone_data               = {}

    cone_data['unit of measurement (length)'] = 'm'
    cone_data['Energy[keV]'] = dic['Energy[keV]']
    cone_data['z1']         = dic['z1[m]']
    cone_data['z2']         = dic['z1+z2[m]'] - dic['z1[m]']
    cone_data['pixel_size'] = dic['detectorPixel[m]']
    cone_data['angs']       = numpy.asarray(dic['angles'])

    try:
        cone_data['size'] = dic['reconSize']
    except:
        pass

    try:
        cone_data['meio_sup_comp'] = dic['meio_sup_comp']
    except:
        pass

    try:
        cone_data['rotation axis position'] = dic['shift'] + tomogram.shape[-1]/2
    except:
        cone_data['rotation axis position'] = tomogram.shape[-1]/2

    try:
        cone_data['number of integration points'] = dic['niterations'][3]
    except:
        pass

    try:
        cone_data['save path'] = dic['save path']
    except:
        dic['save path'] = False

    recon = tEM_cone(tomogram, flat, dark, cone_data, niter, gpus, reg, recon_max, save_path = dic['save path'])

    return recon
