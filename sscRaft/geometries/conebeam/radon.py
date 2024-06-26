from ...rafttypes import *

def coneradon_projection(phantom, beta, dic):
    
    z1,z2 = dic['Distances'] 
    cx,cy = dic['Poni'] 
    sx,sy = dic['ShiftPhantom'] 
    rx,ry = dic['ShiftRotation'] 
    points = dic['RayPoints']
    
    ny, nx = phantom.shape
    xm, ym, tm, rm = (1,1,1,1)
    nr, nt = ny, nx  
    
    frame = numpy.zeros((nr, nt))

    
    h = lambda L, n:(L - (-L))/n
    id = lambda x, L, s, h: numpy.floor((x + (L - s))/h).astype(int)
    
    dx = h(xm,nx); dy = h(ym,ny)

    t = numpy.linspace(-tm - (cx), tm - (cx), nt, endpoint=False)
    r = numpy.linspace(-rm - (cy), rm - (cy), nr, endpoint=False)
    p = numpy.linspace(0, z1+z2, (z1+z2)//(dx/points))


    rr, tt = numpy.meshgrid(r, t, indexing='ij')

    if dic['Interpolation'] == 'Nearest': 
        for pi in p: # p
            u = pi*(tt) + rx 
            v = pi*(rr) + ry 
            w = -z1 + pi*(z2)
            xx = u*numpy.cos(beta) + w*numpy.sin(beta)

            ix = id(xx,xm,sx,dx)
            iy = id(v,ym,sy,dy)
            ix[ ix < 0 ] = 0
            iy[ iy < 0 ] = 0
            ix[ ix > nx-1] = nx-1
            iy[ iy > ny-1] = ny-1
            frame += phantom[iy,ix]
    
    if dic['Interpolation'] == 'Linear':    
        # Bilinear interpolation:
        for pi in p: # p
            u = pi*(tt) + rx 
            v = pi*(rr) + ry 
            w = -z1 + pi*(z2)
            xx = u*numpy.cos(beta) + w*numpy.sin(beta)
            
            ix = id(xx,xm,sx,dx)
            iy = id(v,ym,sy,dy)
 
            for neigx in [0,1]:
                for neigy in [0,1]:
                    iyy = iy + neigy 
                    ixx = ix + neigx 
                    ixx[ ixx < 0 ] = 0
                    iyy[ iyy < 0 ] = 0
                    ixx[ ixx > nx-1] = nx-1
                    iyy[ iyy > ny-1] = ny-1
                    frame += phantom[iyy,ixx]/4
    
    dp = p[1] - p[0]
    frame *= dp    

    return frame

def coneradon_gpu(phantom, blockangles, dic, device):

    z1,z2 = dic['Distances'] 
    cx,cy = dic['Poni'] 
    sx,sy = dic['ShiftPhantom'] 
    rx,ry = dic['ShiftRotation'] 
    points = dic['RayPoints']
    
    nz, ny, nx = phantom.shape
    xm, ym, zm, tm, rm = (1,1,1,1,1)
    nr, nt = ny, nx  

    tomogram = numpy.zeros((blockangles.shape[0], nr, nt))

    h = lambda L, n:(L - (-L))/n
    id = lambda x, L, s, h: numpy.floor((x + (L - s))/h).astype(int)
    
    dx = h(xm,nx); dy = h(ym,ny); dz = h(zm,nz)

    t = numpy.linspace(-tm - (cx), tm - (cx), nt, endpoint=False)
    r = numpy.linspace(-rm - (cy), rm - (cy), nr, endpoint=False)
    p = numpy.linspace(0, z1+z2, points)

    beta, rr, tt = numpy.meshgrid(blockangles, r, t, indexing='ij')

    if dic['Interpolation'] == 'Nearest': 
        for pi in p: # p
            u = pi*(tt) + rx 
            v = pi*(rr) + ry 
            w = -z1 + pi*(z2)
            xx = u*numpy.cos(beta) + w*numpy.sin(beta)
            zz = -u*numpy.sin(beta) + w*numpy.cos(beta)   

            ix = id(xx,xm,sx,dx)
            iy = id(v,ym,sy,dy)
            iz = id(zz,zm,0,dz)
            iz[ iz < 0 ] = 0
            ix[ ix < 0 ] = 0
            iy[ iy < 0 ] = 0
            iz[ iz > nz-1] = nz-1
            ix[ ix > nx-1] = nx-1
            iy[ iy > ny-1] = ny-1
            tomogram += phantom[iz,iy,ix]
    
    if dic['Interpolation'] == 'Linear':    
        #trilinear interpolation:
        for pi in p: # p
            u = pi*(tt) + rx 
            v = pi*(rr) + ry 
            w = -z1 + pi*(z2)
            xx = u*numpy.cos(beta) + w*numpy.sin(beta)
            zz = -u*numpy.sin(beta) + w*numpy.cos(beta)   

            ix = id(xx,xm,sx,dx)
            iy = id(v, ym,sy,dy)
            iz = id(zz,zm,0,dz)

            for neigx in [0,1]:
                for neigy in [0,1]:
                    for neigz in [0,1]:
                        izz = iz + neigz 
                        iyy = iy + neigy 
                        ixx = ix + neigx 
                        izz[ izz < 0 ] = 0
                        iyy[ iyy < 0 ] = 0
                        ixx[ ixx < 0 ] = 0
                        izz[ izz > nz-1] = nz-1
                        ixx[ ixx > nx-1] = nx-1
                        iyy[ iyy > ny-1] = ny-1
                        tomogram += phantom[izz,iyy,ixx]/8
    
    dp = p[1] - p[0]
    tomogram *= dp    

    return tomogram 

def _worker_block_frames_(params, idx_start,idx_end, gpu, blocksize):

    nblocks = ( idx_end + 1 - idx_start ) // blocksize

    output_blockframe = params[0]
    phantom = params[1]
    blockangles = params[3]
    
    for k in range(nblocks):
        _start_ = idx_start + k * blocksize
        _end_   = _start_ + blocksize
        output_blockframe[_start_:_end_,:,:] = coneradon_gpu( phantom, blockangles[_start_:_end_], params[6], device = gpu )
    
    
def _build_package_of_frames_(params):

    nangles = params[2]
    gpus = params[4]
    blocksize = params[5]
    ngpus = gpus
    
    b = int( numpy.ceil( nangles/ngpus )  ) 
    
    processes = []
    for k in range( ngpus ):
        begin_ = k*b
        end_   = min( (k+1)*b, nangles )

        p = multiprocessing.Process(target=_worker_block_frames_, args=(params, begin_, end_, k, blocksize ))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
   

def coneradon_gpu_block( phantom, gpus, blocksize, dic):

    """ Cone Ray integral for a given input phantom
    
    Args:
        phantom: a digital squared matrices.
        nangles: number of angles.
        angles: a tuple (start angle, end angle) in degrees
        gpus: sequence of GPUs
        blocksize: number of frames to process (batch, per GPU)
             
    Returns:
        (ndarray): block of frames with shape [nangles, size, size], with size corresponding to the
        size of each phantom. 

    * Requires a GPU: be sure that both phantom and sinogram fit into your GPU memory.  
    * This function uses a shared array through package 'SharedArray'.
    * The total number of images will be divided by the number of GPUs used in the input list. Each block
      will be processed by an independent GPU (as long as the GPU memory holds these pointers)
    * SharedArray names are provided by uuid.uuid4() 

    * Blocksize variable is bounded by :math:`N/L` with :math:`N` the number of tomogram frames
      and :math:`L` the number of GPUs used.


    """
    angle_start, angle_end = dic['angles']
    name    = str( uuid.uuid4())
    name2   = str( uuid.uuid4())
    nimages = phantom.shape[0]
    imgsize = phantom.shape[-1]
    nangles = dic['nangles']

    blockangles = numpy.linspace(angle_start*numpy.pi/180,angle_end*numpy.pi/180, nangles,endpoint=False)

    if blocksize > nangles // gpus:
        print('ssc-radon: Error! Please check block size! Function coneradon_gpu_block().')
    
    try:
        sa.delete(name)
    except:
        pass

    try:
        sa.delete(name2)
    except:
        pass
            
    output_blockframe  = sa.create(name,[nangles, imgsize, imgsize], dtype=numpy.float32)
    input_phantom      = sa.create(name2, phantom.shape, dtype=numpy.float32)
    
    input_phantom = phantom
    _params_ = ( output_blockframe, input_phantom, nangles, blockangles, gpus, blocksize, dic)
    
    _build_package_of_frames_ ( _params_ )

    sa.delete(name)
    sa.delete(name2)

    return output_blockframe



def radon_gpu360(phantom, angles, device, *args):

    """ Radon transform for a given input phantom
    
    Args:
        phantom: digital squared input phantom.
        angles: number of angles.
        device: GPU device number.
        args: support domain at the x-axis, default value is 1. 
             
    Returns:
        (ndarray): digital radon matrix

    * Requires a GPU: be sure that both phantom and sinogram fit into your GPU memory. 

    """
    if not args:
        a = 1.0
    else:
        a = args[0] #numpy.sqrt(2.0)

    if len(phantom.shape)==3:
        
        nimgs = phantom.shape[0]
        n     = phantom.shape[1]
        rays  = numpy.ceil( a * n ).astype(numpy.int)
        img_size = n
        
        f = CNICE(phantom)
        f_p = f.ctypes.data_as(ctypes.c_void_p)
        
        sino = numpy.zeros((nimgs,angles,rays),dtype=numpy.float32)
        sino_p = sino.ctypes.data_as(ctypes.c_void_p)
        
        libraft.radonp_gpu360(sino_p, f_p, ctypes.c_int(img_size), ctypes.c_int(rays), ctypes.c_int(angles), ctypes.c_int(device), ctypes.c_int(nimgs), ctypes.c_float(a) ) 
        
        dt = (2.0*a)/(rays-1)
        itmin = ( numpy.ceil( (-1 + a)/dt) ).astype(numpy.int) 
        itmax = ( numpy.ceil( ( 1 + a)/dt) ).astype(numpy.int) 
        
        return sino[:,:,itmin:itmax+1]

    else:
        
        n     = phantom.shape[0]
        rays  = numpy.ceil( a * n ).astype(numpy.int)
        img_size = n
        
        f = CNICE(phantom)
        f_p = f.ctypes.data_as(ctypes.c_void_p)
        
        sino = numpy.zeros((angles,rays),dtype=numpy.float32)
        sino_p = sino.ctypes.data_as(ctypes.c_void_p)
        
        libraft.radonp_gpu360(sino_p, f_p, ctypes.c_int(img_size), ctypes.c_int(rays), ctypes.c_int(angles), ctypes.c_int(device), ctypes.c_int(1), ctypes.c_float(a) ) 
        
        dt = (2.0*a)/(rays-1)
        itmin = ( numpy.ceil( (-1 + a)/dt) ).astype(numpy.int) 
        itmax = ( numpy.ceil( ( 1 + a)/dt) ).astype(numpy.int) 
        
        return sino[:,itmin:itmax+1]
    

def _calc_n_chunks(n_gpus, phantom, nbeta, nv, nh, gpu_mem_GB=40):
    tomo_GB = nbeta*nv*nh*4/1e9
    phan_GB = phantom.nbytes/1e9
    n_chunks = None
    if nbeta < n_gpus:
        raise ValueError('nbeta < n_gpus, perhaps you should try another aproach to use the GPUs in parallel.')
    for i in range(1, (nbeta//n_gpus)+1):
        if phan_GB + tomo_GB/(i*n_gpus) < .98*gpu_mem_GB:
            if nbeta // i >= n_gpus:
                n_chunks = i
                break
    if n_chunks is None:
        raise MemoryError('GPU(s) memory shortage error: not enough memory to allocate the phantom.')
    return n_chunks

def _tomo_manager_worker(data_dict, gpu_id):
    tomo = ConeBeamRadon(data_dict=data_dict)
    tomo.make_tomo(gpu_id)

def _load_phantom(data_dict):
    try: 
        if isinstance(data_dict["phantom"], str) or isinstance(data_dict["phantom"], pathlib.PurePath):
            phantom_path = data_dict["phantom"]
            if phantom_path.endswith(".npy"):
                phantom = numpy.load(phantom_path)
            elif phantom_path.endswith(".h5"):
                with h5py.File(phantom_path, "r") as h5_phantom:
                    phantom = h5_phantom["refractive_index"][:]
            else:
                raise ValueError("The phantom format is not supported. Use '.npy' or '.h5' formats.")
        elif isinstance(data_dict["phantom"], numpy.ndarray):
            phantom = data_dict["phantom"]
            if phantom.ndim != 3:
                raise ValueError(
                    "The object 'data_dict[\"phantom\"]' should be a 3 dimensional numpy.ndarray." 
                    + " It has {} dimension(s).".format(phantom.ndim))
        else:
            raise ValueError("The object 'data_dict[\"phantom\"]' is unknown.")
    except KeyError:
        raise KeyError("Phantom is missing from 'data_dict'." 
            + " Expected key: 'phantom' was not defined.")
    return phantom.astype(numpy.float32)

def make_tomo(data_dict, n_gpus, save="hdf5", verbose=True):
    nh, nv = data_dict["nh"], data_dict["nv"]
    phantom = _load_phantom(data_dict)
    nbeta = data_dict["beta"].shape[0]
    res = []
    if verbose:
        print("Número de bytes do phantom: ", phantom.nbytes)
        print("Número de bytes de um phantom de mesmas dimesões: ", numpy.empty(phantom.shape, dtype=numpy.float32).nbytes)
    t_start = time.time()
    n_chunks = _calc_n_chunks(n_gpus, phantom, nbeta, nv, nh)
    beta_chunks = numpy.array_split(data_dict["beta"], n_chunks)
    if verbose:
        print("Número de chunks: ", n_chunks)
    for idx_chunk in range(n_chunks):
        betas = numpy.array_split(beta_chunks[idx_chunk], n_gpus)
        if verbose:
            print("Tamanho do beta de cada GPU: ", [ len(b) for b in betas])
        dicts = [ make_data_dict(
            phantom=phantom, 
            beta=betas[i], 
            px=data_dict["px"], py=data_dict["py"], pz=data_dict["pz"], 
            sx=data_dict["sx"], sy=data_dict["sy"], sz=data_dict["sz"], 
            Lx=data_dict["Lx"], Ly=data_dict["Ly"], Lz=data_dict["Lz"],
            x0=data_dict["x0"], y0=data_dict["y0"], z0=data_dict["z0"],
            nh=data_dict["nh"], nv=data_dict["nv"],
            n_ray_points=data_dict["n_ray_points"]) for i in range(n_gpus) ]
        tomo_gpus = []
        for i, dic in enumerate(dicts):
            tomo_shape = (dic["beta"].shape[0], nv, nh)
            dic["tomo_mem"] = SM(create=True, size=numpy.empty(tomo_shape, dtype=numpy.float32).nbytes)
            tomo_gpus.append(numpy.ndarray(tomo_shape, buffer=dic["tomo_mem"].buf, dtype=numpy.float32, order='C'))

        t_start_chunk = time.time()
        tomo_procs = []
        manager = mp.Manager()
        for arg in zip(dicts, numpy.arange(n_gpus)):
            p = mp.Process(target=_tomo_manager_worker, args=(arg[0], arg[1]))
            tomo_procs.append(p)
            p.start()
        for proc in tomo_procs:
            proc.join()
        print("Total time on GPUs: " + str(time.time()-t_start_chunk))
        res.append(numpy.vstack(tomo_gpus))
        print("Total time after concatenating: " + str(time.time()-t_start_chunk))
        for dic in dicts:
            dic["tomo_mem"].close()
            dic["tomo_mem"].unlink()
    res = numpy.vstack(res)
    print("Total time: " + str(time.time()-t_start))
    if save == "hdf5":
        with h5py.File("tomo.h5", "w") as h5_tomo: 
            h5_tomo.create_dataset("refractive_index", data=res)
    elif save == "npy":
        numpy.save("tomo.npy", res)
        print("Total time after saving: " + str(time.time()-t_start))
    plt.imshow(res[nbeta//2])
    plt.savefig("tomo_print.png")
    plt.close()
    return res

def make_tomo_for_zoom_iterative(data_dict, n_gpus, save="hdf5", verbose=True):
    nh, nv = data_dict["nh"], data_dict["nv"]
    phantom = data_dict['phantom']
    nbeta = data_dict["beta"].shape[0]
    res = []
    if verbose:
        print("Número de bytes do phantom: ", phantom.nbytes)
        print("Número de bytes de um phantom de mesmas dimesões: ", numpy.empty(phantom.shape, dtype=numpy.float32).nbytes)
    t_start = time.time()
    n_chunks = _calc_n_chunks(n_gpus, phantom, nbeta, nv, nh)
    beta_chunks = numpy.array_split(data_dict["beta"], n_chunks)
    if verbose:
        print("Número de chunks: ", n_chunks)
    for idx_chunk in range(n_chunks):
        betas = numpy.array_split(beta_chunks[idx_chunk], n_gpus)
        if verbose:
            print("Tamanho do beta de cada GPU: ", [ len(b) for b in betas])
        dicts = [ make_data_dict(
            phantom=phantom, 
            beta=betas[i], 
            px=data_dict["px"], py=data_dict["py"], pz=data_dict["pz"], 
            sx=data_dict["sx"], sy=data_dict["sy"], sz=data_dict["sz"], 
            Lx=data_dict["Lx"], Ly=data_dict["Ly"], Lz=data_dict["Lz"],
            x0=data_dict["x0"], y0=data_dict["y0"], z0=data_dict["z0"],
            nh=data_dict["nh"], nv=data_dict["nv"],
            n_ray_points=data_dict["n_ray_points"]) for i in range(n_gpus) ]
        tomo_gpus = []
        for i, dic in enumerate(dicts):
            tomo_shape = (dic["beta"].shape[0], nv, nh)
            dic["tomo_mem"] = SM(create=True, size=numpy.empty(tomo_shape, dtype=numpy.float32).nbytes)
            tomo_gpus.append(numpy.ndarray(tomo_shape, buffer=dic["tomo_mem"].buf, dtype=numpy.float32, order='C'))

        t_start_chunk = time.time()
        tomo_procs = []
        manager = mp.Manager()
        for arg in zip(dicts, numpy.arange(n_gpus)):
            p = mp.Process(target=_tomo_manager_worker, args=(arg[0], arg[1]))
            tomo_procs.append(p)
            p.start()
        for proc in tomo_procs:
            proc.join()
        #print("Total time on GPUs: " + str(time.time()-t_start_chunk))
        res.append(numpy.vstack(tomo_gpus))
        #print("Total time after concatenating: " + str(time.time()-t_start_chunk))
        for dic in dicts:
            dic["tomo_mem"].close()
            dic["tomo_mem"].unlink()
    res = numpy.vstack(res)
    #print("Total time: " + str(time.time()-t_start))
    if save == "hdf5":
        with h5py.File("tomo.h5", "w") as h5_tomo: 
            h5_tomo.create_dataset("refractive_index", data=res)
    elif save == "npy":
        numpy.save("tomo.npy", res)
        #print("Total time after saving: " + str(time.time()-t_start))
    plt.imshow(res[nbeta//2])
    plt.savefig("tomo_print.png")
    plt.close()
    return res

class ConeBeamRadon:
    """Class to handle cone beam tomographies of samples/phantom.
        IMPORTANT: the origin of the coordinate system is in the center of rotation.

    Important Notes
    ---------------
        1. The coordinate system is located at the center of rotation.
        2. The z axis of the coordinate system is the same as the axis of rotation.
    Therefore, whenever there is reference about coordinates in this code, it is referring to the coordinate system described above in (1) and (2).

    Parameters 
    ----------
    Described in the constructor.

    Methods (public)
    ----------------
    make_tomo(self) -> int
    load_new_sample(self, new_sample) -> None

    Class attributes
    ----------------
    BEAMLINE : str
        Beamline name: "VIRTUAL_MOGNO".

    Object attributes
    -----------------
    data_dict : dict
        Created by the constructor. 
        Input for extremely customizable configuration (non-equidistant projection angles, arbitray location for: source, each detector pixel and phantom):
            data_dict['phantom']: phantom (3 dimensional numpy array). 
            data_dict['source']: location of the x-ray source ([x, y, z] numpy array). 
            data_dict['beta']: betas (numpy array).
            data_dict['px']: x coordinate of detector pixels (numpy array).
            data_dict['py']: y coordinate of detector pixels (numpy array).
            data_dict['pz']: z coordinate of detector pixels (numpy array).
            data_dict['shift_phantom_x0']: x coordinate of phantom shift (float).
            data_dict['shift_phantom_y0']: y coordinate of phantom shift (float).
            data_dict['shift_phantom_z0']: z coordinate of phantom shift (float).
            data_dict['phantom_X']: phantom size in the x direction, where the size interval is [-Lx,Lx] (float).
            data_dict['phantom_Y']: phantom size in the y direction, where the size interval is [-Ly,Ly] (float).
            data_dict['phantom_Z']: phantom size in the z direction, where the size interval is [-Lz,Lz] (float).
            data_dict['nh']: number of columns of pixels in the detector (int).
            data_dict['nv']: number of rows of pixels in the detector  (int).
            data_dict['n_detector']: nh*nv = number of detector pixels (int).
            data_dict['n_ray_points']: number of ray integration points (int).

    """
    BEAMLINE = "VIRTUAL_MOGNO"
    def __init__(self, *, data_dict: dict = {}, input_file_path: str = ""):
        """Constructor for the ConeBeamRadon class.

        Parameters
        ----------
        input_file_path : str
            Input for standard configuration.
        data_dict : dict
            Input for extremely customizable configuration (non-equidistant projection angles, arbitray location for: source, each detector pixel and phantom):
                data_dict['phantom']: phantom (3 dimensional numpy array). 
                data_dict['source']: location of the x-ray source ([x, y, z] numpy array). 
                data_dict['beta']: betas (numpy array).
                data_dict['px']: x coordinate of detector pixels (numpy array).
                data_dict['py']: y coordinate of detector pixels (numpy array).
                data_dict['pz']: z coordinate of detector pixels (numpy array).
                data_dict['shift_phantom_x0']: x coordinate of phantom shift (float).
                data_dict['shift_phantom_y0']: y coordinate of phantom shift (float).
                data_dict['shift_phantom_z0']: z coordinate of phantom shift (float).
                data_dict['phantom_X']: phantom size in the x direction, where the size interval is [-Lx,Lx] (float).
                data_dict['phantom_Y']: phantom size in the y direction, where the size interval is [-Ly,Ly] (float).
                data_dict['phantom_Z']: phantom size in the z direction, where the size interval is [-Lz,Lz] (float).
                data_dict['nh']: number of columns of pixels in the detector (int).
                data_dict['nv']: number of rows of pixels in the detector  (int).
                data_dict['n_detector']: nh*nv = number of detector pixels (int).
                data_dict['n_ray_points']: number of ray integration points (int).

        Returns
        -------
        None

        Raises
        ------
        ValueError
        TypeError     
        """
        if input_file_path and data_dict:
            warnings.warn("Both input_file_path and data_dict were provided. The default for this case is to use just the data_dict.\n", RuntimeWarning)
        if data_dict:
            if isinstance(data_dict, dict):
                self.data_dict = data_dict
                #print(self.data_dict)
            else:
                raise TypeError("The parameter 'data_dict' is not a dictionary.")
        elif input_file_path:
            if isinstance(input_file_path, str) or isinstance(input_file_path, pathlib.PurePath):
                self.data_dict = self.__make_data_dict_from_input_file(input_file_path)
            else:
                raise TypeError("The parameter 'input_file_path' is not a string or a pathlib.PurePath object.")
        else:
            raise ValueError("Missing parameters to create ConeBeamRadon object. input_file_path or data_dict must be provided.\n")
        if self.__is_valid_data():
            print("Input parameters validated.")
        else:
            raise ValueError("Input data is not valid.\n" + self.error_msg)

    def make_tomo(self, gpu_id: int = 0) -> int:
        """Method to call the cone beam tomography simulator algorithm implemented in C++/CUDA.
                The time complexity of the implemented algorithm is O(M*I*B), where:
                        B is the number of projections in the tomography, i.e., the number of angles; 
                        I is the number of integration points along a ray;
                        M is the number of pixel detectors.
        
        Parameters
        ----------
        None

        Returns
        -------
        int
            0 if successful, error code otherwise.

            The error code describes runtime errors in C++/CUDA. 

            Meaning of each error code (more information about the errors may be printed by the C++ std::cerr function):
                -1: error to allocate memory in GPU.
                -2: segmentation fault.
                ...        
        """
        t_start = time.time()
        # ctypes.POINTER(ctypes.c_float)
        self.data_dict["px"] = numpy.ascontiguousarray(self.data_dict["px"].astype(numpy.float32))
        px_pointer = self.data_dict["px"].ctypes.data_as(ctypes.c_ctypes.c_void_p)
        self.data_dict["py"] = numpy.ascontiguousarray(self.data_dict["py"].astype(numpy.float32))
        py_pointer = self.data_dict["py"].ctypes.data_as(ctypes.c_ctypes.c_void_p)
        self.data_dict["pz"] = numpy.ascontiguousarray(self.data_dict["pz"].astype(numpy.float32))
        pz_pointer = self.data_dict["pz"].ctypes.data_as(ctypes.c_ctypes.c_void_p)
        self.data_dict["beta"] = numpy.ascontiguousarray(self.data_dict["beta"].astype(numpy.float32))
        beta_pointer = self.data_dict["beta"].ctypes.data_as(ctypes.c_ctypes.c_void_p)
        self.data_dict["phantom"] = numpy.ascontiguousarray(self.data_dict["phantom"].astype(numpy.float32))
        phantom_pointer = self.data_dict["phantom"].ctypes.data_as(ctypes.c_ctypes.c_void_p)
        # creating the struct lab:
        self.lab = Lab_CB(
            x=self.data_dict["Lx"],
            y=self.data_dict["Ly"],
            z=self.data_dict["Lz"],
            x0=self.data_dict["x0"],
            y0=self.data_dict["y0"],
            z0=self.data_dict["z0"],
            nx=self.data_dict["phantom"].shape[2],
            ny=self.data_dict["phantom"].shape[1],
            nz=self.data_dict["phantom"].shape[0],
            nbeta=self.data_dict["beta"].shape[0],
            sx=self.data_dict["sx"],
            sy=self.data_dict["sy"],
            sz=self.data_dict["sz"],
            n_detector=self.data_dict["n_detector"],
            n_ray_points=self.data_dict["n_ray_points"])
        print(self.lab.sx, self.lab.sy, self.lab.sz)
        # creating the tomography 3D array:
        try:
            shm = shared_memory.SharedMemory(name=self.data_dict["tomo_mem"].name)
            self.tomo = numpy.ndarray((self.lab.nbeta, self.data_dict["nv"], self.data_dict["nh"]), buffer=shm.buf, dtype=numpy.float32, order='C') 
        except:
            self.tomo = numpy.empty((self.lab.nbeta, self.data_dict["nv"], self.data_dict["nh"]), dtype=numpy.float32, order='C')
            self.tomo = numpy.ascontiguousarray(self.tomo.astype(numpy.float32))
        tomo_pointer = self.tomo.ctypes.data_as(ctypes.c_ctypes.c_void_p)
        # calling the C++/CUDA function to simulate the tomography:
        print('Starting the computation of the x-ray path integrals...')
        status_code = libraft.cbradon(self.lab, px_pointer, py_pointer, pz_pointer, beta_pointer, phantom_pointer, tomo_pointer, gpu_id)
        try:
            shm.close()
        except:
            warnings.warn("conebram tomo (cbtomo) module was used with no shared memory.")
        t_end = time.time()
        print("Time: " + str(t_end-t_start))
        return status_code

    def load_new_sample(self, new_sample) -> None:
        """Load a new phantom/sample.
        
        Parameters
        ----------
        new_sample : 3 dimensional array_like object

        Returns
        -------
        None
        """
        self.phantom = numpy.array(new_sample) # atualizar o data_dict!!!!!!!!!!!!!!

    def __make_data_dict_from_input_file(self, input_file_path: str) -> dict:
        """Make the data dictionary as in the format expected by the self.make_tomo method. 

        Parameters
        ----------
        input_file_path : str
            Path of the input file from wich the data will be loaded.

        Returns
        -------
        dict
            dictionary as described in ConeBeamRadon class for the instance attribute 'data_dict'. 

        Raises
        ------
        ValueError
        """
        print('Loading files...')
        input_dict = self.__read_input_file(input_file_path)
        if input_dict["phantom_path"].endswith(".npy"):
            phantom = numpy.load(input_dict["phantom_path"])
        else:
            raise ValueError("Unknown file extension: " + input_dict["phantom_path"].split(".")[-1])
        # extracting input data from input_dict:
        phantom_X = float(input_dict["phantom_X"])
        phantom_Y = float(input_dict["phantom_Y"])
        phantom_Z = float(input_dict["phantom_Z"])
        z1 = float(input_dict["z1"])
        z2 = float(input_dict["z2"])
        source = numpy.array([0, -z1, 0])
        detector_H = float(input_dict["detector_H"])
        detector_V = float(input_dict["detector_V"])
        nh = int(input_dict["nh"])
        nv = int(input_dict["nv"])
        nbeta = int(input_dict["nbeta"])
        if input_dict["number_of_integration_points_on_ray_path(optional)"]:
            n_ray_points = int(input_dict["number_of_integration_points_on_ray_path(optional)"])
        else:
            n_ray_points = 10*numpy.max(phantom.shape)
        beta = numpy.linspace(0, 2*numpy.pi, nbeta, endpoint=False)
        n_detector = nh * nv
        h = numpy.linspace(-detector_H, detector_H, nh)
        v = numpy.linspace(-detector_V, detector_V, nv)
        hh, vv = numpy.meshgrid(h, v)
        hh = hh.flatten()
        vv = vv.flatten()
        yy = numpy.linspace(z2, z2, n_detector)
        shift_phantom = numpy.array([0, 0, 0])
        # creating the data dictionary as expected by the make_tomo method:
        data_dict = {}
        data_dict['x0'] = shift_phantom[0]
        data_dict['y0'] = shift_phantom[1] 
        data_dict['z0'] = shift_phantom[2] 
        data_dict['Lx'] = phantom_X
        data_dict['Ly'] = phantom_Y
        data_dict['Lz'] = phantom_Z
        data_dict['phantom'] = phantom
        data_dict['sx'] = source[0]
        data_dict['sy'] = source[1]
        data_dict['sz'] = source[2]
        data_dict['beta'] = beta
        data_dict['px'] = hh
        data_dict['py'] = yy
        data_dict['pz'] = vv
        data_dict['n_detector'] = n_detector
        data_dict['n_ray_points'] = n_ray_points
        # keeping track of the dimensions of the detector:
        data_dict['nh'] = nh
        data_dict['nv'] = nv
        return data_dict

    def __read_input_file(self, input_file_path: str) -> dict:
        """Load the data needed to simulate a cone beam tomography. 
        
        Parameters
        ----------
        input_file_path : str
            Path of the input file from wich the data will be loaded.

        Returns
        -------
        dict
        """
        input_dict = {}
        with open(input_file_path) as input_file:
            lines = input_file.read().splitlines()
        for line in lines:
            print(line.split("="))
            key, value = line.split("=")
            input_dict[key] = value
        return input_dict

    def __is_valid_data(self) -> bool:
        """Check if the providaded data is both physically and mathematically consistent.
        
        Parameters
        ----------
        None

        Returns
        -------
        bool
            Returns True if data is consistent, False otherwise.
        """
        return True


def paola_dic_to_cbdata_dict(dic: dict) -> dict:
    """Create a 'data_dict' dictionary (as expected from the class ConeBeamRadon) from Paola's 'dic' dictionary.
    
    Parameters
    ----------
    dict
        Paola's 'dic' dictionary.

    Returns
    -------
    dict
        Returns the 'data_dict' dictionary.
    """
    # unpacking Payola's dic:
    nbeta = dic['nangles']
    beta_ini, beta_fin = dic['angles']
    z1, z2 = dic['Distances']
    Dsd = z1+z2
    beta = numpy.linspace(beta_ini*numpy.pi/180, beta_fin*numpy.pi/180, nbeta, endpoint=False)
    nh = dic['phantom'].shape[2]
    nv = dic['phantom'].shape[0]
    n_detector = nh*nv
    H, V = dic['DetectorSize']
    rx, ry = dic['ShiftRotation']
    cx, cz = dic['Poni']
    h = numpy.linspace(-H-rx-cx, H-rx-cx, nh)
    v = numpy.linspace(-V-cz, V-cz, nv)
    hh, vv = numpy.meshgrid(h, v)
    hh = hh.flatten()
    vv = vv.flatten()
    yy = numpy.linspace(z2, z2, n_detector)
    source = numpy.array([0-rx, -z1-ry, 0])
    sx, sy = dic['ShiftPhantom']
    # creating the data dictionary as expected by the make_tomo method:
    data_dict = {}
    data_dict['x0'] = sx - rx
    data_dict['y0'] = sy - ry
    data_dict['z0'] = 0
    data_dict['Lx'] = dic['PhantomSize'][0]
    data_dict['Ly'] = dic['PhantomSize'][1]
    data_dict['Lz'] = dic['PhantomSize'][2]
    data_dict['phantom'] = dic['phantom'] # lembrar de fazer isso antes de alimentar essa função.
    data_dict['sx'] = source[0]
    data_dict['sy'] = source[1]
    data_dict['sz'] = source[2]
    data_dict['beta'] = beta
    data_dict['px'] = hh
    data_dict['py'] = yy
    data_dict['pz'] = vv
    data_dict['n_detector'] = n_detector
    data_dict['n_ray_points'] = dic['RayPoints']
    # keeping track of the dimensions of the detector:
    data_dict['nh'] = nh
    data_dict['nv'] = nv
    return data_dict

def make_detector(z2, nh, nv, H, V):
    h = numpy.linspace(-H, H, nh)
    v = numpy.linspace(-V, V, nv)
    hh, vv = numpy.meshgrid(h, v)
    hh = hh.flatten()
    vv = vv.flatten()
    yy = numpy.linspace(z2, z2, nh*nv)
    return hh, yy, vv

def make_data_dict(
    phantom=None,
    z1=2, z2=2,
    beta=None,
    px=None, py=None, pz=None,
    sx=None, sy=None, sz=None,
    Lx=1, Ly=1, Lz=1,
    x0=0, y0=0, z0=0,
    nh=None, nv=None,
    H=None, V=None,
    n_ray_points=None) -> dict:
    """ Create a 'data_dict' dictionary (as expected from the class ConeBeamRadon). 

    Parameters
    ----------
    

    Returns
    -------
    dict
        Returns the 'data_dict' dictionary.

    Raises
    ------
    ValueError
    """
    if (z1 <= 0) or (z2 <= 0):
        raise ValueError("'z1' and 'z2' are distances, not coordinates. They must be positive values.")
    if phantom is not None:
        n = int(numpy.max(phantom.shape))
    else:
        n = 1024
    if nh is None:
        nh = n 
    if nv is None:
        nv = n
    if n_ray_points is None:
        n_ray_points = int(1e4) # 10*int(max(nh, nv))
    if (sx is None) or (sy is None) or (sz is None):
        sx, sy, sz = 0, -z1, 0
    else:
        z1 = -sy
    if (px is None) or (py is None) or (pz is None):
        if (H is None) or (V is None):
            H = Lx*(z1+z2)/z1
            V = Lz*(z1+z2)/z1
        px, py, pz = make_detector(z2, nh, nv, H, V)
    if beta is None:
        beta = numpy.linspace(0, 2*numpy.pi, n, endpoint=False)
    data_dict = {}
    data_dict['phantom'] = phantom
    data_dict['x0'] = x0 # shift phantom from center of rotation in the x axis.
    data_dict['y0'] = y0 # shift phantom from center of rotation in the x axis.
    data_dict['z0'] = z0 # shift phantom from center of rotation in the x axis.
    data_dict['Lx'] = Lx
    data_dict['Ly'] = Ly
    data_dict['Lz'] = Lz
    data_dict['sx'] = sx
    data_dict['sy'] = sy
    data_dict['sz'] = sz
    data_dict['beta'] = beta
    data_dict['px'] = px
    data_dict['py'] = py
    data_dict['pz'] = pz
    data_dict['n_detector'] = nh*nv
    data_dict['n_ray_points'] = n_ray_points
    # keeping track of the dimensions of the detector:
    data_dict['nh'] = nh
    data_dict['nv'] = nv
    return data_dict


def main(input_file_path: str = "") -> None:
    """Example."""
    # aceitar arg que pode ser input_file_path ou dict, verificar tipo com isinstance.
    status_code: int = 0
    if not input_file_path:
        input_file_path = "cbradon/input.txt" # ou: sys.argv[1]
    cbradon = ConeBeamRadon(input_file_path=input_file_path)
    logging.info("Created object of class ConeBeamRadon.\n\t {data_dict} \n\n".format(data_dict=str(cbradon.data_dict)))
    status_code = cbradon.make_tomo()
    logging.info("'make_tomo' function from ConeBeamRadon class finished with status code {}.\n\n\n\n".format(status_code))

#logging.basicConfig(filename="log_cbradon.txt", format='%(asctime)s %(message)s', level=logging.DEBUG)
#logging.info("Module cbradon.tomo imported.\n")