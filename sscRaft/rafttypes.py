#We use the one encoding: utf8

# Ctypes =========
import ctypes
from ctypes import *
import ctypes.util

from .utils import *

from sscRaft import __version__


############# Load required libraries ##############

libcudart = ctypes.CDLL('libcudart.so', mode=ctypes.RTLD_GLOBAL)
libstdcpp = ctypes.CDLL( ctypes.util.find_library( "stdc++" ), mode=ctypes.RTLD_GLOBAL )
# libblas   = ctypes.CDLL( ctypes.util.find_library( "blas" ), mode=ctypes.RTLD_GLOBAL )
# libfftw3  = ctypes.CDLL( ctypes.util.find_library( "fftw3" ), mode=ctypes.RTLD_GLOBAL )
# libfftw3_threads  = ctypes.CDLL( ctypes.util.find_library( "fftw3_threads" ), mode=ctypes.RTLD_GLOBAL )

############# Load library C/C++/CUDA ##############

_lib = "lib/libraft"

ext = '.so'

def load_library(lib,ext):
    _path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib + ext
    lib = ctypes.CDLL(_path)
    return lib

libraft  = load_library(_lib, ext)


#########################
#|      ssc-raft       |#
#|  Struct Prototypes  |#
#########################

class coord(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]

class dim3(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("z", ctypes.c_int)]

class DIM(ctypes.Structure):
    _fields_ = [("size", dim3), ("pad", dim3), ("blocksize", ctypes.c_int), ("padding_mode", ctypes.c_int)]

class GEO(ctypes.Structure):
    _fields_ = [("detector_pixel", coord), 
                ("obj_pixel", coord), 
                ("z1", ctypes.c_float), 
                ("z2", ctypes.c_float), 
                ("magnitude", ctypes.c_float), 
                ("energy", ctypes.c_float), 
                ("wavelength", ctypes.c_float)
                ]

class CEF(ctypes.Structure):
    _fields_ = [("method", ctypes.c_int), 
                ("beta_delta", ctypes.c_float), 
                ("regularization", ctypes.c_float),
                ("post_process", ctypes.c_int)]   

class TomoAttributes(ctypes.Structure):
    _fields_ = [("n_length", ctypes.c_int), 
                ("n_angle", ctypes.c_int), 
                ("n_slice", ctypes.c_int),
                ("size_median", ctypes.c_int), 
                ("dim", ctypes.c_int), 
                ("large_stripes_size", ctypes.c_int),
                ("background_ratio", ctypes.c_float), 
                ("drop_ratio", ctypes.c_float)
            ]
    
class RF(ctypes.Structure):
    _fields_ = [("method", ctypes.c_int), 
                ("rings_block", ctypes.c_int), 
                ("rings_lambda", ctypes.c_float),
                ("all_stripes", TomoAttributes)
                ]   

class ALGN(ctypes.Structure):
    _fields_ = [("method", ctypes.c_int), 
                ("excentric_offset", ctypes.c_int), 
                ("rotation_axis_offset", ctypes.c_float),
                ("offset_method", ctypes.c_int)]  

class REC(ctypes.Structure):
    _fields_ = [("method", ctypes.c_int), 
                ("filter", ctypes.c_int), 
                ("filter_reg", ctypes.c_float),
                ("filter_pad", ctypes.c_int),
                ("filter_padMode", ctypes.c_int),
                ("paganin_slices", ctypes.c_float),
                ("iterations", ctypes.c_int),
                ("rotation_axis_offset", ctypes.c_float),
                ("total_variation", ctypes.c_float),
                ("interpolation", ctypes.c_int)
                ] 

class CFG(ctypes.Structure):
    _fields_ = [("back_corr_method", ctypes.c_int),
                ("geometry", GEO), 
                ("obj", DIM),
                ("tomo", DIM),
                ("flat", DIM),
                ("dark", DIM),
                ("ContrastParam", CEF),
                ("RingsParam", RF),
                ("AlignParam", ALGN),
                ("ReconParam", REC)
                ] 

#########################
#|      ssc-raft       |#
#|  Struct Functions   |#
#########################   
#        
def dimension(size, pad = (0,0,0), blocksize = 0, padd_mode = 'none'):
    x, y, z = size
    px      = int(pad[0])
    py      = int(pad[1])
    pz      = int(pad[2])

    pad_sum = px + py + pz

    size_   = dim3(x =  x, y =  y, z =  z)
    pad_    = dim3(x = px, y = py, z = pz)

    padd_mode = PaddMode(padd_mode)

    if int(pad_sum) == 0:
        padd_mode = PaddMode('none')

    if padd_mode == PaddMode('none'):
        pad_ = dim3(x = 0, y = 0, z = 0)

    dim_size = DIM(size = size_, pad = pad_, blocksize = blocksize, padding_mode = padd_mode)

    return dim_size

def define_geometry(detector_pixel, obj_pixel, z1, z2, magnitude, energy, wavelength):

    detector_pixel_ = coord(x = detector_pixel[0], y = detector_pixel[1], z = detector_pixel[2])
    obj_pixel_      = coord(x =      obj_pixel[0], y =      obj_pixel[1], z =      obj_pixel[2])

    geometry_       = GEO(detector_pixel = detector_pixel_,
                          obj_pixel      = obj_pixel_,
                          z1             = z1,
                          z2             = z2,
                          magnitude      = magnitude, 
                          energy         = energy, 
                          wavelength     = wavelength)
    return geometry_

def contrast_param(method, beta_delta, regularization = 0.0, post_process = 0):
    return CEF(method         = method, 
               beta_delta     = beta_delta, 
               regularization = regularization, 
               post_process   = post_process)


def set_all_tripes_attributes(size, settings):
    
    if len(size) == 2:
        n_angle, n_length = size
        n_slice           = 1
    elif len(size) == 3:
        n_slice, n_angle, n_length = size
    else:
        logger.warning(f'Tomogram shape {size} needs to have size 3. Results may be wrong!')
        n_slice, n_angle, n_length = size

    size_median                     = settings['size_median'] + 1 if settings['size_median'] % 2 == 0 else settings['size_median']
    dim                             = settings['dim']
    large_stripes_size              = settings['large_stripes_size'] + 1 if settings['large_stripes_size'] % 2 == 0 else settings['large_stripes_size']
    background_ratio                = settings['background_ratio']
    drop_ratio                      = settings['drop_ratio']

    return TomoAttributes( n_length, n_angle, n_slice,
                            size_median, dim, large_stripes_size,
                            background_ratio, drop_ratio)

def rings_param(method = 'all stripes', tomosize = (1,1,1), median_s = 21, large_s = 61, 
                b_ratio = 10.0, d_ratio = 0.1, d = 1, 
                rings_block = 1, rings_lambda = -1) -> RF:
    
    methodRings = RingsMethod(method)

    all_stripes = set_all_tripes_attributes(size = tomosize, settings = {'size_median': median_s,
                                                                        'large_stripes_size': large_s,
                                                                        'background_ratio': b_ratio,
                                                                        'drop_ratio': d_ratio,
                                                                        'dim': d})
    return RF(method       = methodRings,
              all_stripes  = all_stripes,
              rings_block  = rings_block,
              rings_lambda = rings_lambda)

def align_param(method = 'rot axis', offset_method = 0, excentric_offset = 0, rotation_axis_offset = 0) -> ALGN:
    
    methodAlign = AlignMethod(method)

    return ALGN(method               = methodAlign,
                excentric_offset     = excentric_offset,
                rotation_axis_offset = rotation_axis_offset,
                offset_method        = offset_method)
    
def recon_param(geometry: GEO, method: str = 'fbp', filter: str = 'ramp', 
                filter_pad: int = 1, filter_padMode: str = 'zero',
                beta_delta: float = 0.0, rotation_axis_offset: int = 0,
                iterations: int = 0, filter_reg: float  = 1.0,
                total_variation: float = 0.0, interpolation: str = 'none') -> REC:
    
    methodRecon       = ReconMethod(method)
    filterType        = FilterNumber(filter)
    interpolationType = setInterpolation(interpolation)

    padd      = int(filter_pad)
    padd_mode = PaddMode(filter_padMode)

    if padd_mode == PaddMode('none'):
        padd = 0

    # Paganin by Slices here
    if beta_delta != 0.0:
        delta_beta = ( 1.0 / beta_delta )
        paganin_slices_regularization = geometry.wavelength * geometry.z2 * numpy.pi * delta_beta / ( geometry.detector_pixel.x * geometry.detector_pixel.x ); 
    else:
        paganin_slices_regularization = 0.0

    return REC( method               = methodRecon,
                filter               = filterType, 
                filter_reg           = filter_reg,
                filter_pad           = padd,
                filter_padMode       = padd_mode,  
                paganin_slices       = paganin_slices_regularization, 
                iterations           = iterations, 
                rotation_axis_offset = rotation_axis_offset,
                total_variation      = total_variation, 
                interpolation        = interpolationType)

def configs_param(tomo_dim, obj_dim, flat_dim, dark_dim, geometry, rings_param, contrast_param, align_param, recon_param, back_corr_method = 'classic'):
    methodBckCorr = BckCorrMethod(back_corr_method)
    
    return CFG(back_corr_method = methodBckCorr,
               geometry         = geometry,
               obj              = obj_dim,
               tomo             = tomo_dim,
               flat             = flat_dim,
               dark             = dark_dim,
               ContrastParam    = contrast_param,
               RingsParam       = rings_param,
               AlignParam       = align_param,
               ReconParam       = recon_param)

#########################

#########################
#|      ssc-raft       |#
#| Function Prototypes |#
#########################

############# Raft ##############

try:
    # Float flipx on CPU
    libraft.flip_x.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    
    libraft.flip_x.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_FLIP_X-')
    pass

try:
    # Float transpose on CPU
    libraft.transpose_cpu_zyx2xyz.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    
    libraft.transpose_cpu_zyx2xyz.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_TRANSPOSE_CPU_ZYX2XYZ-')
    pass

######## Parallel Raft ##########

try:
    # EM Ray Tracing MultiGPU without semafaro
    libraft.get_tEM_RT_MultiGPU.argtypes = [
        DIM, DIM, GEO, REC,
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p
    ]
    
    libraft.get_tEM_RT_MultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_eEMRT-')
    pass

try:
    libraft.get_eEM_RT_MultiGPU.argtypes = [
        DIM, DIM, GEO, REC,
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, 
    ]

    libraft.get_eEM_RT_MultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_tEMRT-')
    pass

try:
    # EM Frequency MultiGPU without semafaro
    libraft.get_tEM_FQ_MultiGPU.argtypes = [
        DIM, DIM, GEO, REC,
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p
    ]
        
    libraft.get_tEM_FQ_MultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_tEMFQ-')
    pass

try:
    # BST 
    libraft.getBSTMultiGPU.argtypes = [
        DIM, DIM, GEO, REC,
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int
    ]
    
    libraft.getBSTMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_BST-')
    pass

try:
    # FBP 
    libraft.getFBPMultiGPU.argtypes = [
        DIM, DIM, GEO, REC,
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
    ]
    
    libraft.getFBPMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_FBP-')
    pass

try:
    # FBP Filter only 
    libraft.getFilterLowPassMultiGPU.argtypes = [
        DIM, GEO, REC,
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, 
    ]
    
    libraft.getFilterLowPassMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_LOW_PASS-')
    pass

######## Raft - Rotation Axis Correction ##########
try:
    libraft.findcentersino_subpixel.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int
    ]

    libraft.findcentersino_subpixel.restype = ctypes.c_float    
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_CENTERSINO_SUBPIXEL-')
    pass

try:
    libraft.findcentersino.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int
    ]

    libraft.findcentersino.restype = ctypes.c_int    
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_CENTERSINO-')
    pass

try:
    libraft.correctRotationAxis.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    libraft.restype = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_CORRECTION_AXIS-')
    pass

try:
    libraft.getRotAxisCorrectionMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_float, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    libraft.restype = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_CORRECTION_AXIS_SUB_PIXEL-')
    pass

######## Raft - Rings Titarenko ##########
try:
    libraft.getTitarenkoRingsMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,  
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_float, ctypes.c_int, ctypes.c_int
    ]
    
    libraft.getTitarenkoRingsMultiGPU.restype  = None
    
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_TITARENKO_RINGS-')
    pass

######## Raft - Rings All Stripes & All Stripes Multiaxis ##########
try:
    libraft.remove_all_stripes.argtypes = [
        ctypes.c_void_p, TomoAttributes, 
        ctypes.c_int, ctypes.c_int
    ]
    
    libraft.remove_all_stripes.restype  = None
    
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_ALL_STRIPES_RINGS-')
    pass

try:
    libraft.remove_all_stripes_transposed.argtypes = [
        ctypes.c_void_p, TomoAttributes, 
        ctypes.c_int, ctypes.c_int
    ]
    
    libraft.remove_all_stripes_transposed.restype  = None
    
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_ALL_STRIPES_MULTIAXIS_RINGS-')
    pass

######## Raft - Flat/Dark Correction ##########
try:
    libraft.getBackgroundCorrectionMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int
    ]
    
    libraft.getBackgroundCorrectionMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_BACKGROUND_CORRECTION-')
    pass

######## Raft - Stitching Offset 360 ##########
try:
    libraft.getOffsetExcentricTomoGPU.argtypes = [
        ctypes.c_int, ctypes.c_void_p,  
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    
    libraft.getOffsetExcentricTomoGPU.restype  = ctypes.c_int
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_OFFSET_EXCENTRIC_TOMO-')
    pass

######## Raft - Excentric Tomography Stitch ##########
try:
    libraft.getExcentricTomoMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_int,
        ctypes.c_int
    ]
    
    libraft.getExcentricTomoMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_EXCENTRIC_TOMO_STITCH-')
    pass

######## Raft - Phase Retrieval Paganin method and similar methods ##########
try:
    libraft.getContrastEnhencementMultiGPU.argtypes = [
        DIM, GEO, CEF,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, 
    ]
    
    libraft.getContrastEnhencementMultiGPU.restype  = None
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_PHASE_RETRIEVAL-')
    pass

######## Raft - Parallel Radon ##########
try:
    libraft.getRadonRTMultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int, ctypes.c_int, 
        ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        ctypes.c_float, ctypes.c_float
    ]

    libraft.getRadonRTMultiGPU.restype  = None 
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_RADON_RT-')
    pass

######## Raft - FDK ##########
class Lab(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float),
                ("dx", ctypes.c_float), ("dy", ctypes.c_float),("dz", ctypes.c_float),
                ("nx", ctypes.c_int), ("ny", ctypes.c_int), ("nz", ctypes.c_int),
                ("h", ctypes.c_float), ("v", ctypes.c_float),
                ("dh", ctypes.c_float), ("dv", ctypes.c_float),
                ("nh", ctypes.c_int), ("nv", ctypes.c_int),
                ("D", ctypes.c_float), ("Dsd", ctypes.c_float),
                ("beta_max", ctypes.c_float),
                ("dbeta", ctypes.c_float),
                ("nbeta", ctypes.c_int),
                ("fourier", ctypes.c_int),
                ("filter_type", ctypes.c_int),
                ("reg", ctypes.c_float),
                ("is_slice", ctypes.c_int),
                ("slice_recon_start", ctypes.c_int),
                ("slice_recon_end", ctypes.c_int),
                ("slice_tomo_start", ctypes.c_int),
                ("slice_tomo_end", ctypes.c_int),
                ("nph", ctypes.c_int),
                ("padh", ctypes.c_int),
                ("energy", ctypes.c_float),
                ("rotation_axis_offset", ctypes.c_int),
                ("blocksize", ctypes.c_int),
                ("padmode", ctypes.c_int),
                ("npfh", ctypes.c_int), 
                ("padfh", ctypes.c_int), ("padfmode", ctypes.c_int)
            ]

try:
    libraft.gpu_fdk.argtypes = [
        Lab, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
    ]
    libraft.getBackgroundCorrectionMultiGPU.restype  = None
except:
    print('Cannot find C/CUDA library: -.RAFT_FDK-')
    pass

## Transmission Expectation-Maximization (tEMRT - conebeam):
class Lab_EM(ctypes.Structure):
    _fields_ = [
        ("Lx", ctypes.c_float), ("Ly", ctypes.c_float), ("Lz", ctypes.c_float),
        ("x0", ctypes.c_float), ("y0", ctypes.c_float), ("z0", ctypes.c_float),
        ("nx", ctypes.c_int), ("ny", ctypes.c_int), ("nz", ctypes.c_int),
        ("sx", ctypes.c_float), ("sy", ctypes.c_float), ("sz", ctypes.c_float),
        ("nbeta", ctypes.c_int),
        ("ndetc", ctypes.c_int),
        ("n_ray_points", ctypes.c_int)]

lib_cone_tEM  = load_library(_lib, ext)

try:
    libraft.conebeam_tEM_gpu.argtypes = [
        Lab_EM, # struct Lab lab.
        ctypes.c_void_p, # float *flat.
        ctypes.c_void_p, # float *px.
        ctypes.c_void_p, # float *py.
        ctypes.c_void_p, # float *pz.
        ctypes.c_void_p, # float *angles.
        ctypes.c_void_p, # float *recon.
        ctypes.c_void_p, # float *tomo.
        ctypes.c_int, # int ngpus.
        ctypes.c_int, # int niter.
        ctypes.c_float, # float tv.
        ctypes.c_float] # float max_val.
    libraft.conebeam_tEM_gpu.restype = ctypes.c_int
except:
    raise NotImplementedError()

######## Raft - Conebeam Radon by Ray Tracing ##########
class Lab_CB(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float),
                ("y", ctypes.c_float),
                ("z", ctypes.c_float),
                ("x0", ctypes.c_float),
                ("y0", ctypes.c_float),
                ("z0", ctypes.c_float),
                ("nx", ctypes.c_int),
                ("ny", ctypes.c_int),
                ("nz", ctypes.c_int),
                ("sx", ctypes.c_float),
                ("sy", ctypes.c_float),
                ("sz", ctypes.c_float),
                ("nbeta", ctypes.c_int),
                ("n_detector", ctypes.c_int),
                ("n_ray_points",  ctypes.c_int)]

try:
    libraft.cbradon.argtypes = [
        Lab_CB, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
        ctypes.c_int
    ]
    libraft.cbradon.restype  = ctypes.c_int
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_CONEBEAM_RADON_RT-')
    pass


try:
    libraft.cbradon_MultiGPU.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        Lab_CB, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    libraft.cbradon_MultiGPU.restype  = ctypes.c_int
except:
    logger.error(f'Cannot find C/CUDA library: -.RAFT_CONEBEAM_RADON_MULTIGPU_RT-')
    pass

######## Raft - PIPELINE C ##########
try:
    libraft.ReconstructionPipelineMultiGPU.argtypes = [
        CFG, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p
    ]
    libraft.ReconstructionPipelineMultiGPU.restype = None

except:
    logger.error('Cannot find C/CUDA library: -.RAFT_RECONSTRUCTION_PIPELINE-')

# try:
#     libraft.ReconstructionPipelineProcessMultiGPU.argtypes = [
#         CFG, ctypes.c_void_p, ctypes.c_int,
#         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
#         ctypes.c_void_p, ctypes.c_void_p
#     ]
#     libraft.ReconstructionPipelineProcessMultiGPU.restype = None

# except:
#     logger.error('Cannot find C/CUDA library: -.RAFT_RECONSTRUCTION_PIPELINE_PROCESS-')

#########################
#|      ssc-raft       |#
#|      Functions      |#
#########################

CUDA_SUCCESS = 0

def pinned_empty(shape, dtype=numpy.float32):
    size = numpy.prod(shape)
    nbytes = size * numpy.dtype(dtype).itemsize
    host_ptr = ctypes.c_void_p()
    result = libcudart.cudaMallocHost(ctypes.byref(host_ptr), ctypes.c_size_t(nbytes))
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"cudaMallocHost failed with error code {result}")

    print('HostAlloc ptr: ', host_ptr)
    array = numpy.ctypeslib.as_array(
        ctypes.cast(host_ptr,ctypes.POINTER(ctypes.c_byte)), shape=(nbytes,))
    array = array.view(dtype).reshape(shape)

    return array

def free_pinned_array(host):
    host_ptr = host.ctypes.data_as(ctypes.c_void_p)
    result = libcudart.cudaFreeHost(host_ptr)
    if result != CUDA_SUCCESS:
        raise RuntimeError(f"cudaFreeHost failed with error code {result}")

def pin_array(data):
    libcudart.cudaHostRegister(ctypes.c_void_p(data.ctypes.data),
                               ctypes.c_size_t(data.size * data.dtype.itemsize),
                               ctypes.c_uint(0))
    return data

def unpin_array(data):
    libcudart.cudaHostUnregister(ctypes.c_void_p(data.ctypes.data))

if __name__ == "__main__":
   pass

