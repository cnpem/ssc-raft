from ._version import __version__
import logging
import os
import ctypes
from shutil import which

# def _set_LD_LIBRARY_PATH_cuda():
#     nvcc = which("nvcc")

#     if nvcc:
#         cuda_root = os.path.abspath(os.path.join(nvcc, "..", ".."))
#         lib_path = os.path.join(cuda_root, "lib64")
#     else:
#         lib_path = None

#     print(f"Set CUDA LD_LIBRARY_PATH={lib_path} (__init__.py)")

#     if lib_path:
#         os.putenv("LD_LIBRARY_PATH", lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "") )
#         os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")
#     else:
#         raise RuntimeError(
#             "CUDA runtime not found. Load a CUDA module or install CUDA."
#         )
# try:
#     ctypes.CDLL( ctypes.util.find_library( "libcudart.so" ), mode=ctypes.RTLD_GLOBAL)
# except:
#     print('here2')
#     _set_LD_LIBRARY_PATH_cuda()

# try:
from .rafttypes   import *
from .geometries  import *
from .processing  import *
from .phase_retrieval  import *
from .pipelines.structured   import *
from .io   import *

# except OSError:
#     logging.error("Could not load sscRaft shared libraries (__init__.py)")
