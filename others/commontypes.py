import inspect
import ctypes
import atexit
import os
import numpy as np

def load_library(lib, ext):
  _path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + lib + ext
  try:
    lib = ctypes.CDLL(_path)
    return lib
  except:
    pass
  return None


def on_shutdown():
    libcommons.ssc_log_stop()

# import pdb; pdb.set_trace()
libcommons = load_library("lib/libssccommons", ".so")

#########################

##########################
#|      ssc-raft      |#
#| Function prototypes  |#
##########################

proj_name = "sscRaft"
log_level = "error"
telem_key = "https://0486f9be2e157d5e4dd2bbb9a17353da@o1066143.ingest.sentry.io/4506825999712256"
from ..sscRaft import __version__

# Define the ssc_param_tag_t enum
class ssc_param_tag_t(ctypes.c_int):
    INT = 0
    DOUBLE = 1
    STRING = 2

# Define the param_val_t union
class param_val_t(ctypes.Union):
    _fields_ = [("i", ctypes.c_int),
                ("d", ctypes.c_double),
                ("s", ctypes.c_char_p)]

# Define the ssc_param_t struct
class ssc_param_t(ctypes.Structure):
    _fields_ = [("tag", ssc_param_tag_t),
                ("key", ctypes.c_char_p),
                ("val", param_val_t)]

try:
  libcommons.ssc_log_start.argtypes = [
      ctypes.c_char_p,
      ctypes.c_char_p,
      ctypes.c_char_p,
      ctypes.c_char_p,
  ]
  libcommons.ssc_log_start.restype = None

  libcommons.ssc_log_start(
      ctypes.c_char_p(proj_name.encode()),
      ctypes.c_char_p(__version__.encode()),
      ctypes.c_char_p(log_level.encode()),
      ctypes.c_char_p(telem_key.encode()))

  libcommons.ssc_param_int.restype = ssc_param_t
  libcommons.ssc_param_int.argtypes = [
      ctypes.c_char_p,
      ctypes.c_int
  ]

  libcommons.ssc_param_double.restype = ssc_param_t
  libcommons.ssc_param_double.argtypes = [
      ctypes.c_char_p,
      ctypes.c_double
  ]

  libcommons.ssc_param_string.restype = ssc_param_t
  libcommons.ssc_param_string.argtypes = [
      ctypes.c_char_p,
      ctypes.c_char_p
  ]

  libcommons.ssc_event_start.argtypes = [
      ctypes.c_char_p,
      ctypes.c_int,
      ctypes.POINTER(ssc_param_t)
  ]

  libcommons.ssc_event_stop.argtypes = []

  atexit.register(on_shutdown)
except:
    import logging
    logging.error("Error loading commons")

def log_event(func):
    def wrapper(*args, **kwargs):
        # Get the names of the parameters and their default values
        params = inspect.signature(func).parameters
        param_info = [(name, param.default) for name, param in params.items()]

        # Merge positional and keyword arguments
        all_args = dict(zip(params.keys(), args))
        all_args.update(kwargs)

        evt_params = { name: all_args.get(name, default) for name, default in param_info }
        print(evt_params)

        event_start(func.__name__, evt_params)
        func_ret = func(*args, **kwargs)
        event_stop()

        return func_ret
    return wrapper

def evt_param(key, v):
    c_key = ctypes.c_char_p(key.encode())
    if isinstance(v, int):
        return libcommons.ssc_param_int(c_key, ctypes.c_int(v))
    elif isinstance(v, float):
        return libcommons.ssc_param_double(c_key, ctypes.c_double(v))
    elif isinstance(v, str):
        return libcommons.ssc_param_string(c_key, ctypes.c_char_p(v.encode()))
    elif isinstance(v, np.ndarray):
        array_info = "{} {}".format(v.shape, v.dtype)
        return libcommons.ssc_param_string(c_key, ctypes.c_char_p(array_info.encode()))
    else: #default to string representation
        str_val = str(v)
        return libcommons.ssc_param_string(c_key, ctypes.c_char_p(str_val.encode()))

def event_start(event: str, params: dict):

    params_array = [evt_param(k, v) for k, v in params.items()]
    n = len(params_array)

    libcommons.ssc_event_start(ctypes.c_char_p(event.encode()),
                            ctypes.c_int(n),
                           (ssc_param_t*n)(*params_array))

def event_stop():
    libcommons.ssc_event_stop()

