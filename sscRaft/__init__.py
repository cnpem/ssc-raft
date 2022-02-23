# -*- coding: utf-8 -*-
try:
    import pkg_resources 
    __version__ = pkg_resources.require("raft")[0].version
except:
    pass

from .rafttypes import *
from .parallel import *
