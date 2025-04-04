from ..rafttypes import *
from .constructors import dont_process
from .constructors import apply_filter_constructor
from .pre_process import apply_log
from ..processing.opt import transpose_gpu
from ..processing.rings import *
from ..phase_retrieval.phase import *

import time

# Specific implementation of the Rings removal Titarenko
def remove_rings_titarenko(tomogram: numpy.ndarray, recon: numpy.ndarray, dic: dict) -> numpy.ndarray:
    dic['regularization'] = dic.get('rings_regularization', -1)
    dic['blocks'] = dic.get('rings_block', 1)

    tomogram = rings(tomogram, dic)

    return tomogram

remove_rings_titarenko = partial(
    apply_filter_constructor,
    process_fn=remove_rings_titarenko,
    process_name='remove rings Titarenko',
)


def paganin_by_frames(tomogram: numpy.ndarray, recon: numpy.ndarray, dic: dict) -> numpy.ndarray:
    tomogram = transpose_gpu(tomogram)
    dic['magn'] = 1
    tomogram = phase_retrieval(tomogram, dic)
    tomogram = transpose_gpu(tomogram)
    return tomogram

paganin_by_frames = partial(
    apply_filter_constructor,
    process_fn=paganin_by_frames,
    process_name='Paganin by frames',

)

paganin_methods = {
    'paganin_by_frames': paganin_by_frames,
    'paganin_by_slices': dont_process, # The paganin by slices is implemented inside the CUDA kernel in ssc-raft, when reconstructing
    'none': dont_process
}

rings_methods = {
    'titarenko': remove_rings_titarenko,
    # 'all_stripes': remove_rings_all_stripes,
    # 'all_stripes_multiaxis': remove_rings_all_stripes_multiaxis,
    'none': dont_process
}

def multiple_filters(tomogram: numpy.ndarray, recon: numpy.ndarray, dic: dict) -> numpy.ndarray:
    
    is_rings = dic.get('rings', 0)
    # Rings Correction
    if is_rings == 1:
        rings_method = dic.get('rings_method')
        if rings_method in rings_methods:
            logger.info(f"Applying rings correction method: {rings_method}")
            tomogram = rings_methods[rings_method](tomogram=tomogram, recon=recon, dic=dic)
            logger.info("Rings process completed.")
        else:
            raise ValueError(f"Unknown rings method `{rings_method}` is not implemented!")

    # Phase Retrieval
    paganin_method = dic.get('paganin_method')
    if paganin_method in paganin_methods:
        if paganin_method == 'paganin_by_frames':
            logger.info(f"Applying Paganin by frames method: {paganin_method}")
            tomogram *= -1
            tomogram = numpy.exp(tomogram, tomogram) # calculate inplace exp(-tomogram)
            tomogram = paganin_methods[paganin_method](tomogram=tomogram, recon=recon, dic=dic)
            tomogram = apply_log(tomogram, dic)
            logger.info("Paganin by frames process completed.")
        else:
            tomogram = paganin_methods[paganin_method](tomogram=tomogram, recon=recon, dic=dic)
    else:
        raise ValueError(f"Unknown phase method `{paganin_method}` is not implemented!")

    return tomogram
