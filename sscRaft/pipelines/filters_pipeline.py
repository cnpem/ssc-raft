from ..rafttypes import *
from .constructors import dont_process
from .constructors import apply_filter_constructor
from .pre_process import apply_log
from ..processing.opt import transpose
from ..processing.rings import *
from ..phase_retrieval.phase import *

try:
    import sscRings

    # Specific implementation of the Rings All Stripes removal of `ssc-rings`
    def remove_rings_all_stripes(tomogram: numpy.ndarray, dic: dict) -> numpy.ndarray:
        rings_parameters = dic.get('rings_parameters')
        number_of_gpus = len(dic.get('gpu', [0]))

        snr     = rings_parameters[0]
        la_size = rings_parameters[1]
        sm_size = rings_parameters[2]
        dim     = rings_parameters[3]

        tomogram = sscRings.rings_remove_all_stripes(tomogram, number_of_gpus,
                                                    settings = {'size_median':sm_size, 
                                                        'large_stripes_size':la_size, 
                                                        'background_ratio':snr, 
                                                        'drop_ratio':0.1, 
                                                        'dim':dim})

        return tomogram

    remove_rings_all_stripes = partial(
        apply_filter_constructor,
        process_fn=remove_rings_all_stripes,
        process_name='remove rings all stripes',
        save_prefix='remove_rings_all_stripes',
        should_save_key='save_rings'
    )

    # Specific implementation of the Rings All Stripes removal of `ssc-rings` for multiaxis
    def remove_rings_all_stripes_multiaxis(tomogram: numpy.ndarray, dic: dict) -> numpy.ndarray:
        rings_parameters = dic.get('rings_parameters')
        number_of_gpus = len(dic.get('gpu', [0]))

        snr     = rings_parameters[0]
        la_size = rings_parameters[1]
        sm_size = rings_parameters[2]
        dim     = rings_parameters[3]

        tomogram = sscRings.rings_remove_all_stripes(tomogram, 
                                            number_of_gpus,
                                            settings = {'size_median':sm_size, 
                                                        'large_stripes_size':la_size, 
                                                        'background_ratio':snr, 
                                                        'drop_ratio':0.1, 
                                                        'dim':dim})

        tomogram = tomogram.transpose(2, 1, 0)
        tomogram = sscRings.rings_remove_all_stripes(tomogram, 
                                            number_of_gpus,
                                            settings = {'size_median':sm_size, 
                                                        'large_stripes_size':la_size, 
                                                        'background_ratio':snr, 
                                                        'drop_ratio':0.1, 
                                                        'dim':dim})

        tomogram = tomogram.transpose(2, 1, 0)

        return tomogram

    remove_rings_all_stripes_multiaxis = partial(
        apply_filter_constructor,
        process_fn=remove_rings_all_stripes_multiaxis,
        process_name='remove rings all stripes multiaxis',
        save_prefix='remove_rings_all_stripes_multiaxis',
        should_save_key='save_rings'
    )
except KeyError as e:
    logger.error(f"Cannot find package sscRings: {str(e)}. Rings method `all_stripes` and `all_stripes_multiaxis` cannot be loaded.")

# Specific implementation of the Rings removal Titarenko
def remove_rings_titarenko(tomogram: numpy.ndarray, dic: dict) -> numpy.ndarray:
    dic['regularization'] = dic.get('rings_regularization_raft', -1)
    dic['blocks'] = dic.get('rings_block_raft', 1)

    tomogram = rings(tomogram, dic)

    return tomogram

remove_rings_titarenko = partial(
    apply_filter_constructor,
    process_fn=remove_rings_titarenko,
    process_name='remove rings Titarenko',
    save_prefix='remove_rings_titarenko',
    should_save_key='save_rings'
)


def paganin_by_frames(tomogram: numpy.ndarray, dic: dict) -> numpy.ndarray:
    tomogram = transpose(tomogram)
    # dic['detectorPixel[m]'] = dic['detector_pixel[m]']
    # dic['energy[eV]'] = dic['beam_energy']
    dic['magn'] = 1 + (dic['z2[m]']/dic['z1[m]'])
    # dic['beta/delta'] = dic['paganin_filter_beta_delta']
    tomogram = phase_retrieval(tomogram, dic)
    tomogram = transpose(tomogram)
    return tomogram

paganin_by_frames = partial(
    apply_filter_constructor,
    process_fn=paganin_by_frames,
    process_name='Paganin by frames',
    save_prefix='paganin_by_frames',
    should_save_key='save_paganin_filter'
)

paganin_methods = {
    'paganin_by_frames': paganin_by_frames,
    'paganin_by_slices': dont_process, # The paganin by slices is implemented inside the CUDA kernel in ssc-raft, when reconstructing
    'none': dont_process
}

rings_methods = {
    'titarenko': remove_rings_titarenko,
    'all_stripes': remove_rings_all_stripes,
    'all_stripes_multiaxis': remove_rings_all_stripes_multiaxis,
    'none': dont_process
}

def multiple_filters(tomogram: numpy.ndarray, dic: dict) -> numpy.ndarray:

    # Phase Retrieval
    paganin_method = dic.get('paganin_method')
    if paganin_method in paganin_methods:
        if paganin_method == 'paganin_by_frames':
            logger.info(f"Applying Paganin by frames method: {paganin_method}")
            tomogram = numpy.exp(-tomogram, tomogram) # calculate inplace exp(-tomogram)
            tomogram = paganin_methods[paganin_method](tomogram=tomogram, dic=dic)
            tomogram = apply_log(tomogram, dic)
            logger.info("Paganin by frames process completed.")
        else:
            tomogram = paganin_methods[paganin_method](tomogram=tomogram, dic=dic)
    else:
        raise ValueError(f"Unknown phase method `{paganin_method}` is not implemented!")
    
    # Rings Correction
    rings_method = dic.get('rings_method', 'titarenko')
    if rings_method in rings_methods:
        logger.info(f"Applying rings correction method: {rings_method}")
        tomogram = rings_methods[rings_method](tomogram=tomogram, dic=dic)
        logger.info("Rings process completed.")
    else:
        raise ValueError(f"Unknown rings method `{rings_method}` is not implemented!")

    return tomogram
