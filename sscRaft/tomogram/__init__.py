from .flatdark import *
from .rotationaxis import *
from .alignment import *

def set_conical_slices(slice_recon_start,slice_recon_end,
                                  nslices,nx,ny,
                                  z1,z12,pixel_det):
        
        magn       = z1/z12
        v          = nslices * pixel_det / 2
        dx, dy, dz = pixel_det*magn, pixel_det*magn, pixel_det*magn
        x, y, z    = dx*nx/2, dy*ny/2, dz*nslices/2
        L          = numpy.max(x,y)


        z_min_recon = - z + slice_recon_start * dz
        z_max_recon = - z + slice_recon_end   * dz

        Z_min_proj = numpy.max(-v, numpy.min(z12*z_min_recon/(z1 - L), z12*z_min_recon/(z1 + L) ))
        Z_max_proj = numpy.min( v, numpy.max(z12*z_max_recon/(z1 + L), z12*z_max_recon/(z1 - L)))

        slice_projection_start = numpy.max(0      , int(numpy.floor((Z_min_proj + v)/pixel_det)))
        slice_projection_end   = numpy.min(nslices, int(numpy.ceil( (Z_max_proj + v)/pixel_det)))

        return slice_projection_start, slice_projection_end

def set_conical_tomogram_slices(tomogram, dic):

    nrays   = tomogram.shape[-1]
    nslices = tomogram.shape[-2]    
    nangles = tomogram.shape[ 0]

    z1, z12   = dic['z1[m]'], dic['z1+z2[m]']
    pixel_det = dic['detectorPixel[m]']
    
    start_recon_slice = dic['slices'][0]
    end_recon_slice   = dic['slices'][1] 
    
    blockslices = end_recon_slice - start_recon_slice

    try:
        reconsize = dic['reconSize']

        if isinstance(reconsize,list) or isinstance(reconsize,tuple):
        
            if len(dic['reconSize']) == 1:
                nx, ny, nz = int(reconsize[0]), int(reconsize[0]), int(blockslices)
            else:
                nx, ny, nz = int(reconsize[0]), int(reconsize[1]), int(blockslices)

        elif isinstance(reconsize,list):
            nx, ny, nz = int(reconsize), int(reconsize), int(blockslices)
        else:
            logger.error(f'Dictionary entry `reconsize` wrong ({reconsize}). The entry `reconsize` is optional, but if it exists it needs to be a list = [nx,ny,nz].')
            logger.error(f'Finishing run...')
            sys.exit(1)

    except:
        nx, ny, nz = int(nrays), int(nrays), int(blockslices)
        logger.info(f'Set default reconstruction size ({nz},{ny},{nx}) = (z,y,x).')

    
    start_tomo_slice,end_tomo_slice = set_conical_slices(start_recon_slice,end_recon_slice,nslices,nx,ny,z1,z12,pixel_det)

    _tomo_ = tomogram[start_tomo_slice:(end_tomo_slice + 1),:,:]

    dic.update({'slice tomo': [start_tomo_slice,end_tomo_slice]})

    return _tomo_