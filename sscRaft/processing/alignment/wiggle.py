from curses import keyname
import os
import ctypes

from ctypes import c_int as int32
from ctypes import c_float as float32
from ctypes import POINTER
from ctypes import c_void_p  as void_p

import numpy
import sys
from ...rafttypes import *
import uuid
import SharedArray as sa
from scipy.optimize import minimize

import warnings

####
#-----------------------------
#      Shift for volume 
#

def _translate_frame( frame, cx, cy, w):
        
    wx,wy = numpy.meshgrid(w,w)
    
    z = numpy.exp(- 2*numpy.pi*1j* (wx*cy + wy*cx))

    frame_translated = numpy.fft.ifft2( numpy.fft.fft2( frame ) * numpy.fft.fftshift(z) ).real 

    return frame_translated

def _translate_frame_2( frame, cx, cy, wx, wy):
        
    wx,wy = numpy.meshgrid(wx,wy)
    
    z = numpy.exp(- 2*numpy.pi*1j* (wx*cy + wy*cx))

    frame_translated = numpy.fft.ifft2( numpy.fft.fft2( frame ) * numpy.fft.fftshift(z) ).real 

    return frame_translated


def _worker_shift_(params, idx_start,idx_end):

    volume = params[1]
    output = params[5]
    shiftx = params[3]
    shifty = params[4]
    axis   = params[0]
    
    if axis==0:
        
        dx = 2.0/(volume.shape[1]-1)
        wc = 1/(2*dx)
        wx = numpy.linspace(-wc,wc,volume.shape[2])

        dy = 2.0/(volume.shape[2]-1)
        wc = 1/(2*dy)
        wy = numpy.linspace(-wc,wc,volume.shape[1])
        
        for k in range(idx_start, idx_end):
            output[k,:,:] = _translate_frame_2( volume[k,:,:], shiftx[k], shifty[k], wx, wy)

    elif axis==1:

        dx = 2.0/(volume.shape[0]-1)
        wc = 1/(2*dx)
        wx = numpy.linspace(-wc,wc,volume.shape[0])

        dy = 2.0/(volume.shape[2]-1)
        wc = 1/(2*dy)
        wy = numpy.linspace(-wc,wc,volume.shape[2])
        
        for k in range(idx_start, idx_end):
            output[:,k,:] = _translate_frame_2( volume[:,k,:], shiftx[k], shifty[k], wx, wy)
    else:
        dx = 2.0/(volume.shape[0]-1)
        wc = 1/(2*dx)
        wx = numpy.linspace(-wc,wc,volume.shape[0])

        dy = 2.0/(volume.shape[1]-1)
        wc = 1/(2*dy)
        wy = numpy.linspace(-wc,wc,volume.shape[1])

        for k in range(idx_start, idx_end):
            output[:,:,k] = _translate_frame_2( volume[:,:,k], shiftx[k], shifty[k], wx, wy)

            

def _build_shift_volume (params):

    axis    = params[0]
    volume  = params[1]
    nproc   = params[2]
    nimages = volume.shape[axis]
    
    b = int( numpy.ceil( nimages/nproc )  ) 
    
    processes = []
    for k in range( nproc ):
        begin_ = k*b
        end_   = min( (k+1)*b, nimages )

        p = multiprocessing.Process(target=_worker_shift_, args=(params, begin_, end_ ))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
   

def set_wiggle( volume, axis, shiftx, shifty, nproc ):

    """ Set shifting values for the x,y axis at a given volume (considering a third axis fixed) 
    
    Args:
        volume: a digital volume squared matrices.
        axis: 0, 1 or 2 (axis to be fixed).
        shiftx: shifting array for x.
        shifty: shifting array for y.
        nproc: number of processes.
             
    Returns:
        (ndarray): block of sinograms with shape [nangles, size, size], with size corresponding to the
        size of each phantom. 

    * CPU function
    * This function uses a shared array through package 'SharedArray'.
    * The total number of images will be divided by the number of processes given as input
    * SharedArray names are provided by uuid.uuid4() 
    """
    
    name = str( uuid.uuid4() ) 

    if len(shiftx) != volume.shape[axis] or len(shifty) != volume.shape[axis]:
        print('ssc-radon error: len(shift) does not match with volume.shape[{}]'.format(axis))
        return None

    shape = volume.shape

    try:
        sa.delete(name)
    except:
        print('ssc-radon: creating {}x{}x{} shared arrays (tomogram)'.format(shape[0],shape[1],shape[2]) )
            
    output  = sa.create(name,shape, dtype=numpy.float32)
    
    _params_ = ( axis, volume, nproc, shiftx, shifty, output )
    
    _build_shift_volume ( _params_ )

    sa.delete(name)
    
    return output

# ---------------------------- 
#      Volume alignment 
#
# Parallel tomographic data
# Order: Theta, Y,X
#
#

def azevedo_matrix(n):

    H = numpy.zeros([3,3])

    n24 = (float(n*n)/4)
    n22 = (float(n*n)/2)
    n2  = (float(n)/2)

    pi2n = (numpy.pi/(2*n))
    t = numpy.tan( pi2n )
    t1 = (1.0/t)
    d = n2*(n22 - 1 - t1**2)
    d1  = (1.0/d)

    H[0,0] = d1*n24
    H[0,1] = -d1*n2
    H[0,2] = -d1*n2*t1
    H[1,1] = d1*( n22 - t1**2)
    H[1,2] = d1*t1
    H[2,2] = d1*(n22 - 1)
    H[1,0] = H[0,1]
    H[2,0] = H[0,2]
    H[2,1] = H[1,2]
    
    return H
    
def _get_alignment_params( tomogram ):

    nslices = tomogram.shape[1]
    nrays   = tomogram.shape[2]
    nangles = tomogram.shape[0]

    #t = numpy.linspace(-1,1,nrays).reshape([1,nrays])
    #T = numpy.kron( numpy.ones([nslices,1]), t )

    dx = 2.0/(nrays-1)
    wc = 1/(2*dx)
    w = numpy.linspace(-wc,wc,nrays)
    
    ###
    
    th = numpy.linspace(0, numpy.pi, nangles, endpoint=False).reshape([nangles, 1] )
    t = numpy.linspace(-1.0, 1.0, nrays).reshape([nrays,1])
    T = numpy.kron(t, numpy.ones([1, nangles]))
    T = numpy.flipud(T)

    a1 = numpy.ones([nangles,1])
    a2 = numpy.cos(th)
    a3 = numpy.sin(th)
 
    A = numpy.hstack([a2, a3])
 
    pinv = numpy.dot( numpy.linalg.inv( numpy.dot(A.T, A) ), A.T)

    Ah = numpy.hstack([a1, a2, a3])

    H = azevedo_matrix(nangles)
  
    pinv2 = numpy.dot( H, numpy.transpose( Ah ) )
    
    ###
    
    return (T, nslices, nrays, nangles, w, dx, A, pinv, pinv2)

def _criteria_align_frame_at_angle(x, *args):

    optp  = args[0]
    
    tomo      = optp[0]
    params    = optp[1]
    angle     = optp[2]
    currFrame = optp[3]
    wx        = optp[4]
    wy        = optp[5]

    nslices   = params[1]
    nrays     = params[2]
    nangles   = params[3]
    w         = params[4]
    
    cx = x[0]
    cy = x[1]
    alpha = x[2]
    
    newFrame = _translate_frame_2( tomo[angle,:,:], cx, cy, wx, wy)
    
    array = currFrame.sum(1) - alpha * newFrame.sum(1)
    
    fun = (array**2).sum()
    
    return fun


def _translate(y, c, t0, tf):
    #translate array y(t) to y(t-c)
 
    dt = (tf-t0)/(float(len(y)))
    wc = 1.0/(2.0*dt)
    w = numpy.linspace(-wc, wc, len(y))
    sig = numpy.exp(-2* numpy.pi * 1j * c * w)   
  
    fy = numpy.fft.fft(y)
    w = numpy.fft.ifft( fy * numpy.fft.fftshift(sig) ) 
 
    return w.real

def _get_offset_array(value, t0, tf, N):
     
    t = numpy.linspace(t0, tf, N)
    t.shape = [len(t), 1]
    dt = float(t[1] - t0)
 
    ind_zero = ( numpy.ceil( (0-t0)/dt ) )
    ind_value =  ( numpy.ceil( (value-t0)/dt ) )
 
    offset = ind_zero - ind_value
 
    return offset

def _prince(sino):
 
    # Overview: Prince's least square algorithm
    # DOI: 10.1109/83.236529

    N = sino.shape[1]
    R = sino.shape[0]
 
    th = numpy.linspace(0, 180, N, endpoint=False) * (numpy.pi/180)
    th.shape = [len(th), 1]
    t = numpy.linspace(-1.0, 1.0, R)
    t.shape = [len(t), 1]
 
    a2 = numpy.cos(th)
    a3 = numpy.sin(th)
 
    A = numpy.hstack([a2, a3])
 
    pinv = numpy.dot( numpy.linalg.inv( numpy.dot(A.T, A) ), A.T)
    #print(pinv)
    
    T = numpy.kron(t, numpy.ones([1, N]))
    T = numpy.flipud(T) #because t is flipped 

    m = sino.sum(0)
    mask = numpy.abs(m) < 1e-5
    m[mask] = 1.0
   
    w = T * sino
    b = w.sum(0)/ m
    
    b.shape = [N, 1]
 
    #z = numpy.linalg.lstsq(A, b)[0]
    z = numpy.dot(pinv, b)
     
    fit2 = numpy.dot(A,z)
    pixel = b - fit2
     
    newsino = numpy.zeros( sino.shape )
    for j in range(sino.shape[1]): 
        newsino[:,j] = _translate( sino[:,j] , pixel[j] , -1, 1)

    return pixel, newsino

def _prince_value_(sino):
 
    # Overview: Prince's least square algorithm
    # DOI: 10.1109/83.236529

    N = sino.shape[1]
    R = sino.shape[0]
 
    th = numpy.linspace(0, 180, N, endpoint=False) * (numpy.pi/180)
    th.shape = [len(th), 1]
    t = numpy.linspace(-1.0, 1.0, R)
    t.shape = [len(t), 1]
 
    a2 = numpy.cos(th)
    a3 = numpy.sin(th)
 
    A = numpy.hstack([a2, a3])
 
    pinv = numpy.dot( numpy.linalg.inv( numpy.dot(A.T, A) ), A.T)
    #print(pinv)
    
    T = numpy.kron(t, numpy.ones([1, N]))
    T = numpy.flipud(T) #because t is flipped 

    m = sino.sum(0)
    mask = numpy.abs(m) < 1e-5
    m[mask] = 1.0
   
    w = T * sino
    b = w.sum(0)/ m
    
    b.shape = [N, 1]
 
    #z = numpy.linalg.lstsq(A, b)[0]
    z = numpy.dot(pinv, b)
     
    fit2 = numpy.dot(A,z)
    pixel = b - fit2
     
    return pixel


'''
def get_alignment_p_tomogram_vaxis( tomogram, idx ):

    nslices = tomogram.shape[1]
    nrays   = tomogram.shape[2]
    nangles = tomogram.shape[0]
    
    params = _get_alignment_params( tomogram )
    frame  = tomogram[idx,:,:]
    
    _cx_ = numpy.zeros([nangles,])
    _cy_ = numpy.zeros([nangles,])

    to_exclude = [ idx ]
    index      = numpy.arange(nangles)
    index      = index[~numpy.in1d(range(len(index)),to_exclude)]
    
    for k in index:
        
        optp = (tomogram, params, k, frame) 
                
        x0 = [0,0]
        res = minimize( _criteria_align_frame_at_angle , x0, args=(optp,), method='Nelder-Mead') 
        cx, cy = res.x
        
        _cx_[k] = cx
        _cy_[k] = cy

    newtomo = shift_volume_along_axis( tomogram, 0, _cx_, _cy_, 4)
    
    return newtomo, _cx_, _cy_
'''


def _worker_p_tomogram_vaxis(params, idx_start,idx_end):

    volume = params[0]
    output = params[2]

    frame   = params[3]
    oparams = params[4]
    shift   = params[5]

    #
    dx = 2.0/(volume.shape[2]-1)
    wc = 1/(2*dx)
    wx = numpy.linspace(-wc,wc,volume.shape[2])
    
    dy = 2.0/(volume.shape[1]-1)
    wc = 1/(2*dy)
    wy = numpy.linspace(-wc,wc,volume.shape[1])
    
    for k in range(idx_start, idx_end):

        optp = (volume, oparams, k, frame, wx, wy) 
                
        x0 = [0,0,1]
        res = minimize( _criteria_align_frame_at_angle , x0, args=(optp,), method='Nelder-Mead') 
        cx, cy, alpha = res.x
        
        output[k,:,:] = alpha * _translate_frame_2( volume[k,:,:], cx, cy, wx, wy)

        shift[k, 0] = cx
        shift[k, 1] = cy


def _build_p_tomogram_vaxis (params):

    volume  = params[0]
    nproc   = params[1]
    
    nimages = volume.shape[0]
    
    b = int( numpy.ceil( nimages/nproc )  ) 
    
    processes = []
    for k in range( nproc ):
        begin_ = k*b
        end_   = min( (k+1)*b, nimages )

        p = multiprocessing.Process(target=_worker_p_tomogram_vaxis, args=(params, begin_, end_ ))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
   


def get_wiggle_p_vertical( tomogram, idx, nproc ):

    '''
    name = str( uuid.uuid4() )
    name2= str( uuid.uuid4() )

    shape = tomogram.shape

    try:
        sa.delete(name)
    except:
        pass
        #print('ssc-radon: creating {}x{}x{} shared arrays (tomogram)'.format(shape[0],shape[1],shape[2]) )
            
    output  = sa.create(name,shape, dtype=numpy.float32)

    shift  = sa.create(name2, [tomogram.shape[0], 2], dtype=numpy.float32)

    params = _get_alignment_params( tomogram )
    frame  = tomogram[idx,:,:]
    
    _params_ = ( tomogram, nproc, output, frame, params, shift )
    
    _build_p_tomogram_vaxis( _params_ )

    sa.delete(name)
    sa.delete(name2)
    
    return output, shift
    '''

    nth,ny,nx = tomogram.shape

    _y_ = numpy.linspace(-1,1,ny)  
    _x_ = numpy.linspace(-1,1,nx) 
    x,y = numpy.meshgrid(_x_, _y_)
    y = numpy.flipud(y)
    
    def operators(tomogram, theta, y):
        M = (y * tomogram[theta, :, :]).sum()
        N = (tomogram[theta, :, :]).sum()
        #D = numpy.diff( tomogram, n=1, axis=1, prepend=tomogram[theta,0,0])[theta, :, :].sum()
        return M, N
    
    h = numpy.zeros([nth,])

    _M_,_N_ = operators(tomogram, idx, y)

    for j in range(nth):
        M,N  = operators(tomogram, j, y)
        if abs( _N_ ) < 1e-5 or abs( N ) < 1e-5:
            h[j] = 0
        else:
            h[j] = - _M_/_N_ + M/N 
    

    '''
    _M_,_N_,_ = operators(tomogram, idx, y)

    for j in range(nth):
        M,N,D  = operators(tomogram, j, y)
        if abs(D) < 1e-5:
            h[j] = 0
        else:
            h[j] = (N - _N_) / D 
    '''
    
    output = set_wiggle(tomogram, 0, h, numpy.zeros(h.shape), nproc)

    return output, h


def _worker_p_tomogram_haxis(params, idx_start,idx_end, ptype):

    #_params_ = ( tomogram, nproc, output, params, shift )

    
    volume = params[0]
    output = params[2]
    shift  = params[4]
    cmass  = params[5]
    oparams = params[3]
    
    T, nslices, nrays, nangles, _, _, A, pinv, pinv2 = oparams

    if ptype == "slices":
    
        for k in range(idx_start, idx_end):

            sino = volume[:,k,:].T
            
            m = sino.sum(0)
            mask = numpy.abs(m) < 1e-5
            m[mask] = 1.0
            
            w = T * sino
            b = (w.sum(0)/m).reshape([nangles,1])
            
            c = numpy.dot(pinv, b)
            
            fit = numpy.dot(A,c)
            _shift_ = (b - fit/m.reshape([nangles,1]) )
            
            shift[k,:] = _shift_.flatten()
            
            newsino = numpy.zeros( sino.shape )
            for j in range( nangles ): 
                newsino[:,j] = _translate( sino[:,j] , shift[k,j], -1, 1)
        
            output[:,k,:] = newsino.T

            #center of mass using Azevedo's matrix
            _c_ = numpy.dot(pinv2, b)
            
            cmass[k,0] = _c_[1,0]
            cmass[k,1] = _c_[2,0]

    elif ptype == "mean":
        
        for k in range(idx_start, idx_end):

            sino = volume[:,k,:].T
                        
            newsino = numpy.zeros( sino.shape )
            for j in range( nangles ): 
                newsino[:,j] = _translate( sino[:,j] , shift.mean(1)[j], -1, 1)
        
            output[:,k,:] = newsino.T

            

def _build_p_tomogram_haxis (params, ptype):
            
    volume  = params[0]
    nproc   = params[1]
    shift   = params[4]
    cmass   = params[5]
    
    oparams = params[3]

    T, nslices, nrays, nangles, _, _, A, pinv, pinv2 = oparams
        
    if ptype == "mean":

        for k in range(nslices):
            
            sino = volume[:,k,:].T

            m = sino.sum(0)
            mask = numpy.abs(m) < 1e-5
            m[mask] = 1.0

            w = T * sino
            b = (w.sum(0)/m).reshape([nangles,1])

            c = numpy.dot(pinv, b)

            fit = numpy.dot(A,c)
            _shift_ = ( b - fit/m.reshape([nangles,1]) )
        
            shift[k,:] = _shift_.flatten()

            #center of mass using Azevedo's matrix
            c = numpy.dot(pinv2, b)
            
            cmass[k,0] = c[1,0]
            cmass[k,1] = c[2,0]

    ##
    b = int( numpy.ceil( nslices/nproc )  ) 
            
    processes = []
    for k in range( nproc ):
        begin_ = k*b
        end_   = min( (k+1)*b, nslices )

        p = multiprocessing.Process(target=_worker_p_tomogram_haxis, args=(params, begin_, end_, ptype ))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()


def get_wiggle_p_horizontal_slices( tomogram , nproc):

    name = str( uuid.uuid4() )
    name2= str( uuid.uuid4() )
    name3= str( uuid.uuid4() )

    shape = tomogram.shape

    try:
        sa.delete(name)
    except:
        #pass
        print('ssc-radon: creating {}x{}x{} shared arrays (tomogram)'.format(shape[0],shape[1],shape[2]) )
            
    output  = sa.create(name,shape, dtype=numpy.float32)

    shift  = sa.create(name2, [tomogram.shape[1], tomogram.shape[0] ], dtype=numpy.float32)

    cmass  = sa.create(name3, [tomogram.shape[1], 2 ], dtype=numpy.float32)
    
    params = _get_alignment_params( tomogram )
     
    _params_ = ( tomogram, nproc, output, params, shift, cmass )
    
    _build_p_tomogram_haxis( _params_, "slices")

    sa.delete(name)
    sa.delete(name2)
    sa.delete(name3)
    
    return output, shift, cmass
    
    
def get_wiggle_p_horizontal_mean( tomogram, nproc ):

    name = str( uuid.uuid4() )
    name2= str( uuid.uuid4() )
    name3= str( uuid.uuid4() )
    shape = tomogram.shape

    try:
        sa.delete(name)
    except:
        #pass
        print('ssc-radon: creating {}x{}x{} shared arrays (tomogram)'.format(shape[0],shape[1],shape[2]) )
            
    output  = sa.create(name,shape, dtype=numpy.float32)
    
    shift  = sa.create(name2, [tomogram.shape[1], tomogram.shape[0] ], dtype=numpy.float32)

    cmass  = sa.create(name3, [tomogram.shape[1], 2 ], dtype=numpy.float32)
    
    params = _get_alignment_params( tomogram )
     
    _params_ = ( tomogram, nproc, output, params, shift, cmass )
    
    _build_p_tomogram_haxis( _params_, "mean" )

    sa.delete(name)
    sa.delete(name2)
    sa.delete(name3)
    
    return output, shift, cmass
    

def get_wiggle( tomogram, direction, nproc, idx, *args ):

    """ Get horizontal shifting values for the x,y axis at a given tomogram volume obtained from parallel geometry
    
    Args:
        tomogram: a digital tomogram with shape [nangles, size, size] with size, the detector width/heigth.
        direction: string ``horizontal`` or ``vertical``
        nproc: number of processes.
        idx: a given index (from angle axis), to be used as a reference.

    Returns:
        (ndarray): tomogram with shape [nangles, size, size], with size corresponding to the
        size of each phantom. 

    * CPU function
    * This function uses a shared array through package 'SharedArray'.
    * The total number of images will be divided by the number of processes given as input
    * SharedArray names are provided by uuid.uuid4() 
    
    """

    if not args:
        function_horizontal = get_wiggle_p_horizontal_slices
    else:
        g = args[0]
        if g == "mean":
            function_horizontal = get_wiggle_p_horizontal_mean
        elif g=="slices":
            function_horizontal = get_wiggle_p_horizontal_slices
        else:
            print('ssc-radon: Error! Check extra arguments ... "mean"?')

    if direction=="horizontal":    
        return function_horizontal( tomogram, nproc )
         
    elif direction=="vertical":

        return get_wiggle_p_vertical( tomogram, idx, nproc )
        
    else:
        print('ssc-radon: Error wrong movement direction string!')

    
        
        
