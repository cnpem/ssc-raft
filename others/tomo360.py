from ast import arg
import os

import numpy 
import sys
import gc
from ....rafttypes import *
import uuid
import SharedArray as sa
from scipy.optimize import minimize
from scipy.optimize import brute
from scipy.optimize import minimize_scalar
# from sscOldRaft import *

def _translate(frame, cx, cy, wx, wy):
        
    wx,wy = numpy.meshgrid(wx,wy)
    
    z = numpy.exp(-2 * numpy.pi*1j * (wx*cy + wy*cx))

    frame_translated = numpy.fft.ifft2( numpy.fft.fft2( frame ) * numpy.fft.fftshift(z) ).real 

    return frame_translated

def _criteria_find_tomo360_alignment_fourier(x, *args):

    optp   = args[0]
    tomoL  = optp[0]
    tomoR  = optp[1]
    slices = optp[2]
    dt     = optp[3]
    tomo   = optp[4]
    
    M1 = x

    wc = 1/(2*dt)
    wx = numpy.linspace(-wc,wc,tomoL.shape[2])
    
    dy = numpy.pi/(tomoL.shape[1]-1)
    wc = 1/(2*dy)
    wy = numpy.linspace(-wc,wc,tomoL.shape[1])

    vL = numpy.ones((tomoL.shape[1],tomoL.shape[2]))
    vR = numpy.ones((tomoL.shape[1],tomoL.shape[2]))
    
    atL = numpy.linspace(0,1,2*int(M1/dt))
    atR = numpy.linspace(0,1,2*int(M1/dt))

    nimages = tomoL.shape[2] // 2
    print('olha ela:', int(M1/dt),nimages, nimages - 2*int(M1/dt), nimages + 2*int(M1/dt) )
    if int(M1/dt) != 0.0:
        for i in range(tomoL.shape[1]):
            vL[i,nimages - 2*int(M1/dt):nimages] = numpy.flip(atL)
            vR[i,nimages:nimages + 2*int(M1/dt)] = atR

        print(atL.shape)
        newFrameL  = _translate(vL*tomoL[slices,:,:]+0, 0,  M1, wx, wy)
        newFrameR  = _translate(vR*tomoR[slices,:,:]+0, 0, -M1, wx, wy)
    else:
        newFrameL  = _translate(tomoL[slices,:,:]+0, 0,  M1, wx, wy)
        newFrameR  = _translate(tomoR[slices,:,:]+0, 0, -M1, wx, wy)

    for i in range(0,int(M1/dt)+1):
        newFrameL[:,i] = newFrameL[:,int(M1/dt)+1]
        newFrameR[:,-i] = newFrameR[:,-(int(M1/dt)+1)]

    sumFrameL = newFrameL.sum(1)
    sumFrameR = newFrameR.sum(1)
    sumFrame  = sumFrameL + sumFrameR
    
    gradFrame  = numpy.gradient(dt * sumFrame)
    
    array = gradFrame
    
    fun = (array**2).sum()

    print(fun, M1)
    
    return fun

def _criteria_find_tomo360_alignment_fouriercrop(x, *args):

    optp   = args[0]
    tomoL  = optp[0]
    tomoR  = optp[1]
    slices = optp[2]
    dt     = optp[3]
    tomo   = optp[4]
    
    nimages = tomoL.shape[2] // 2

    M1 = x
    # M2 = x[1]
    
    # tomoT = Tomo360To180(tomo,-int(M1/dt),device=-1)
    # sumFrame = tomoT[slices,:,:].sum(1)

    wc = 1/(2*dt)
    wx = numpy.linspace(-wc,wc,tomoL.shape[2])
    
    dy = numpy.pi/(tomoL.shape[1]-1)
    wc = 1/(2*dy)
    wy = numpy.linspace(-wc,wc,tomoL.shape[1])

    newFrameL  = _translate(tomoL[slices,:,:]+0, 0,  M1, wx, wy)
    newFrameR  = _translate(tomoR[slices,:,:]+0, 0, -M1, wx, wy)

    newFrameL = newFrameL[:,int(M1/dt):nimages+int(M1/dt)]
    newFrameR = newFrameR[:,nimages-int(M1/dt):2*nimages-int(M1/dt)]

    sumFrameL = newFrameL.sum(1)
    sumFrameR = newFrameR.sum(1)
    sumFrame  = sumFrameL + sumFrameR

    gradFrame  = numpy.gradient(dt * sumFrame)
    
    array = gradFrame
    
    fun = (array**2).sum()

    print(fun, M1)
    
    return fun

def _criteria_find_tomo360_alignment(x, *args):

    optp  = args[0]
    tomoL = optp[0]
    tomoR = optp[1]
    slices = optp[2]
    dt    = optp[3]
    
    M1 = x[0]
    M2 = x[1]

    wc = 1/(2*dt)
    wx = numpy.linspace(-wc,wc,tomoL.shape[2])
    
    dy = numpy.pi/(tomoL.shape[1]-1)
    wc = 1/(2*dy)
    wy = numpy.linspace(-wc,wc,tomoL.shape[1])

    newFrameL  = _translate(tomoL[slices,:,:]+0, 0,  M1, wx, wy)
    newFrameR  = _translate(tomoR[slices,:,:]+0, 0, -M2, wx, wy)

    newFrameL[:,:int(M1/dt)+1] = 0
    newFrameR[:,:-(int(M1/dt)+1)] = 0

    sumFrameL = newFrameL.sum(1)
    sumFrameR = newFrameR.sum(1)
    sumFrame  = sumFrameL + sumFrameR
    
    gradFrame  = numpy.gradient(dt * sumFrame)
    
    array = gradFrame
    
    fun = (array**2).sum()

    print(fun, M1, M2)
    
    return fun

def _criteria_find_tomo360_alignment_2(x, *args):

    optp   = args[0]
    tomoL  = optp[0]
    tomoR  = optp[1]
    slices = optp[2]
    dt     = optp[3]
    
    M1 = int(x[0]/dt)
    M2 = int(x[1]/dt)
    
    if M1 == 0:
        newFrameL  = numpy.copy(tomoL[slices,:,:])
    else:
        newFrameL  = numpy.copy(tomoL[slices,:,:-M1])
    if M2 == 0:
        newFrameR  = numpy.copy(tomoR[slices,:,:])
    else:
        newFrameR  = numpy.copy(tomoR[slices,:,M2:])

    sumFrameL = newFrameL.sum(1)
    sumFrameR = newFrameR.sum(1)
    sumFrame  = sumFrameL + sumFrameR
    
    gradFrame  = numpy.gradient(dt * sumFrame)
    
    array = gradFrame
    
    fun = (array**2).sum()

    print(fun, x, M1, M2)
    
    return fun

def _criteria_find_tomo360_alignment_single(x, *args):

    optp   = args[0]
    tomoL  = optp[0]
    tomoR  = optp[1]
    slices = optp[2]
    dt     = optp[3]
    
    M = int(x/dt)

    if M == 0:
        newFrameL  = numpy.copy(tomoL[slices,:,:])
        newFrameR  = numpy.copy(tomoR[slices,:,:])
    else:
        newFrameL  = numpy.copy(tomoL[slices,:,:-M])
        newFrameR  = numpy.copy(tomoR[slices,:,M:])

    sumFrameL = newFrameL.sum(1)
    sumFrameR = newFrameR.sum(1)
    sumFrame  = sumFrameL + sumFrameR
    
    gradFrame  = numpy.gradient(dt * sumFrame)
    
    array = gradFrame
    
    fun = (array**2).sum()

    print(fun, M)
    
    return fun


def _worker_tomo360_align(args, idx_start, idx_end):

    volumeL = args[0]
    volumeR = args[1]
    output  = args[2]
    tomo    = args[4]

    dt    = 1.0/(volumeL.shape[-1]-1)
    bound = ((0,0.3),(0,0.3))
    # bound = (0,0.5)

    for slices in range(idx_start, idx_end):

        optp = (volumeL, volumeR, slices, dt, tomo)

        x0 = (0,0.3)
        
        res = minimize(_criteria_find_tomo360_alignment, x0, args=(optp,), bounds = bound, method='SLSQP')
        # res = minimize_scalar(_criteria_find_tomo360_alignment_single, args=(optp,), bounds = bound, method='Bounded') 
        M = res.x
        
        output[slices,:] = M


def _build_tomo360_align(args):

    volumeL = args[0]
    nproc   = args[3]
    
    nslices = volumeL.shape[0]
    
    b = int( numpy.ceil( nslices/nproc )  ) 
    
    processes = []
    for k in range( nproc ):
        begin_ = k*b
        end_   = min( (k+1)*b, nslices )

        p = multiprocessing.Process(target=_worker_tomo360_align, args=(args, begin_, end_ ))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
   


def get_tomo360_align(tomo, tomogramLeft, tomogramRight, nproc):
    
    name  = str( uuid.uuid4() )
    shape = (tomogramLeft.shape[0],2)

    try:
        sa.delete(name)
    except:
        pass
            
    output  = sa.create(name, shape, dtype=numpy.float32)

    _params_ = (tomogramLeft, tomogramRight, output, nproc, tomo)
    
    _build_tomo360_align( _params_ )

    sa.delete(name)
    
    return output

def _worker_tomo360_align2(args, idx_start, idx_end):

    volumeL = args[0]
    volumeR = args[1]
    output  = args[2]
    tomo    = args[4]

    dt    = 2.0/(volumeL.shape[-1]-1)
    # bound = ((0,0.5),(0,.05))
    bound = (0,0.5)

    for slices in range(idx_start, idx_end):

        optp = (volumeL, volumeR, slices, dt, tomo)

        x0 = 0
        
        # res = minimize(_criteria_find_tomo360_alignment, x0, args=(optp,), bounds = bound, method='Nelder-Mead')
        res = minimize_scalar(_criteria_find_tomo360_alignment_fourier, args=(optp,), bounds = bound, method='Bounded') 
        M = res.x
        
        output[slices,:] = (M,0)


def _build_tomo360_align2(args):

    volumeL = args[0]
    nproc   = args[3]
    
    nslices = volumeL.shape[0]
    
    b = int( numpy.ceil( nslices/nproc )  ) 
    
    processes = []
    for k in range( nproc ):
        begin_ = k*b
        end_   = min( (k+1)*b, nslices )

        p = multiprocessing.Process(target=_worker_tomo360_align2, args=(args, begin_, end_ ))
        processes.append(p)
    
    for p in processes:
        p.start()

    for p in processes:
        p.join()
   


def get_tomo360_align2(tomo, tomogramLeft, tomogramRight, nproc):
    
    name  = str( uuid.uuid4() )
    shape = (tomogramLeft.shape[0],2)

    try:
        sa.delete(name)
    except:
        pass
            
    output  = sa.create(name, shape, dtype=numpy.float32)

    _params_ = (tomogramLeft, tomogramRight, output, nproc, tomo)
    
    _build_tomo360_align2( _params_ )

    sa.delete(name)
    
    return output

def _find_tomo360_alignment_single(*optp):

    tomoL  = optp[0]
    tomoR  = optp[1]
    slices = optp[2]
    dt     = optp[3]
    
    tol = 1e-10
    M1 = 0
    M2 = 0
    sum = 100
    # i = -1
    # while ( i < (tomoL.shape[2] // 2) ):
    for i in range(0,tomoL.shape[2] // 3):
        for j in range(0,tomoL.shape[2] // 3):
            # i =+ 1
            M1 = i 
            M2 = j
            
            if M1 == 0:
                newFrameL  = numpy.copy(tomoL[slices,:,:])
            else:
                newFrameL  = numpy.copy(tomoL[slices,:,:-M1])
            if M2 == 0:
                newFrameR  = numpy.copy(tomoR[slices,:,:])
            else:
                newFrameR  = numpy.copy(tomoR[slices,:,M2:])

            sumFrameL = newFrameL.sum(1)
            sumFrameR = newFrameR.sum(1)
            sumFrame  = sumFrameL + sumFrameR
            
            gradFrame  = numpy.gradient(dt * sumFrame)
            
            array = gradFrame
            
            fun = (array**2).sum()

            print(fun, M1, M2)

            res = abs(fun-sum)
            print(res)

            if fun < sum:
                sum = fun

            if res < tol:
                break
        
        if res < tol:
                break

    return newFrameL, newFrameR

def tomo_360_to_180(volume, nproc):

    nslices = volume.shape[0]
    nangles2pi = volume.shape[1]
    nimages = volume.shape[2]

    volume = volume/numpy.max(volume)

    if nangles2pi % 2 == 0:
        nangles = nangles2pi // 2
        volumeL = volume[:,:nangles,:]
        volumeR = volume[:,nangles:,:]
    else:
        nangles = (nangles2pi - 1) // 2 
        volumeL = volume[:,:nangles,:]
        volumeR = volume[:,nangles:-1,:]

    volumeR = numpy.flip(volumeR,axis=2)

    # tomo = numpy.concatenate((volumeL,volumeR), axis=2)

    na = nangles // 2

    lineL = volumeL[0,na,:]
    lineR = volumeR[0,na,:]
    # numpy.save('tomocurveL.npy',lineL)
    # numpy.save('tomocurveR.npy',lineR)

    dt = 1.0/(volumeL.shape[2]-1)

    alignment = get_tomo360_align(volume, volumeL, volumeR, nproc) 

    print('Alignment:', alignment)
    align = int(numpy.mean(alignment[:,0]/dt))
    align2 = int(numpy.mean(alignment[:,1]/dt))

    print('Alignment:', align, align2)
    
    if align == 0:
        vL  = volumeL[:,:,:]+0
    else:
        vL  = volumeL[:,:,:-align]+0

    if align2 == 0:
        vR  = volumeR[:,:,:]+0
    else:
        vR  = volumeR[:,:,align2:]+0

    volL  = volumeL[:,:,:-389]+0
    volR  = volumeR[:,:,389:]+0

    # optp = (volumeL,volumeR,0,dt)
    # vvL, vvR = _find_tomo360_alignment_single(*optp)

    newVolume = numpy.concatenate((vL,vR),axis=2)
    newVol = numpy.concatenate((volL,volR),axis=2)
    numpy.save('F7volume.npy', newVol)
    
    # newV = numpy.concatenate((vvL,vvR),axis=1)
    # numpy.save('F4volume.npy', newV)

    # volL = numpy.concatenate((volumeL,volumeL*0),axis=2)
    # volR = numpy.concatenate((volumeR*0,volumeR),axis=2)
    # vL = numpy.ones(volL.shape)
    # vR = numpy.ones(volR.shape)
    
    # wc = 1/(2*dt)
    # wx = numpy.linspace(-wc,wc,volL.shape[2])
    
    # dy = numpy.pi/(nangles-1)
    # wc = 1/(2*dy)
    # wy = numpy.linspace(-wc,wc,volL.shape[1])

    # for slices in range(nslices):
    #     atL = numpy.linspace(0,1,2*align)
    #     atR = numpy.linspace(0,1,2*align)

    #     for i in range(nangles):
    #         vL[slices,i,nimages - 2*align:nimages] = numpy.flip(atL)
    #         vR[slices,i,nimages:nimages + 2*align] = atR

    #     L = _translate(vL[slices,:,:]*volL[slices,:,:], 0,  align*dt, wx, wy)
    #     R = _translate(vR[slices,:,:]*volR[slices,:,:], 0, -align*dt, wx, wy)
       
    #     # L = _translate(volumeL[slices,:,:], 0,  align*dt, wx, wy)
    #     # R = _translate(volumeR[slices,:,:], 0, -align*dt, wx, wy)
    
    # for i in range(0,align+1):
    #     L[:,i] = L[:,align+1]
    #     R[:,-i] = R[:,-(align+1)]
 
    # newVolume =  L + R      

    numpy.save('Fvolume.npy', newVolume)

    return newVol

def tomogram_360_to_180(volume, nproc):

    nslices = volume.shape[0]
    nangles2pi = volume.shape[1]
    nimages = volume.shape[2]

    volume = volume/numpy.max(volume)

    if nangles2pi % 2 == 0:
        nangles = nangles2pi // 2

        volumeL = numpy.concatenate((volume[:,:nangles,:],volume[:,:nangles,:]*0),axis=2)
        volumeR = numpy.concatenate((volume[:,nangles:,:]*0,numpy.flip(volume[:,nangles:,:], axis = 2)),axis=2)
    else:
        nangles = (nangles2pi - 1) // 2 

        volumeL = numpy.concatenate((volume[:,:nangles,:],volume[:,:nangles,:]*0),axis=2)
        volumeR = numpy.concatenate((volume[:,nangles:-1,:]*0,numpy.flip(volume[:,nangles:-1,:], axis = 2)),axis=2)

    alignment = get_tomo360_align2(volume, volumeL, volumeR, nproc) 
    print("Aligment:",alignment, volumeL.shape)

    dt = 2.0/(volumeL.shape[2]-1)

    align  = int(alignment[0,0]/dt)
    # print("Align:", align)

    # newVol = Tomo360To180(volume,-align,device=-1)

    # align = 288
    wc = 1/(2*dt)
    wx = numpy.linspace(-wc,wc,volumeL.shape[2])
    
    dy = numpy.pi/(nangles-1)
    wc = 1/(2*dy)
    wy = numpy.linspace(-wc,wc,volumeL.shape[1])

    vL = numpy.ones(volumeL.shape)
    vR = numpy.ones(volumeR.shape)
    
    for slices in range(nslices):
        atL = numpy.linspace(0,1,2*align)
        atR = numpy.linspace(0,1,2*align)

        for i in range(nangles):
            vL[slices,i,nimages - 2*align:nimages] = numpy.flip(atL)
            vR[slices,i,nimages:nimages + 2*align] = atR

        L = _translate(vL[slices,:,:]*volumeL[slices,:,:], 0,  align*dt, wx, wy)
        R = _translate(vR[slices,:,:]*volumeR[slices,:,:], 0, -align*dt, wx, wy)
       
        # L = _translate(volumeL[slices,:,:], 0,  alignment[0,0], wx, wy)
        # R = _translate(volumeR[slices,:,:], 0, -alignment[0,0], wx, wy)
    
    for i in range(0,align+1):
        L[:,i] = L[:,align+1]
        R[:,-i] = R[:,-(align+1)]

    # L = L[:,align:nimages+align]
    # R = R[:,nimages-align:2*nimages-align]
 
    newVolume =  L + R      

    numpy.save('F2volume.npy', newVolume)
    # numpy.save('F5volume.npy', newVol)


    return newVolume


def tomo_corr(volume):
    from scipy import signal
    import matplotlib.pyplot as plt

    nslices = volume.shape[0]
    nangles2pi = volume.shape[1]
    nimages = volume.shape[2]

    # volume = volume/numpy.max(volume)

    if nangles2pi % 2 == 0:
        nangles = nangles2pi // 2
        volumeL = volume[:,:nangles,:]
        volumeR = volume[:,nangles:,:]
    else:
        nangles = (nangles2pi - 1) // 2 
        volumeL = volume[:,:nangles,:]
        volumeR = volume[:,nangles:-1,:]

    # volumeR = numpy.flip(volumeR,axis=2)

    # na = nangles // 2

    index = numpy.zeros((nangles,nslices))
    
    for slices in range(nslices):
        for na in range(nangles):
            # lineL = numpy.load('tomocurveL.npy') #
            # lineR = numpy.load('tomocurveR.npy') #
            lineL = volumeL[slices,na,:]
            lineR = volumeR[slices,na,:]

            # lineL = (lineL - numpy.mean(lineL))/numpy.std(lineL)
            # lineR = (lineR - numpy.mean(lineR))/numpy.std(lineR)
            lineL = lineL/numpy.linalg.norm(lineL)
            lineR = lineR/numpy.linalg.norm(lineR)


            # corr_img = signal.fftconvolve(lineL,lineR)
            corr_img = signal.correlate(lineL,numpy.flip(lineR), mode = 'full')
            lags = signal.correlation_lags(lineL.size,lineR.size, mode = 'full')
            idx = numpy.unravel_index(numpy.argmax(corr_img),corr_img.shape)
            lag = lags[numpy.argmax(corr_img)]
            idx = idx[0]
            print('angle:', na, 2048 - idx//2, lag)
            index[na,slices] = lag #2048 - (idx//2)

    ind = int(numpy.mean(numpy.mean(index[0])))
    indmin = int(numpy.min(numpy.min(index[0])))
    indmax = int(numpy.max(numpy.max(index[0])))

    print('Mean:',ind, 'Min:', indmin, 'Max:', indmax)
    print('Index:',index)

    # plt.plot(corr_img)
    # plt.show()

    vol = numpy.concatenate((volumeL[:,:,:-indmin],numpy.flip(volumeR[:,:,:-indmin], axis=2)),axis=2)
    line2 = numpy.concatenate((lineL[:-indmin],numpy.flip(lineR[:-indmin])))
    numpy.save('F8volume.npy',vol)
    plt.plot(line2)
    plt.show()

    return vol
        

