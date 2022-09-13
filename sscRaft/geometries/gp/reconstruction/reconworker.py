from ....rafttypes import *
import numpy as np
import gc
from time import time
import uuid
import SharedArray as sa
from .fbp import *
from .em import *
from .bst import *
from .fst import *
from ..aligment.aligment import *
from ..rings.rings import *
from ....tomogram.flatdark import *
import sys


def em_mpfs_(data, dic, gpu):

   blocksize = dic['blocksize']

   nblocks = ( data.shape[0] ) // blocksize + 1

   niter     = dic['niterations']
   reg       = dic['regularization']
   eps       = dic['epsilon']
   method    = dic['method']

   if method=="tEM":
      InversionMethod = _iterations_tem_mpfs_
   elif method=="eEM":
      InversionMethod = _iterations_eem_mpfs_
   elif method=="EMTV":
      InversionMethod = _iterations_emtv_mpfs_
   else:
      sa.delete(dic['SAname'])
      message = 'ssc-raft: Error! Wrong method: "tEM"/"eEM"/"EMTV". Deleting shared array.'
      logger.error(message)
      raise Exception(message)
   
   reconsize = dic['recon size']
   output = numpy.zeros((data.shape[0],reconsize,reconsize))
   
   for k in range(nblocks):
      _start_ = k * blocksize
      _end_   = min(blocksize, data.shape[0]) 
      output[_start_:_end_,:,:] = InversionMethod( data[_start_:_end_,:,:], niter, gpu, reg, eps, k)
   
   return output


def reconstruction_parallelGPU(data, flat, dark, dic, gpu, **kwargs):
   
   tomogram = flatdarkGPU( data, flat, dark, gpu )

   tomogram = RingsGPU( tomogram, dic, gpu )
   
   if dic['360pan'] == True:
      tomogram = Tomo360To180GPU( tomogram, dic['tomooffset'], gpu )

   if dic['method'] == 'fbp':
      output = fbpGPU( tomogram, dic, gpu )
   elif dic['method'] in ('tEM', 'eEM', 'EMTV'):
      output = em_mpfs_(tomogram, dic, gpu)
   else:
      sa.delete(dic['SAname'])
      message = 'ssc-raft: Error! Wrong method: "tEM"/"eEM"/"EMTV". Deleting shared array.'
      logger.error(message)
      raise Exception(message)
   
   return output


def _worker_reconparallel_(params, start, end, gpu, process):

   output = params[0]
   data   = params[1]
   flat   = params[7]
   dark   = params[8]
   dic    = params[6]

   logger.info(f'Reconstruction pipeline: begin process {process} on gpu {gpu}')

   output[start:end,:,:] = reconstruction_parallelGPU( data[:,start:end, :], flat, dark, dic, gpu )

   logger.info(f'Reconstruction pipeline: end process {process} on gpu {gpu}')
    

def _build_reconparallel_(params):
 
   nslices = params[2]
   gpus    = params[5]
   ngpus = len(gpus)

   b = int( numpy.ceil( nslices/ngpus )  ) 

   processes = []
   for process in range( ngpus ):
            begin_ = process*b
            end_   = min( (process+1)*b, nslices )

            p = multiprocessing.Process(target=_worker_reconparallel_, args=(params, begin_, end_, gpus[process], process))
            processes.append(p)

   for p in processes:
            p.start()

   for p in processes:
            p.join()


def reconstruction_parallel_gpublock( frames, flat, dark, dic ):

   nslices     = frames.shape[1]
   nangles     = frames.shape[0]
   nrays       = frames.shape[2]
   reconsize   = dic['recon size']
   gpus        = dic['gpu']
   outtype     = dic['recon type']

   name  = str( uuid.uuid4())

   try:
      sa.delete(name )
   except:
      pass

   recon  = sa.create(name,[nslices, reconsize, reconsize], dtype=outtype)

   _params_ = ( recon, frames, nslices, nangles, nrays, gpus, dic, flat, dark)

   _build_reconparallel_( _params_ )

   dic.update({'SAname': name})

   sa.delete(name)

   return recon


def reconstruction_parallel(frames, flat, dark, dic, **kwargs):
        
   nrays   = frames.shape[2]
   nslices = frames.shape[1]
   nangles = frames.shape[0]

   dicparams = ('gpu','angles','filter','recon size','precision','regularization','threshold',
               'shift center','tomooffset','360pan','lambda rings','rings block','nangles',
               'blocksize','niterations','epsilon','method')
   defaut = ([0],None,None,nrays,'float32',0,0,False,0,False,0,2,nangles,20,[10,3,8],1e-15,'fbp')
   
   SetDictionary(dic,dicparams,defaut)
   
   gpus  = dic['gpu']
   blocksize = dic['blocksize']

   if blocksize > nslices // len(gpus):
      message = 'ssc-raft: Error! Please check blocksize!'
      logger.error(message)
      raise Exception(message)

   reconsize = dic['recon size']

   if reconsize % 32 != 0:
            reconsize += 32-(reconsize%32)
            logger.info(f'Reconsize not multiple of 32. Setting to: {reconsize}')

   precision = dic['precision'].lower()
   if precision == 'float32':
            precision = 5
            recondtype = np.float32
   elif precision == 'uint16':
            precision = 2
            recondtype = np.uint16
   elif precision == 'uint8':
            precision = 1
            recondtype = np.uint8
   else:
            logger.error(f'Invalid recon datatype:{precision}')
   
   dic.update({'recon size': reconsize,'recon type': recondtype, 'precision': precision})

   if len(gpus) == 1:
            gpu = gpus[0]
            output = reconstruction_parallelGPU( frames, flat, dark, dic, gpu )
   else:
            output = reconstruction_parallel_gpublock( frames, flat, dark, dic ) 

   return output



