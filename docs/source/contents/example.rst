Examples
========

EM/Sinograms 
************

* Transmission Expectation Maximization from a parallel tomogram: 

 .. code-block:: python

        from sscRadon import radon
        
        device = 0
        f = numpy.load( <a_square_image> )
        r = 1024
        device = 0
        center = [512,512]

        s = radon.radon_local_gpu( f, r, n, device, center ) 


 
