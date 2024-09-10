import numpy as np
import scipy


def intgrad1(fx,dx=1,f1=0):
    """
    Author: Matteo Berto
    Institution: University of Padua (Università degli Studi di Padova)
    Date: 10th September 2024
    https://github.com/matteo-berto/intgrad1-python
    
    Python implementation of MATLAB "intgrad1" function by John D'Errico.
    
    Note: only the method 2 (integrated spline model) has been implemented.
    
    John D'Errico (2024). Inverse (integrated) gradient
    (https://www.mathworks.com/matlabcentral/fileexchange/9734-inverse-integrated-gradient)
    MATLAB Central File Exchange. Retrieved September 10, 2024.
    
    intgrad: generates a vector, integrating derivative information.
    usage: fhat = intgrad1(dfdx)
    usage: fhat = intgrad1(dfdx,dx)
    usage: fhat = intgrad1(dfdx,dx,f1)
    usage: fhat = intgrad1(dfdx,dx,f1,method)

    arguments: (input)
     dfdx - vector of length nx, as gradient would have produced.

       dx - (OPTIONAL) scalar or vector - denotes the spacing in x
            if dx is a scalar, then spacing in x (the column index
            of fx and fy) will be assumed to be constant = dx.
            if dx is a vector, it denotes the actual coordinates
            of the points in x (i.e., the column dimension of fx
            and fy.) length(dx) == nx

            DEFAULT: dx = 1

       f1 - (OPTIONAL) scalar - defines the first eleemnt of fhat
            after integration. This is just the constant of integration.

            DEFAULT: f1 = 0

       method = 2 --> integrated spline model. This will almost always
            be the most accurate among the alternative methods.
            
    arguments: (output)
      fhat - vector of length nx, containing the integrated function

    Example usage: 
     x = 0:.001:1;
     f = exp(x) + exp(-x);
     dfdx = exp(x) - exp(-x);
     fhat = intgrad1(dfdx,.001,2,2)
    Author; John D'Errico
    Current release: 2
    Date of release: 1/27/06
    size
    """
    if len(fx.shape)>1:
        raise Exception('dfdx must be a vector.')
    sx = np.array([dx]).shape
    nx = fx.size
    if nx<2:
      raise Exception('dfdx must be a vector of length >= 2')
    # if scalar spacings, expand them to be vectors
    uniflag = 1
    if np.array([dx]).size == 1:
      mdx = dx# mean of dx
      xp = np.linspace(0,nx-1,num=nx)*dx
      dx = np.matlib.repmat(dx,nx-1,1)
    elif np.array([dx]).size==nx:
      # dx was a vector, use diff to get the spacing
      xp = dx
      dx = np.diff(dx)
      mdx = np. mean(dx)
      ddx = np.diff(dx)
      if np.any(dx<=0):
        raise Exception('x points must be monotonic increasing')
      elif np.any(np.abs(ddx)>(xp[-1]*1.e-15)):
        uniflag = 0
    else:
      raise Exception('dx is not a scalar or of length == nx')
    if np.size(f1) > 1 or np.isnan(f1) or not np.isfinite(f1): #|| ~isnumeric(f11)
      raise Exception('f1 must be a finite scalar numeric variable.')

    # case 2
    # integrate a spline model
    pp = scipy.interpolate.CubicSpline(xp,fx)
    c = pp.c.T
    fhat = dx*(c[:,3] + dx*(c[:,2]/2 + dx*(c[:,1]/3 + dx*c[:,0]/4)))
    fhat = f1+np.insert(np.cumsum(fhat),0,0)
    
    return fhat