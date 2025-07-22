#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.special as sp
import bie.bie as bie

def intervalrootsboyd(f, interv, Nmax = 256, Ftol = 1e-12):
    """
    Find roots of a real analytic function to spectral accuracy. This function
    finds roots of the function f in the real interval (a, b] using sampling at
    Chebyshev points which are doubled in resolution until Fourier coefficients
    have sufficiently decayed, then via Boyd's 2002 method of degree doubling.
    Zeros precisely at the endpoints are not guaranteed to be included or
    excluded. Taken from MSPack/utils/intervalrootsboyd.m by Alex Barnett.

    Parameters
    ----------
    f: callable
        Real analytic function, need not be able to accept vector arguments.
    interv: tuple [float]
        A tuple (a, b) defining the interval in which the roots will be sought.
    Nmax: int (optional)
        Maximum number of iterations. Default 256.
    Ftol: float (optional)
        Fourier coeff decay relativetolerance. Default 1e-12.

    Returns
    -------
    x: array [float]
        Roots over the interval.
    """
    a = interv[0]
    b = interv[1]
    rad = (b - a)/2
    cen = (b + a)/2

    N = 4 # Half minimum number of points on half-circle
    t = np.pi*np.linspace(0, 1, N+1) # Angles theta
    u = np.zeros(N+1) # The data
    y = cen + rad*np.cos(t) # Ordinates
    for i in range(N+1):
        u[i] = f(y[i]) # Initialize function evaluations
    F = np.array([1, 1]) # Dummy value greater than tolerance (if specified...)
    def Fmetr(X):
        print("X[:2]",np.abs(X[:2]))
        print("max X",np.max(np.abs(X)))
        return np.max(np.abs(X[:2]))/np.max(np.abs(X))
    print("decay, tol:",Fmetr(F), Ftol)   
    while N < Nmax and Fmetr(F) > Ftol:
        N *= 2
        print("Trying N={}".format(N))
        t = np.pi*np.linspace(0, 1, N+1)
        y = cen + rad*np.cos(t)
        unew = np.zeros(N+1)
        unew[0:N+1:2] = u
        for i in range(N//2):
            unew[2*i+1] = f(y[2*i+1])
        np.pad(u, (0, N))
        u = unew.copy()
        print("decay, tol:",Fmetr(F), Ftol)
    ufold = np.concatenate((unew[-1:0:-1], unew[:-1])) # Samples on -pi + 2*pi*range(0, N)/N
    F = np.fft.fftshift(np.fft.fft(np.fft.fftshift(ufold)))
    print("Decay metric:",Fmetr(F))
    if Fmetr(F) <= Ftol:
        print("Fourier coeffs have decayed enough, now trying to find roots")
        tz, derr = trigpolyzeros(F)
        x = cen + rad*np.cos(tz)
    if Fmetr(F) > Ftol:
        print("Warning: Fourier coeffs never sufficiently decayed! Try a smaller interval")
    return np.sort(x)[::2] # Repeated elements for some reason...

def trigpolyzeros(F, tol = 1e-3, real = False, realaxis = True):
    """
    Return a list of zeros of complex 2*pi-periodic trigonometric polynomials.
    Returns a list of roots in (-pi, pi] of 2*pi-periodic function given by
    trigonometric polynomial with Fourier series coeff vector F. The ordering
    of F is as returned by fftshift(fft(fftshift(f))), where f samples the
    2*pi-periodic funtcion at -pi + 2*pi*range(0, N)/N, i.e. complex
    exponentials from frequency -N/2 to N/2-1, where N = length(F). The trig
    poly's highest freq is chosen to be real (cos(N/2*t)) as in Trefethen,
    Spectral Methods in MATLAB book. Taken from MSPack/utils/trigpolyzeros.m by
    Alex Barnett.

    Parameters
    ----------

    Returns
    -------

    """
    print("Called trigpolyzeros")
    r = np.roots(np.concatenate((np.array([F[0]/2]), F[-1:0:-1], np.array([F[0]/2]))))
    print("Found trigpolyroots")
    print(r)
    derr = np.abs(np.abs(r) - 1)
    print(derr)
    if not real:
        ii = np.nonzero(derr <= tol)
    else:
        ii = np.nonzero(derr <= tol and r.imag >= 0)
    if realaxis:
        r = np.angle(r[ii]) # was only real part of angle
    else:
        r = np.log(r[ii])/1j
    derr = derr[ii]
    return r, derr















