#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import pandas
import time

class panel():

    def __init__(self, nodes, zpvec, zppvec, weights, index, neighbors,\
                 colinears = [], corneradj = np.array([False, False]),\
                 endpoints = np.zeros((2, 2)), origin = np.array([[0.0], [0.0]])):
        """
        nodes: np.ndarray(float)
            Shape (2, n)
        weights: np.ndarray(float)
            Shape (1, n)
        index: int
        neighbors: tuple(int)
        zpvec: np.ndarray(float)
            Shape (2, n)
        zppvec: np.ndarray(float)
            Shape (2, n)
        zp: np.ndarray(float)
            Shape (1, n)
        n: np.ndarray(float)
            Shape (2, n)
        kappa: np.ndarray(float)
            Shape (1, n)
        corneradj: np.ndarray(bool)
            Is this edge a corner?
        """
        self.nodes = nodes
        self.weights = weights
        n = weights.shape[1]
        self.index = index # not sure if needed
        self.neighbors = neighbors
        self.zpvec = zpvec
        self.zppvec = zppvec
        self.zp = np.linalg.norm(self.zpvec, axis = 0).reshape((1,n))
        self.n = 1/self.zp*np.array([self.zpvec[1], -self.zpvec[0]])
        self.kappa = - np.einsum('ij,ij->j', self.n, self.zppvec)/self.zp**2
        self.corneradj = corneradj
        self.colinears = colinears
        self.endpoints = endpoints
        self.origin = origin



def cheb(n):
    """
    Returns a differentiation matrix D of size (n+1, n+1) and (n+1) Chebyshev
    nodes x for the standard 1D interval [-1, 1]. The matrix multiplies a
    vector of function values at these nodes to give an approximation to the
    vector of derivative values. Nodes are output in descending order from 1 to
    -1. The nodes are given by
    
    .. math:: x_p = \cos \left( \\frac{\pi n}{p} \\right), \quad n = 0, 1, \ldots p.

    Parameters
    ----------
    n: int
        Number of Chebyshev nodes - 1.

    Returns
    -------
    D: numpy.ndarray [float]
        Array of size (n+1, n+1) specifying the differentiation matrix.
    x: numpy.ndarray [float]
        Array of size (n+1,) containing the Chebyshev nodes.
    """
    if n == 0:
        x = 1
        D = 0
        w = 0
    else:
        a = np.linspace(0.0, np.pi, n+1)
        x = np.cos(a)
        b = np.ones_like(x)
        b[0] = 2
        b[-1] = 2
        d = np.ones_like(b)
        d[1::2] = -1
        c = b*d
        X = np.outer(x, np.ones(n+1))
        dX = X - X.T
        D = np.outer(c, 1/c) / (dX + np.identity(n+1))
        D = D - np.diag(D.sum(axis=1))
    return D, x

def sort_panels(panels):

    panelinds = [panel.index for panel in panels]
    cands = []
    for panel in panels:
        if len(panel.neighbors) == 1:
            cands.append(panel)
    if cands[0].index < cands[1].index:
        newpanels = [cands[0]]
    else:
        newpanels = [cands[1]]
    newpanelinds = [panel.index for panel in newpanels]
    for pa in range(1, len(panels)):
        for panel in panels:
            if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                newpanels.append(panel)
                newpanelinds.append(panel.index)
                break
    return newpanels


# To be deprecated
def nystrom(a, b, K, f, n, quad = "trap"):
    """
    Implements the Nystrom method to solve the integral equation 
    $$ (I - K_n)u_n = f $$
    on the interval $t \in [a, b]$, with n quadrature points and the specified quadrature method.
    
    Parameters
    ----------
    a, b: float
        The integration interval.
    K: callable
        The kernel function with calling signature `k = K(t, s)`, where $k$, $t$, $s$ are all scalars.
    f: callable
        The right-hand-side of the integral equation, with calling signature `g = f(t)`, where $g$ and $t$ are scalars.
    n: int
        Number of quadrature nodes.
    quad: string, optional
        Quadrature formula to use. `trap` for the composite trapezoidal rule and `Gauss` for Gaussian quadrature.
    
    Returns
    -------
    sn: np.ndarray [float]
        The quadrature nodes used over the interval [a, b].
    wn: np.ndarray [float]
        The quadrature weights used over the interval [a, b].
    un: np.ndarray [float]
        Array containing the values of the numerical solution $u_n$ evaluated at the quadrature nodes.
    """
    
    A = np.zeros((n, n))
    I = np.identity(n)
    if quad == 'trap':
        sn = np.linspace(a, b, n)
        wn = np.ones(n)*(b-a)/(n-1)
        wn[0] /= 2
        wn[-1] /= 2 
    elif quad == 'Gauss':
        sn, wn = np.polynomial.legendre.leggauss(n)
        sn = sn*(b-a)/2 + (a+b)/2
        wn = wn*(b-a)/2
    else:
        raise TypeError("Quadrature option needs to be 'trap' or 'Gauss'")
    fn = f(sn)
    for i in range(n):
        A[i, :] = K(sn[i], sn)*wn
    un = np.linalg.solve(I - A, fn)
    return sn, wn, un

# To be deprecated
def nystrom_un(K, f, sn, wn, un, t):
    """
    Interpolates the Nystrom quadrature outputs un, using nodes sn and weights wn.    
    """
    n = wn.shape[0]
    unt = f(t) + sum([wn[i]*K(t, sn[i])*un[i] for i in range(n)])
    return unt


def F(t, kd, cosalpha0):
    """
    Helper function for the fast evaluation of lattice sums, defined in Eq.
    (16) in (Yasumoto++, 1999).
    """
    return np.exp(-1j*kd*np.sqrt(1 - t**2))/np.sqrt(1 - t**2)/\
                 (1 - np.exp(-1j*kd*(np.sqrt(1 - t**2) - cosalpha0)))


def F_H1(t, kd, cosalpha0):
    """
    Yasumoto lattice sum helper function, but for H1 instead of H2.
    """
    return np.exp(1j*kd*np.sqrt(1 - t**2))/np.sqrt(1 - t**2)/\
                 (1 - np.exp(1j*kd*(np.sqrt(1 - t**2) - cosalpha0)))


def F_H1_offset(t, kd, cosalpha0):
    """
    Yasumoto lattice sum helper function, but for H1 instead of H2.
    """
    return np.exp(2j*kd*np.sqrt(1 - t**2))/np.sqrt(1 - t**2)/\
                 (1 - np.exp(1j*kd*(np.sqrt(1 - t**2) - cosalpha0)))

def Gn(n, t):
    """
    Helper function for the fast evaluation of lattice sums, defined in Eq.
    (15) in (Yasumoto++, 1999).
    """
    return (t - 1j*np.sqrt(1 - t**2))**n

def trap(f, a, N):
    """
    N-point trapezoidal rule to evaluate the integral of f from 0 to a.
    """
    ts = np.linspace(0, a, N)
    weights = np.ones(N)*a/(N-1)
    weights[0] /= 2
    weights[-1] /= 2
    vals = f(ts)
    if vals.ndim > 1:
        print(vals.shape, weights.shape)
        result = np.einsum('i,ij->j', weights, vals)
    else:
        result = np.dot(weights, vals)
    return result

def Sn(n, kd, cosalpha0, a, N = 1000):
    """
    nth order lattice sum Sn as defined in (Yasumoto++, 1999).
    """
    tau = lambda t: (1j + 1)*t
    f1 = lambda t: (Gn(n, tau(t)) + Gn(n, -tau(t)))*F(tau(t), kd, cosalpha0)
    f2 = lambda t: (Gn(n, tau(t)) + Gn(n, -tau(t)))*F(tau(t), kd, -cosalpha0)
    int1 = trap(f1, a, N)
    int2 = trap(f2, a, N)
    result = np.exp(1j*np.pi/4)*np.sqrt(2)/np.pi*(\
                  (-1)**n*np.exp(1j*kd*cosalpha0)*int1 + np.exp(-1j*kd*cosalpha0)*int2)
    return result

def Sn_H1(n, kd, cosalpha0, a, N = 1000):
    """
    nth order lattice sum Sn as defined in (Yasumoto++, 1999), but starting
    from H1 instead of H2.
    """
    tau = lambda t: (-1j + 1)*t
    f1 = lambda t: (Gn(n, tau(t)) + Gn(n, -tau(t)))*F_H1(tau(t), kd, cosalpha0)
    f2 = lambda t: (Gn(n, tau(t)) + Gn(n, -tau(t)))*F_H1(tau(t), kd, -cosalpha0)
    int1 = trap(f1, a, N)
    int2 = trap(f2, a, N)
    result = -1j*np.exp(1j*np.pi/4)*np.sqrt(2)/np.pi*(\
                    (-1)**n*np.exp(-1j*kd*cosalpha0)*int1 + np.exp(1j*kd*cosalpha0)*int2)
    return result

def Sn_offset(n, kd, cosalpha0, a, N = 1000):
    """
    nth order lattice sum Sn as defined in (Yasumoto++, 1999), but starting
    from H1 instead of H2.
    """
    tau = lambda t: (-1j + 1)*t
    f1 = lambda t: (Gn(n, tau(t)) + Gn(n, -tau(t)))*F_H1_offset(tau(t), kd, cosalpha0)
    f2 = lambda t: (Gn(n, tau(t)) + Gn(n, -tau(t)))*F_H1_offset(tau(t), kd, -cosalpha0)
    int1 = trap(f1, a, N)
    int2 = trap(f2, a, N)
    result = -1j*np.exp(1j*np.pi/4)*np.sqrt(2.0)/np.pi*(\
                    (-1)**n*np.exp(-2j*kd*cosalpha0)*int1 + np.exp(2j*kd*cosalpha0)*int2)
    return result

def Sn_vec(n, kd, cosalpha0, a, N = 1000):
    """
    nth order lattice sum Sn as defined in (Yasumoto++, 1999), but starting
    from H1 instead of H2.
    """
    tau = lambda t: (-1j + 1)*t
    f1 = lambda t: (Gn(n, tau(t)) + Gn(n, -tau(t)))*F_H1(tau(t), kd, cosalpha0)
    f2 = lambda t: (Gn(n, tau(t)) + Gn(n, -tau(t)))*F_H1(tau(t), kd, -cosalpha0)
    int1 = trap(f1, a, N)
    int2 = trap(f2, a, N)
    result = 1j*np.exp(1j*np.pi/4)*np.sqrt(2)/np.pi*np.array([\
                    np.exp(-1j*kd*cosalpha0)*int1, (-1)**n*np.exp(1j*kd*cosalpha0)*int2])
    return result

def Phi_p_man(x, y, k, alpha, M = 10000):
    """ 
    Compute the periodic Green's function by manual sum (very slowly convergent).
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    d = x[..., None] - y[:, None] # Might be slow
    offset = np.zeros_like(d)
    Phi_p = 0.0
    for m in range(-M, M+1):
        offset[0,...] = m
        Phi_p += sp.hankel1(0, k*np.linalg.norm(d - offset, axis = 0)*alpha**m)
    Phi_p *= -1j/4
    return Phi_p

def Phi_p(x, y, k, Sns):
    """
    Periodic fundamental solution of the 2D Helmholtz equation given source
    position y, target position x, wavenumber k, and pre-computed lattice sums
    Sns.
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    d = x[..., None] - y[:, None] # Might be slow
    r = np.linalg.norm(d, axis = 0)
    phi = np.arccos(d[0,...]/r) # TODO: we need to take cos of this anyway, so there may be a shortcut
    n = Sns.shape[0] 
    latsum = sp.hankel1(0, k*r) + Sns[0]*sp.j0(k*r) # May be faster to use the fast versions of J, Y
    for i in range(n-1):
        Jn = sp.jv(i+1, k*r)
        cosnphi = np.cos((i+1)*phi)
        latsum += 2*Sns[i+1]*Jn*cosnphi
    latsum /= 4j
    return latsum

def Phi_p_H2(x, y, k, Sns):
    """
    Periodic fundamental solution of the 2D Helmholtz equation given source
    position y, target position x, wavenumber k, and pre-computed lattice sums
    Sns.
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    d = x[..., None] - y[:, None] # Might be slow
    r = np.linalg.norm(d, axis = 0)
    phi = np.arccos(d[0,...]/r) # TODO: we need to take cos of this anyway, so there may be a shortcut
    n = Sns.shape[0] 
    latsum = sp.hankel2(0, k*r) + Sns[0]*sp.j0(k*r) # May be faster to use the fast versions of J, Y
    for i in range(n-1):
        Jn = sp.jv(i+1, k*r)
        cosnphi = np.cos((i+1)*phi)
        latsum += 2*Sns[i+1]*Jn*cosnphi
    latsum /= 4j
    return latsum


def Phi_p_offset(x, y, k, Sns, alpha, self = False, verbose = False):
    """
    Periodic fundamental solution of the 2D Helmholtz equation given source
    position y, target position x, wavenumber k, and pre-computed lattice sums
    Sns.
    alpha = np.exp(1j*k*d*np.cos(phii))
    If self == True, the n = 0 interaction is omitted from the Green's function.
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    d = x[..., None] - y[:, None] 
    offset = np.zeros_like(d)
    offset[0,...] = 1.0
    r = np.linalg.norm(d, axis = 0)
    rpd = np.linalg.norm(d - offset, axis = 0)
    rmd = np.linalg.norm(d + offset, axis = 0)
    phi = np.arccos(d[0,...]/r) 
    n = Sns.shape[0] 
    result = sp.hankel1(0, k*rpd)*alpha + sp.hankel1(0, k*rmd)/alpha + Sns[0]*sp.j0(k*r)
    if self == True:
        phi = 0.0
    else:
        result += sp.hankel1(0, k*r)
#    if self == True:
#        print("Result before: ", result)
    lats = 0.0
    for i in range(n-1): #TODO: Make the indexing reflect the math
        Jn = sp.jv(i+1, k*r)
        cosnphi = np.cos((i+1)*phi)
        result += 2*Sns[i+1]*Jn*cosnphi
        lats += 2*Sns[i+1]*Jn*cosnphi
#    if self == True:
#        print("result after:", result)
    result *= 1j/4 
    if verbose:
        return sp.hankel1(0, k*r), sp.hankel1(0, k*rpd)*alpha,\
               sp.hankel1(0, k*rmd)/alpha, Sns[0]*sp.j0(k*r), lats
    else:
        return result

def Phi_p_orig(x, y, xorig, yorig, k, Sns, alpha, self = False):
    """
    Periodic fundamental solution of the 2D Helmholtz equation given source
    position y, target position x, wavenumber k, and pre-computed lattice sums
    Sns.
    alpha = np.exp(1j*k*d*np.cos(phii))
    If self == True, the n = 0 interaction is omitted from the Green's function.
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    if np.abs((xorig - yorig)).sum() == 0.0:
        # Same origin
        d = x[..., None] - y[:, None] 
        offset = np.zeros_like(d)
        offset[0,...] = 1.0
        rpd = np.linalg.norm(d - offset, axis = 0)
        rmd = np.linalg.norm(d + offset, axis = 0)
    else:
        # Different origin
        d = x[..., None] - y[:, None] 
        d += xorig[...,None] - yorig[..., None]
        offset = np.zeros_like(d)
        offset[0,...] = 1.0

        rpd = np.linalg.norm(d - offset, axis = 0)
        rmd = np.linalg.norm(d + offset, axis = 0)

    r = np.linalg.norm(d, axis = 0)
    phi = np.arccos(d[0,...]/r) 
    n = Sns.shape[0] 
    result = sp.hankel1(0, k*rpd)*alpha + sp.hankel1(0, k*rmd)/alpha + Sns[0]*sp.j0(k*r)
    if self == True:
        phi = 0.0
    else:
        result += sp.hankel1(0, k*r)
    for i in range(n-1): #TODO: Make the indexing reflect the math
        Jn = sp.jv(i+1, k*r)
        cosnphi = np.cos((i+1)*phi)
        result += 2*Sns[i+1]*Jn*cosnphi
    result *= 1j/4 
    return result





def dPhi_dn_p_man(x, y, n, k, alpha, nxory = 'y', p = 2, hx = 1e-3, hy = 1e-3):
    """
    Normal derivative of the periodic Green's function via Chebyshev
    differentiation matrix and manual summation.
    """
    print("called dPhi_dn manual")
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    if n.ndim < 2:
        n = n.reshape((2, -1))
    dPhi = np.zeros((2, x.shape[1], y.shape[1]), dtype = complex)
    D, xch = cheb(p)
    for i, xi in enumerate(x.T):
#        print("target:",xi)
        # x-component of gradient
        xscaled = xch*hx/2 + xi[0]
        yscaled = np.ones(p+1)*xi[1]
#        print("P:",p)
        targscaled = np.concatenate((xscaled.reshape(1,-1), yscaled.reshape(1,-1)), axis = 0)
#        print("x, y:", targscaled, y)
        Phiix = Phi_p_man(targscaled, y, k, alpha, M = 10000)
#        print("Phiix:",Phiix)
        dPhisx = 2/hx*(D @ Phiix)
        dPhiix = dPhisx[p//2] 
        # y-component of gradient
        xscaled = np.ones(p+1)*xi[0]
        yscaled = xch*hy/2 + xi[1]
        targscaled = np.concatenate((xscaled.reshape(1,-1), yscaled.reshape(1,-1)), axis = 0)
#        print("targscaled:", targscaled)
        Phiiy = Phi_p_man(targscaled, y, k, alpha, M = 10000)
#        print("Phiiy",Phiiy)
        dPhisy = 2/hy*(D @ Phiiy)
        dPhiiy = dPhisy[p//2] 
        dPhi[0,i,:] = dPhiix
        dPhi[1,i,:] = dPhiiy
#        print("gradient:",dPhi[:,i,:])
    if nxory == 'x':
        sumstr = 'ij,ijk->jk'
    else:
        sumstr = 'ik,ijk->jk'
    dPhi_dn = np.einsum(sumstr, n, dPhi)  
    return dPhi_dn

def dPhi_p_dn(x, y, n, k, Sns, nxory = 'y'):
    """
    Returns the directional derivative (wrt the source location) of the
    *periodic* fundamental solution given source position(s) y, target
    position(s) x, and direction(s) ny.
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    if n.ndim < 2:
        n = n.reshape((2, -1))
    d = x[..., None] - y[:, None] # Might be slow
    if nxory == 'x':
        sumstr = 'ij,ijk->jk'
    else:
        sumstr = 'ik,ijk->jk'
    r = np.linalg.norm(d, axis = 0)
    ndotd = np.einsum(sumstr, n, d)
    nxy = np.array([d[1], -d[0]])
    nxydotn = np.einsum(sumstr, n, nxy)
    phi = np.arccos(d[0,...]/r) # TODO: we need to take cos of this anyway, so there may be a shortcut
    phi = np.where(d[1,...] > 0, phi, -phi)
    nsn = Sns.shape[0] 
    # Compute the Jn-s once so we don't have to evaluate them too many times:
    Jns = [sp.jv(j, k*r) for j in range(nsn+1)]

    latsum1 = -sp.hankel1(1, k*r) - Sns[0]*sp.j1(k*r) # May be faster to use the fast versions of J, Y
    latsum2 = np.zeros_like(latsum1)
    for i in range(nsn-1):
        cosnphi = np.cos((i+1)*phi)
        sinnphi = np.sin((i+1)*phi)
        latsum1 += Sns[i+1]*(Jns[i] - Jns[i+2])*cosnphi
        latsum2 += 2*Sns[i+1]*Jns[i+1]*(i+1)/r**2*sinnphi
    latsum = 1/(4j)*(latsum1*k/r*ndotd + latsum2*nxydotn) 
    return latsum

def dPhi_p_dn_orig(x, y, xorig, yorig, n, k, Sns, alpha, nxory = 'y', self = False):
    """
    Returns the x or y normal derivative of the *periodic* fundamental
    solution given source position(s) y, target position(s) x, and direction(s) n.
    The input parameter `nxory` can take the values `x` or `y`. If it is set to `x`, 
    the function returns :math:`n \cdot \\nabla_x \Phi(x, y)`, where `n` 
    should then be the normal at the target `x`; if it is `y`, then it returns :math:`n \cdot\\nabla \Phi(x,
    y)`, where `n` should then be the normal at the source `y`. The inputs `x`,
    `y`, `n`, may be arrays (of vectors), but their shapes need to match as
    described below. The source and its neighboring images are summed by hand,
    the rest are approximated via lattice sum with coefficients `Sns`, as
    defined in Yasumoto & Yoshitomi 1999. `alpha` is the Bloch phase, the phase
    the quasiperiodic solution accrues over a unit cell `alpha` is the Bloch
    phase, the phase the quasiperiodic solution accrues over a unit cell.

    Parameters
    ----------
    x: np.ndarray(float)
        Shape (2,Nx), vector of 2D target location(s)
    y: np.ndarray(float)
        Shape (2,Ny), vector of 2D source location(s)
    n: np.ndarray(float)
        Shape must be (2, Nx) if `nxory = "x"`, or (2,Ny) if `nxory = "y"`.
    k: float, optional
        Wavenumber. If zero, the fundamental solution of the Laplace equation
        is returned, else Helmholtz. Defaults to 0.0.
    nxory: str, optional
        Determines which normal derivative is returned. Takes values `"x"` or
        `"y"`, default `"y"`.

    Returns
    -------
    dphi_dn: np.ndarray(float or complex)
        Shape will be (Nx, Ny), directional derivative of fundamental solution.
    """
#    print("dPhi_p_dn called")
#    print(x.shape)
#    print(y.shape)
#    print(n.shape)
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    if n.ndim < 2:
        n = n.reshape((2, -1))
    if np.abs((xorig - yorig)).sum() == 0.0:
        # Same origin
        d = x[..., None] - y[:, None] 
    else:
        # Different origin
        d = x[..., None] - y[:, None] 
        d += xorig[...,None] - yorig[..., None]
    offset = np.zeros_like(d)
    offset[0,...] = 1.0
    rpd = np.linalg.norm(d - offset, axis = 0)
    rmd = np.linalg.norm(d + offset, axis = 0)
    if nxory == 'x':
        sumstr = 'ij,ijk->jk'
        sign = 1
    elif nxory == 'y':
        sumstr = 'ik,ijk->jk'
        sign = -1
    else:
        print("nxory has to take value 'x' or 'y'!")
    r = np.linalg.norm(d, axis = 0)
    ndotd = np.einsum(sumstr, n, d)
    ndotdm = np.einsum(sumstr, n, d - offset)
    ndotdp = np.einsum(sumstr, n, d + offset)
    nxy = np.array([d[1], -d[0]])
    nxydotn = np.einsum(sumstr, n, nxy)
    if self == True:
        phi = 0.0
    else:
        phi = np.arccos(d[0,...]/r) 
        phi = np.where(d[1,...] > 0, phi, -phi)
    nsn = Sns.shape[0] 
    Jns = [sp.jv(j, k*r) for j in range(nsn+1)]
    latsum1 = -sp.hankel1(1, k*r) - Sns[0]*sp.j1(k*r) 
    latsum2 = np.zeros_like(latsum1)
    latsum3 = -sp.hankel1(1, k*rpd)*alpha
    latsum4 = -sp.hankel1(1, k*rmd)/alpha
    for i in range(nsn-1):
        cosnphi = np.cos((i+1)*phi)
        sinnphi = np.sin((i+1)*phi)
        latsum1 += Sns[i+1]*(Jns[i] - Jns[i+2])*cosnphi
        latsum2 += 2*Sns[i+1]*Jns[i+1]*(i+1)/r**2*sinnphi
    if self == False:
#        print("ndotd:", ndotd)
#        print("nxydotn:",nxydotn)
#        print("nxy:", nxy)
#        print("d:", d)
#        print("n:", n)
        latsum = sign*1j/4*(k*(latsum1*ndotd/r + latsum3*ndotdm/rpd + latsum4*ndotdp/rmd)+ latsum2*nxydotn) 
    else:
        latsum = sign*1j/4*(k*(latsum3*ndotdm/rpd + latsum4*ndotdp/rmd))
    return latsum

def dPhi_p_dn_offset(x, y, n, k, Sns, alpha, nxory = 'y', self = False, verbose = 0, beta = np.pi/2):
    """
    Returns the x or y normal derivative of the *periodic* fundamental
    solution given source position(s) y, target position(s) x, and direction(s) n.
    The input parameter `nxory` can take the values `x` or `y`. If it is set to `x`, 
    the function returns :math:`n \cdot \\nabla_x \Phi(x, y)`, where `n` 
    should then be the normal at the target `x`; if it is `y`, then it returns :math:`n \cdot\\nabla \Phi(x,
    y)`, where `n` should then be the normal at the source `y`. The inputs `x`,
    `y`, `n`, may be arrays (of vectors), but their shapes need to match as
    described below. The source and its neighboring images are summed by hand,
    the rest are approximated via lattice sum with coefficients `Sns`, as
    defined in Yasumoto & Yoshitomi 1999. `alpha` is the Bloch phase, the phase
    the quasiperiodic solution accrues over a unit cell `alpha` is the Bloch
    phase, the phase the quasiperiodic solution accrues over a unit cell.

    Parameters
    ----------
    x: np.ndarray(float)
        Shape (2,Nx), vector of 2D target location(s)
    y: np.ndarray(float)
        Shape (2,Ny), vector of 2D source location(s)
    n: np.ndarray(float)
        Shape must be (2, Nx) if `nxory = "x"`, or (2,Ny) if `nxory = "y"`.
    k: float, optional
        Wavenumber. If zero, the fundamental solution of the Laplace equation
        is returned, else Helmholtz. Defaults to 0.0.
    nxory: str, optional
        Determines which normal derivative is returned. Takes values `"x"` or
        `"y"`, default `"y"`.

    Returns
    -------
    dphi_dn: np.ndarray(float or complex)
        Shape will be (Nx, Ny), directional derivative of fundamental solution.
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    if n.ndim < 2:
        n = n.reshape((2, -1))
    d = x[..., None] - y[:, None] 
    offset = np.zeros_like(d)
    offset[0,...] = 1.0
    rpd = np.linalg.norm(d - offset, axis = 0)
    rmd = np.linalg.norm(d + offset, axis = 0)
    if nxory == 'x':
        sumstr = 'ij,ijk->jk'
        sign = 1
    elif nxory == 'y':
        sumstr = 'ik,ijk->jk'
        sign = -1
    else:
        print("nxory has to take value 'x' or 'y'!")
    r = np.linalg.norm(d, axis = 0)
    ndotd = np.einsum(sumstr, n, d)
    ndotdm = np.einsum(sumstr, n, d - offset)
    ndotdp = np.einsum(sumstr, n, d + offset)
    nxy = np.array([d[1], -d[0]])
    nxydotn = np.einsum(sumstr, n, nxy)
    if self == True:
        phi = 0.0
    else:
        phi = np.arccos(d[0,...]/r) 
        phi = np.where(d[1,...] > 0, phi, -phi)
    nsn = Sns.shape[0] 
    Jns = [sp.jv(j, k*r) for j in range(nsn+1)]
    imagen0 = -1j/4*k*sp.hankel1(1, k*r)/r*ndotd  
    imagenp1 = -1j/4*k*sp.hankel1(1, k*rpd)*alpha/rpd*ndotdm
    imagenm1 = -1j/4*k*sp.hankel1(1, k*rmd)/alpha/rmd*ndotdp
    latsum_der1 = - 1j/4*k*Sns[0]*sp.j1(k*r)/r*ndotd
    latsum_der2 = np.zeros_like(latsum_der1)
    for i in range(1, nsn):
        cosnphi = np.cos(i*phi)
        sinnphi = np.sin(i*phi)
        latsum_der1 += 1j/4*k*Sns[i]*(Jns[i-1] - Jns[i+1])*cosnphi/r*ndotd
        latsum_der2 += 1j/4*2*Sns[i]*Jns[i]*i/r**2*sinnphi*nxydotn
    if self == False:
        result = sign*(imagen0 + imagenp1 + imagenm1 + latsum_der1 + latsum_der2) 
        if verbose == 2:
            print("self is False")
            print("imagen0", imagen0)
            print("imagenp1", imagenp1)
            print("imagenm1", imagenm1)
            print("latsum der 1", latsum_der1)
            print("latsum der 2", latsum_der2)
            print("result:", result)
            print(Sns)
    else:
        latsum_der2_lim = 1j/4*k*Sns[1]/np.sqrt(2.0)
        result = sign*(imagenp1 + imagenm1 + np.where(x[0,...]*x[1,...] > 0, -latsum_der2_lim, latsum_der2_lim))
        if verbose == 2:
            print("self is True")
            print("imagenp1", imagenp1)
            print("imagenm1", imagenm1)
            print("latsumder2lim", latsum_der2_lim)
            print("result", result)
    if verbose == 1:
        return imagen0, imagenp1, imagenm1, latsum_der1, latsum_der2, 1j/4*k*Sns[1]/np.sqrt(2.0), result
    else:
        return result

def Phi(x, y, k = 0.0):
    """
    Returns the fundamental solution given the source position y, and target position x.
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    d = x[..., None] - y[:, None] # Might be slow
    if k == 0.0:
        phi = -1/(2*np.pi)*np.log(np.linalg.norm(d, axis = 0))
    else:
        r = np.linalg.norm(d, axis = 0)
        phi = 1j/4*sp.hankel1(0, k*r)
    return phi

def dPhi_dn(x, y, n, k = 0.0, nxory = 'y'):
    """
    Returns the x or y normal derivative of the fundamental
    solution given source position(s) y, target position(s) x, and direction(s) n.
    The input parameter `nxory` can take the values `x` or `y`. If it is set to `x`, 
    the function returns :math:`n \cdot \\nabla_x \Phi(x, y)`, where `n` 
    should then be the normal at the target `x`; if it is `y`, then it returns :math:`n \cdot\\nabla \Phi(x,
    y)`, where `n` should then be the normal at the source `y`. The inputs `x`,
    `y`, `n`, may be arrays (of vectors), but their shapes need to match as
    described below.

    Parameters
    ----------
    x: np.ndarray(float)
        Shape (2,Nx), vector of 2D target location(s)
    y: np.ndarray(float)
        Shape (2,Ny), vector of 2D source location(s)
    n: np.ndarray(float)
        Shape must be (2, Nx) if `nxory = "x"`, or (2,Ny) if `nxory = "y"`.
    k: float, optional
        Wavenumber. If zero, the fundamental solution of the Laplace equation
        is returned, else Helmholtz. Defaults to 0.0.
    nxory: str, optional
        Determines which normal derivative is returned. Takes values `"x"` or
        `"y"`, default `"y"`.

    Returns
    -------
    dphi_dn: np.ndarray(float or complex)
        Shape will be (Nx, Ny), directional derivative of fundamental solution.
    """
    if x.ndim < 2:
        x = x.reshape((2, -1))
    if y.ndim < 2:
        y = y.reshape((2, -1))
    if n.ndim < 2:
        n = n.reshape((2, -1))
    d = x[..., None] - y[:, None] # Might be slow
    if nxory == 'x':
        sumstr = 'ij,ijk->jk'
        sign = 1
    elif nxory == 'y':
        sumstr = 'ik,ijk->jk'
        sign = -1
    else:
        print("nxory must take the value x or y.")
    if k == 0.0:
        dphi_dn = sign*1/(2*np.pi)*np.einsum(sumstr, n, d)/np.linalg.norm(d, axis = 0)**2
    else:
        r = np.linalg.norm(d, axis = 0)
        dphi_dn = sign*(-1j)*k/4*np.einsum(sumstr, n, d)/r*sp.hankel1(1, k*r)
    return dphi_dn

# To be deprecated
def generate_grid(x1res, x2res, x1lo, x1hi, x2lo, x2hi):
    """
    Generates a 1D array of vectoes pointing to each point on a rectangular
    meshgrid, which is x1res x x2res in terms of resolution, and has edges
    (x1lo, x1hi), (x2lo, x2hi). 
    """
    x1i = np.linspace(x1lo, x1hi, x1res)
    x2i = np.linspace(x2lo, x2hi, x2res)
    x1coords, x2coords = np.meshgrid(x1i, x2i, indexing = 'ij')
    xvecs = np.zeros((x1res*x2res, 2))
    k = 0
    for i in range(x1res):
        for j in range(x2res):
            xvecs[k] = [x1coords[i, j], x2coords[i, j]]
            k += 1
    
    return xvecs, x1coords, x2coords

# To be deprecated
def panels(panelbdrs, quadnodes):
    """
    Takes in a 1D array of panel boundaries and number of quadrature nodes in each panel,
    and returns a flat 1D array of all quadrature nodes and weights.
    """
    M = panelbdrs.shape[0]
    ss, ws = [], []
    for i in range(M-1):
        s, w = np.polynomial.legendre.leggauss(quadnodes[i])
        a = panelbdrs[i]
        b = panelbdrs[i+1]
        s = s*(b-a)/2 + (a+b)/2
        ss.append(s)
        ws.append(w*(b-a)/2)
    return np.array(ss).ravel(), np.array(ws).ravel()


def nystromK_DLP_panels_polygon(panels, k = 0.0):
    """
    Builds a Nystrom matrix K for the double-layer potential based on a list of
    panels that are all linear.
    """
    # Initialize empty K matrix
    Nnodes = sum([p.nodes.shape[1] for p in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)
   
    # Fill K matrix
    i = 0
    for ipanel in panels:
        # Loop over panels
        inodes = ipanel.nodes
        iN = inodes.shape[1]
        j = 0
        for jpanel in panels:
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if ipanel.index == jpanel.index:
                # 1. Source and target on same panel
                # Panels are linear, so entries are identically zero
                pass
            elif jpanel.index in ipanel.colinears:
                pass
#            elif jpanel.index in ipanel.neighbors:
#                # 2. Target is on an adjacent panel to the source
#                pass
            else:
                # 3. Target is on a faraway panel
                K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn(inodes,\
                                       jnodes, jpanel.n, k = k, nxory = 'y')
            j += jN 
        i += iN
    return K

def nystromK_DLP_periodic_offset(panels, k, Sns, alpha, relcorner = False, sort = True, manual = False):
    """
    Builds a Nystrom matrix K for the double-layer potential based on a list of
    panels that are all linear.
    """
    # Initialize empty K matrix
    Nnodes = sum([pa.nodes.shape[1] for pa in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)
  
    # Sort panels by neighbours if prompted
    if sort:
        panelinds = [panel.index for panel in panels]
        cands = []
        for panel in panels:
            if len(panel.neighbors) == 1:
                cands.append(panel)
        if cands[0].index < cands[1].index:
            newpanels = [cands[0]]
        else:
            newpanels = [cands[1]]
        newpanelinds = [panel.index for panel in newpanels]
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break
    else:
        newpanels = panels

#    print([(pa.index, pa.neighbors) for pa in newpanels])
#    print([(pa.nodes[0,0]+pa.origin[0], pa.nodes[0,-1]+pa.origin[0]) for pa in newpanels]) 


    # Fill K matrix
    i = 0
    for ipanel in newpanels:
        # Loop over panels
        inodes = ipanel.nodes
        iN = inodes.shape[1]
        j = 0
        for jpanel in newpanels:
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if ipanel.index == jpanel.index:
                # 1. Source and target on same panel
                # Panels are linear, so entries are identically zero
                pass
            elif jpanel.index in ipanel.colinears:
                pass
            else:
                # 3. Target is on a faraway panel
                if manual:
                    if relcorner:
    #                    print("alpha (inside K)", np.arccos(np.log(alpha)/1j/k)/np.pi)
                        # Check if origins are the same
                        if np.abs((ipanel.origin - jpanel.origin)).sum() == 0.0:
                            K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn_p_man(inodes,\
                                            jnodes, jpanel.n, k, alpha,\
                                            nxory = 'y', p = 6, hx = 1e-3, hy = 1e-3)

                        else:
                            K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn_p_man(inodes + ipanel.origin,\
                                            jnodes + jpanel.origin, jpanel.n, k, alpha,\
                                            nxory = 'y', p = 6, hx = 1e-3, hy = 1e-3)
                    else:
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn_p_man(inodes,\
                                            jnodes, jpanel.n, k, alpha,\
                                            nxory = 'y', p = 6, hx = 1e-3, hy = 1e-3)

                else:
                    if relcorner:
    #                    print("alpha (inside K)", np.arccos(np.log(alpha)/1j/k)/np.pi)
                        # Check if origins are the same
                        if np.abs((ipanel.origin - jpanel.origin)).sum() == 0.0:
                            K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn_offset(inodes,\
                                    jnodes, jpanel.n, k, Sns, alpha, nxory = 'y')
                        else:
                            K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn_offset(inodes + ipanel.origin,\
                                    jnodes + jpanel.origin, jpanel.n, k, Sns, alpha, nxory = 'y')
                    else:
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn_offset(inodes,\
                                    jnodes, jpanel.n, k, Sns, alpha, nxory = 'y')

            j += jN 
        i += iN
    return K







def nystromK_DLP_periodic(panels, k, Sns, relcorner = False, sort = True):
    """
    Builds a Nystrom matrix K for the double-layer potential based on a list of
    panels that are all linear.
    """
    # Initialize empty K matrix
    Nnodes = sum([pa.nodes.shape[1] for pa in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)
  
    # Sort panels by neighbours if prompted
    panelinds = [panel.index for panel in panels]
    cands = []
    for panel in panels:
        if len(panel.neighbors) == 1:
            cands.append(panel)
    if cands[0].index < cands[1].index:
        newpanels = [cands[0]]
    else:
        newpanels = [cands[1]]
    newpanelinds = [panel.index for panel in newpanels]
    if sort:
        for pa in range(1, len(panels)):
#            print(pa)
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
#                    print("found panel")
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break
    print([(pa.index, pa.neighbors) for pa in newpanels])
#    print([(pa.nodes[0,0]+pa.origin[0], pa.nodes[0,-1]+pa.origin[0]) for pa in newpanels]) 

    # Fill K matrix
    i = 0
    for ipanel in newpanels:
#        print("i:",ipanel.index, ipanel.neighbors, ipanel.origin)
        # Loop over panels
        inodes = ipanel.nodes
#        print(inodes)
        iN = inodes.shape[1]
        j = 0
        for jpanel in newpanels:
#            print("j:", jpanel.index, jpanel.neighbors, jpanel.origin)
            # Loop over other panels
            jnodes = jpanel.nodes
#            print(jnodes)
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if ipanel.index == jpanel.index:
#                print("same panel")
                # 1. Source and target on same panel
                # Panels are linear, so entries are identically zero
                pass
            elif jpanel.index in ipanel.colinears:
#                print("colinears")
                pass
#            elif jpanel.index in ipanel.neighbors:
#                # 2. Target is on an adjacent panel to the source
#                pass
            else:
                # 3. Target is on a faraway panel
                if relcorner:
                    # Check if origins are the same
                    if np.abs((ipanel.origin - jpanel.origin)).sum() == 0.0:
#                        print("same origin")
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn(inodes,\
                                jnodes, jpanel.n, k, Sns, nxory = 'y')
                    else:
#                        print("different origin")
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn(inodes + ipanel.origin,\
                                jnodes + jpanel.origin, jpanel.n, k, Sns, nxory = 'y')
                        # TODO: not sure if it's a good idea to do it this way:
                        # maybe better to treat x-y as special when x and y
                        # have different origins at the dPhi_dn level
                else:
#                    print("no rel coords")
                    K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn(inodes,\
                                jnodes, jpanel.n, k, Sns, nxory = 'y')

            j += jN 
        i += iN
    return K


def nystromK_aperiodic(panels, k = 0.0, relcorner = False, layer = 'D'):
    """
    Builds the appropriate Nystrom matrix based on the value of `layer`. If
    `D`, then it is for an exterior Dirichlet problem, i.e. a double layer, and
    if `S`, then an exterior Neumann problem, i.e. the normal derivative of a
    single layer with respect to the target location. The quadrature weights
    are not included, definition is as in Alex Barnett's Math 126 notes.
    """
    # Initialize empty K matrix
    Nnodes = sum([p.nodes.shape[1] for p in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)

    # Fill K matrix
    if layer == 'D':
        nxory = 'y'
    elif layer == 'S':
        nxory = 'x'
    else:
        print("The 'layer' input variable has to be either 'D' or 'S'!")
    i = 0
    for ipanel in panels:
        # Loop over panels
        inodes = ipanel.nodes
        iweights = ipanel.weights
        iN = inodes.shape[1]
        j = 0
        for jpanel in panels:
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if layer == 'D':
                normal = jpanel.n
            else:
                normal = ipanel.n
            if jpanel.index in ipanel.colinears:
                # Panels are colinear (they could be the same panel), entries zero.
                pass
            elif ipanel.index == jpanel.index:
                # Points on the same panel, which is not linear
                for ii in range(iN):
                    for jj in range(jN):
                        if ii == jj:
                            K[i+ii, i+ii] = -1/(4*np.pi)*ipanel.zp[:,ii]*ipanel.kappa[:,ii]*ipanel.weights[:,ii]
                        else:
                            K[i+ii, j+jj] = jweights[:,jj]*jpanel.zp[:,jj]*dPhi_dn(inodes[:,ii], jnodes[:,jj],\
                                    normal[:,jj], k = k, nxory = nxory)
            else:
                # Points on different, non-colinear panels
                if relcorner:
                    # Check if origins are the same
                    if np.abs(ipanel.origin - jpanel.origin).sum() == 0.0:
                        K[i:i+iN,j:j+jN] = jweights*jpanel.zp*dPhi_dn(inodes,\
                                jnodes, normal, k = k, nxory = nxory)
                    else:
                        K[i:i+iN,j:j+jN] = jweights*jpanel.zp*dPhi_dn(inodes + ipanel.origin,\
                                jnodes + jpanel.origin, normal, k = k, nxory = nxory)
                else:
                    K[i:i+iN,j:j+jN] = jweights*jpanel.zp*dPhi_dn(inodes,\
                                jnodes, normal, k = k, nxory = nxory)
            j += jN 
        i += iN
    return K

def nystromK_periodic(panels, k, Sns, alpha, relcorner = False, layer = 'D'):
    """
    Builds a Nystrom matrix K for the double-layer potential based on a list of
    panels that are all linear.
    """
    # Initialize empty K matrix
    Nnodes = sum([pa.nodes.shape[1] for pa in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)

    # Fill K matrix
    if layer == 'D':
        nxory = 'y'
    elif layer == 'dS':
        nxory = 'x'
    elif layer == 'S':
        pass
    else:
        print("The 'layer' input variable has to be either 'D' or 'S'!")

    i = 0
    for ipanel in panels:
        # Loop over panels
        inodes = ipanel.nodes
        iN = inodes.shape[1]
        j = 0
        for jpanel in panels:
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if layer == 'D':
                normal = jpanel.n
            elif layer == 'dS':
                normal = ipanel.n
            elif layer == 'S':
                pass
            else:
                print("layer has to be D or S")
#            if jpanel.index in ipanel.colinears:
#                # Panels are colinear (they could be the same panel), entries zero.
#                 # Check if origins are the same
#                if np.abs((ipanel.origin - jpanel.origin)).sum() == 0.0:
#                    K[i:i+iN,j:j+jN] = jpanel.zp*jweights*dPhi_p_dn_offset(inodes,\
#                            jnodes, normal, k, Sns, alpha, nxory = nxory)
#                else:
#                    K[i:i+iN,j:j+jN] = jpanel.zp*jweights*dPhi_p_dn_offset(inodes + ipanel.origin,\
#                            jnodes + jpanel.origin, normal, k, Sns, alpha, nxory = nxory)               
            if ipanel.index == jpanel.index:
                # Source and target on same panel
                if layer == 'S':
                    for ii in range(iN):
                        for jj in range(jN):
                            if ii == jj:
                                K[i+ii, i+ii] = ipanel.zp[:,ii]*ipanel.weights[:,ii]*(Phi_p_offset(\
                                        inodes[:,ii], jnodes[:, jj], k, Sns, alpha, self = True))
                            else:
                                K[i+ii, j+jj] = jweights[:,jj]*jpanel.zp[:,jj]*Phi_p_offset(inodes[:,ii]\
                                        , jnodes[:,jj], k, Sns, alpha)

                else:
                     for ii in range(iN):
                        for jj in range(jN):
                            if ii == jj:
                                K[i+ii, i+ii] = ipanel.zp[:,ii]*ipanel.weights[:,ii]*dPhi_p_dn_offset(\
                                        inodes[:,ii], jnodes[:, jj], normal[:,ii], k, Sns, alpha,\
                                        nxory = nxory, self = True)
                            else:
                                K[i+ii, j+jj] = jweights[:,jj]*jpanel.zp[:,jj]*dPhi_p_dn_offset(inodes[:,ii]\
                                        , jnodes[:,jj], normal[:,jj], k, Sns, alpha, nxory = nxory)

            else:
                # Target is on a different panel
                if relcorner:
                    if layer == 'D' or layer == 'dS':
                        # Check if origins are the same
                        if np.abs((ipanel.origin - jpanel.origin)).sum() == 0.0:
                            K[i:i+iN,j:j+jN] = jpanel.zp*jweights*dPhi_p_dn_offset(inodes,\
                                    jnodes, normal, k, Sns, alpha, nxory = nxory)
                        else:
                            K[i:i+iN,j:j+jN] = jpanel.zp*jweights*dPhi_p_dn_offset(inodes + ipanel.origin,\
                                    jnodes + jpanel.origin, normal, k, Sns, alpha, nxory = nxory)
                    elif layer == 'S':
                        # Check if origins are the same
                        if np.abs((ipanel.origin - jpanel.origin)).sum() == 0.0:
                            K[i:i+iN,j:j+jN] = jpanel.zp*jweights*Phi_p_offset(inodes,\
                                    jnodes, k, Sns, alpha)
                        else:
                            K[i:i+iN,j:j+jN] = jpanel.zp*jweights*Phi_p_offset(inodes + ipanel.origin,\
                                    jnodes + jpanel.origin, k, Sns, alpha)

                else:
                    if layer == 'D' or layer == 'dS':
                        K[i:i+iN,j:j+jN] = jpanel.zp*jweights*dPhi_p_dn_offset(inodes,\
                                jnodes, normal, k, Sns, alpha, nxory = nxory)
                    elif layer == 'S':
                        K[i:i+iN,j:j+jN] = jpanel.zp*jweights*Phi_p_offset(inodes,\
                                jnodes, k, Sns, alpha)

            j += jN 
        i += iN

#    fig, ax = plt.subplots(1,2)
#    CS1 = ax[0].imshow(np.log10(np.abs(K.real)))
#    CS2 = ax[1].imshow(np.log10(np.abs(K.imag)))
#    fig.colorbar(CS1)
#    fig.colorbar(CS2)
#    plt.show()
#    return
#    print("largest element in K", max(np.abs(K.flatten().real)), max(np.abs(K.flatten().imag)))

    return K



def nystromK_DLP_panels_gen_closed(panels, k = 0.0):
    """
    Builds a Nystrom matrix K for the double-layer potential based on sources
    ss and weights ws.
    """
    # Initialize empty K matrix
    Nnodes = sum([p.nodes.shape[1] for p in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)
    
    # Fill K matrix
    i = 0
    for ipanel in panels:
        # Loop over panels
        inodes = ipanel.nodes
        iweights = ipanel.weights
        iN = inodes.shape[1]
        j = 0
        for jpanel in panels:
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if ipanel.index == jpanel.index:
                # 1. Target is on the same panel as source 
                # Fill element-by-element
                for ii in range(iN):
                    for jj in range(jN):
                        if ii == jj:
                            K[i+ii, i+ii] = (- 1/(2*np.pi)*ipanel.zp[:,ii]*iweights[:,ii]*ipanel.kappa[:,ii])
                        else:
                            K[i+ii, j+jj] = 2*jpanel.zp[:,jj]*jweights[:,jj]*dPhi_dn(inodes[:,ii], jnodes[:,jj],\
                                    jpanel.n[:,jj], k = k, nxory = 'y')
            else:
                # 2. Target is on an adjacent panel to the source OR
                # 3. Target is on a faraway panel
                # We treat these the same way for now
                K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn(inodes,\
                                       jnodes, jpanel.n, k = k, nxory = 'y')

            j += jN 
        i += iN
    return K

def nystromK_dSLP_panels_polygon(panels, k = 0.0, relcorner = False):
    """
    Builds a Nystrom matrix K for the double-layer potential based on a list of
    panels that are all linear.
    """
    # Initialize empty K matrix
    Nnodes = sum([p.nodes.shape[1] for p in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)
   
    # Fill K matrix
    i = 0
    for ipanel in panels:
        # Loop over panels
        inodes = ipanel.nodes
        iN = inodes.shape[1]
        j = 0
        for jpanel in panels:
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if ipanel.index == jpanel.index:
                # Source and target on same panel
                # Panels are linear, so entries are identically zero
                pass
            elif jpanel.index in ipanel.colinears:
                # Panels are colinear, entries are identically zero
                pass
#            elif jpanel.index in ipanel.neighbors:
#                pass
            else:
                # 3. Target is on a faraway panel
                if relcorner:
                    # Check if origins are the same
                    if np.abs(ipanel.origin - jpanel.origin).sum() == 0.0:
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn(inodes,\
                                jnodes, ipanel.n, k = k, nxory = 'x')
                    else:
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn(inodes + ipanel.origin,\
                                jnodes + jpanel.origin, ipanel.n, k = k, nxory = 'x')
                        # TODO: not sure if it's a good idea to do it this way:
                        # maybe better to treat x-y as special when x and y
                        # have different origins at the dPhi_dn level
                else:
                    K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn(inodes,\
                                jnodes, ipanel.n, k = k, nxory = 'x')
            j += jN 
        i += iN
    return K

def nystromK_dSLP_periodic(panels, k, Sns, relcorner = False, sort = True):
    """
    Builds a Nystrom matrix K for the double-layer potential based on a list of
    panels that are all linear, over one or more unit cells.
    """
    # Initialize empty K matrix
    Nnodes = sum([p.nodes.shape[1] for p in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)

    # Sort panels by neighbours if prompted
    panelinds = [panel.index for panel in panels]
    newpanels = [panels[panelinds.index(0)]]
    newpanelinds = [panel.index for panel in newpanels]
    if sort:
        for p in range(1, len(panels)):
#            print(p)
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    print("found panel")
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break
#    print([(p.index, p.neighbors) for p in newpanels])

    # Fill K matrix
    i = 0
    for ipanel in newpanels:
#        print(ipanel.index, ipanel.neighbors)
        # Loop over panels
        inodes = ipanel.nodes
        iN = inodes.shape[1]
        j = 0
        for jpanel in newpanels:
#            print(i, j)
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if ipanel.index == jpanel.index:
#                print("same panel")
                # Source and target on same panel
                # Panels are linear, so entries are identically zero
                pass
            elif jpanel.index in ipanel.colinears:
#                print("colinears")
                # Panels are colinear, entries are identically zero
                pass
#            elif jpanel.index in ipanel.neighbors:
#                pass
            else:
                # 3. Target is on a faraway panel
                if relcorner:
                    # Check if origins are the same
                    if np.abs(ipanel.origin - jpanel.origin).sum() == 0.0:
#                        print("same origin")
                        #print("origins: ", ipanel.origin, jpanel.origin)
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn(inodes,\
                                jnodes, ipanel.n, k, Sns, nxory = 'x')
                    else:
#                        print("different origin")
                        #print("origins: ", ipanel.origin, jpanel.origin)
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn(inodes + ipanel.origin,\
                                jnodes + jpanel.origin, ipanel.n, k, Sns, nxory = 'x')
                        # TODO: not sure if it's a good idea to do it this way:
                        # maybe better to treat x-y as special when x and y
                        # have different origins at the dPhi_dn level
                else:
#                    print("not rel coords")
                    K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn(inodes,\
                                jnodes, ipanel.n, k, Sns, nxory = 'x')
            j += jN 
        i += iN
    return K

def nystromK_dSLP_periodic_offset(panels, k, Sns, alpha, relcorner = False, sort = True):
    """
    Builds a Nystrom matrix K for the double-layer potential based on a list of
    panels that are all linear.
    """
    # Initialize empty K matrix
    Nnodes = sum([pa.nodes.shape[1] for pa in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)

    if sort:
        # Sort panels by neighbours if prompted
        panelinds = [panel.index for panel in panels]
        cands = []
        for panel in panels:
            if len(panel.neighbors) == 1:
                cands.append(panel)
        if cands[0].index < cands[1].index:
            newpanels = [cands[0]]
        else:
            newpanels = [cands[1]]
        newpanelinds = [panel.index for panel in newpanels]
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break
    else:
        newpanels = panels

    # Fill K matrix
    i = 0
    for ipanel in newpanels:
        # Loop over panels
        inodes = ipanel.nodes
        iN = inodes.shape[1]
        j = 0
        for jpanel in newpanels:
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if ipanel.index == jpanel.index:
                # 1. Source and target on same panel
                # Panels are linear, so entries are identically zero
                pass
            elif jpanel.index in ipanel.colinears:
                pass
            else:
                # 3. Target is on a faraway panel
                if relcorner:
                    # Check if origins are the same
                    if np.abs((ipanel.origin - jpanel.origin)).sum() == 0.0:
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn_offset(inodes,\
                                jnodes, ipanel.n, k, Sns, alpha, nxory = 'x')
                    else:
                        K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn_offset(inodes + ipanel.origin,\
                                jnodes + jpanel.origin, ipanel.n, k, Sns, alpha, nxory = 'x')
                else:
                    K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_p_dn_offset(inodes,\
                                jnodes, ipanel.n, k, Sns, alpha, nxory = 'x')

            j += jN 
        i += iN
    return K


def nystromK_dSLP_panels_gen_closed(panels, k = 0.0):
    """
    Builds a Nystrom matrix K for the derivative of the single-layer potential
    based on sources ss and weights ws.
    """
    # Initialize empty K matrix
    Nnodes = sum([p.nodes.shape[1] for p in panels])
    if k == 0.0:
        K = np.zeros((Nnodes, Nnodes))
    else:
        K = np.zeros((Nnodes, Nnodes), dtype = complex)
    
    # Fill K matrix
    i = 0
    for ipanel in panels:
        # Loop over panels
        inodes = ipanel.nodes
        iweights = ipanel.weights
        iN = inodes.shape[1]
        j = 0
        for jpanel in panels:
            # Loop over other panels
            jnodes = jpanel.nodes
            jweights = jpanel.weights
            jN = jnodes.shape[1]
            if ipanel.index == jpanel.index:
                # 1. Target is on the same panel as source 
                # Fill element-by-element
                for ii in range(iN):
                    for jj in range(jN):
                        if ii == jj:
                            K[i+ii, i+ii] = ( 1/(2*np.pi)*ipanel.zp[:,ii]*iweights[:,ii]*ipanel.kappa[:,ii])
                        else:
                            K[i+ii, j+jj] = 2*jpanel.zp[:,jj]*jweights[:,jj]*dPhi_dn(inodes[:,ii], jnodes[:,jj],\
                                    ipanel.n[:,ii], k = k, nxory = 'x')
            else:
                # 2. Target is on an adjacent panel to the source OR
                # 3. Target is on a faraway panel
                # We treat these the same way for now
                K[i:i+iN,j:j+jN] = 2*jpanel.zp*jweights*dPhi_dn(inodes,\
                                       jnodes, ipanel.n, k = k, nxory = 'x')

            j += jN 
        i += iN
    return K

def latsumcoeffs(k, kappa, nmax = 30, a = 10.0, eps = 1e-7, N = int(10.0/0.0005)):
    """
    Calculates the lattice sum coefficients from S_0 to S_{nmax}.
    """
    kd = k*1.0 # TODO: allow different periodicities
    a = max(-np.log(eps)/kd, a)
    alpha = np.exp(1j*kappa)
    Sns = np.zeros(nmax + 1, dtype = complex)
    for i in range(nmax + 1):
        Sns[i] = Sn_offset(i, kd, kappa/k, a, N = N)
    return Sns

def solve_bie_ptsrc_asm(panels, path, xsource, k, fileid, relcorner = False, bc = "Dir"):
    """
     
    """
    # Files to store the density, lattice sum coeffs, and complex integral path
    densityfile = "Helmext_{}_density_{}.txt".format(bc, fileid)
    latsumcoeffile = densityfile.replace("density", "latsumcoef")
    kappapathfile = densityfile.replace("density", "kappapath")
    # Loop over nodes along complex kappa integral path
    with open(densityfile, 'a') as densityf:
        with open(latsumcoeffile, 'a') as latsumf:
            for kappa in path.nodes:
                cosphii = kappa/k
                alph = np.exp(1j*kappa)
                Sns = latsumcoeffs(k, kappa)
                ui = lambda x: Phi_offset(x, xsource, k, Sns, alph, self = False).flatten()
                dui = lambda nvec, x: dPhi_p_dn_offset(x, xsource, nvec, k,\
                                                       Sns, alp, nxory = 'x',\
                                                       self = False).flatten()
                if bc == "Dir":
                    f = lambda x: -ui(x)
                elif bc == "Neu":
                    f = lambda nvec, x: -dui(nvec, x)
                else:
                    print("The boundary conditions (`bc`) must either be 'Dir' or 'Neu'!")
                density = solve_bie_periodic(panels, f, k, Sns, alph, relcorner = relcorner, bc = bc)
                densityf.write("# kappa = {}\n".format(kappa))
                latsumf.write("# kappa = {}\n".format(kappa))
                np.savetxt(densityf, density)
                np.savetxt(latsumf, Sns)
    with open(kappapathfile, 'a') as kappaf:
        for kappa, weight in zip(path.nodes, path.weights):
            f.write("{}, {}\n").format(kappa, weight)
    print("Done!")

def solve_bie_aperiodic(panels, f, k, relcorner = False, bc = 'Dir'):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))

    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    if bc == 'Dir':
        K = nystromK_aperiodic(panels, k = k, relcorner = relcorner, layer = 'D')
        N = K.shape[0]
        fN = f(allnodes)
        A = np.identity(N) + 2*K
        sign = 1
    elif bc == 'Neu':
        K = nystromK_aperiodic(panels, k = k, relcorner = relcorner, layer = 'dS')
        N = K.shape[0]
        fN = f(allns, allnodes)
        A = np.identity(N) - 2*K
        sign = -1
    else:
        print("The bc parameter has to to take the value 'Neu' or 'Dir'!")

    # Solve Nystrom system
    density = np.linalg.solve(A, sign*2*fN)
#    plt.figure()
#    plt.imshow((A-np.identity(N)).real)
#    plt.savefig("helmextdir-A.pdf")
    print((A - np.identity(N))[:10,:10])
    print((sign*2*fN)[:10])
    print(density[:10])
    return density   


def solve_bie_periodic_flat_Neu(panels, f, k, Sns, alpha, relcorner = False, bc = 'Dir'):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    npanel = len(panels)
    p = panels[0].nodes.shape[1]

    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    if bc == 'Dir':
        print("Bc has to be Neu")
    elif bc == 'Neu':
        fN = f(allns, allnodes)
    else:
        print("The bc parameter has to to take the value 'Neu'")

    density = -2*fN
    return density   

def solve_bie_periodic(panels, f, k, Sns, alpha, relcorner = False, bc = 'Dir'):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
#    print("relcorner was set to", relcorner)
#    print("alpha is:", alpha)
#    print("Sns are:", Sns)
    npanel = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, npanel*p))
    allns = np.zeros((2, npanel*p))

    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    if bc == 'Dir':
        K = nystromK_periodic(panels, k, Sns, alpha, relcorner = relcorner, layer = 'D')
#        print(K)
        N = K.shape[0]
        fN = f(allnodes)
#        plt.figure()
#        plt.plot(allnodes[0,:], fN.real)
#        plt.plot(allnodes[0,:], fN.imag)
#        plt.savefig("fn-check.pdf")
        A = np.identity(N, dtype = complex) + 2*K
        sign = 1
    elif bc == 'Neu':
#        print("called neu")
        K = nystromK_periodic(panels, k, Sns, alpha, relcorner = relcorner, layer = 'dS')
#        K = np.zeros((npanel*p, npanel*p), dtype = complex)
        N = K.shape[0]
#        print("N: ", N, "npanel:", len(panels), "p: ", p)
        fN = f(allns, allnodes)
        A = np.identity(N, dtype = complex) - 2*K
#        print("A:", A[:5,:5])
#        print("A shape:", A.shape, "fN shape:", fN.shape)
        sign = -1
        #print(K)
        #print(fN)

    elif bc == 'DirS':
        K = nystromK_periodic(panels, k, Sns, alpha, relcorner = relcorner, layer = 'S')\
#           +nystromK_periodic(panels, k, Sns, alpha, relcorner = relcorner, layer = 'D')
        N = K.shape[0]
#        print(K)
        fN = f(allnodes)
        A = np.identity(N, dtype = complex) + 2*K
        sign = 1
    else:
        print("The bc parameter has to to take the value 'Neu' or 'Dir'!")

#    fig, ax = plt.subplots(1,2)
#    CS1 = ax[0].imshow(np.log10(np.abs(K.real)))
#    CS2 = ax[1].imshow(np.log10(np.abs(K.imag)))
#    plt.show()
#    fig.colorbar(CS1)
#    fig.colorbar(CS2)
#    print(K[:5, :5])




    # Solve Nystrom system
#    density = -2*fN#np.linalg.solve(A, sign*2*fN)
    density = np.linalg.solve(A, sign*2*fN)
#    print("total dof in nyst:",A.shape[0])
#    print(density)
#    plt.figure()
#    plt.plot(allnodes[0,:], density.real)
#    plt.plot(allnodes[0,:], density.imag)
#    plt.savefig("solve-tau.pdf")


#    print("Au = ", , "2*fn*sign", sign*2*fN)
#    print("remainder:", np.max(np.abs(np.dot(A, density) - sign*2*fN)))
    return density   


def solve_Laplace_ext_Neu(panels, f):
    """
    Solve's Laplace's equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(panels):
        allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    K = nystromK_dSLP_panels_gen_closed(panels)

    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allns, allnodes)
    A = np.identity(N) + K
    sigmaN = np.linalg.solve(A, -2*fN)
    return sigmaN   

def reconstruct_sol_SLP_periodic(panels, sigmaN, x, k, Sns, relcorner = True, sort = True):
    """
    Reconstructs a function represented by a SLP `sigmaN`, evaluated on the
    boundary defined by `panels`, at points `x`.
    """
    # Sort panels by neighbours if prompted
    panelinds = [panel.index for panel in panels]
    newpanels = [panels[panelinds.index(0)]]
    newpanelinds = [panel.index for panel in newpanels]
    if sort:
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break

    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    for i, panel in enumerate(newpanels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else: 
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights

    nodevals = Phi_p(x, allnodes, k, Sns)*allzps*sigmaN
    u = np.einsum('ij,ij->i', nodevals, allweights)
    return u

def reconstruct_sol_SLP_periodic_offset(panels, tauN, x, k, Sns, alpha, relcorner = True, sort = True):
    """
    Reconstructs a function represented by a SLP `sigmaN`, evaluated on the
    boundary defined by `panels`, at points `x`.
    """
    # Sort panels by neighbours if prompted
    panelinds = [panel.index for panel in panels]
    cands = []
    for panel in panels:
        if len(panel.neighbors) == 1:
            cands.append(panel)
    if cands[0].index < cands[1].index:
        newpanels = [cands[0]]
    else:
        newpanels = [cands[1]]
    newpanelinds = [panel.index for panel in newpanels]
    if sort:
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break

    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    for i, panel in enumerate(newpanels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else: 
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights


    nodevals = Phi_p_offset(x, allnodes, k, Sns, alpha)*allzps*tauN
    u = np.einsum('ij,ij->i', nodevals, allweights)
    return u


def reconstruct_sol_aperiodic(panels, density, x, k = 0.0, relcorner = False, layer = 'D'):
    """
    """
    # Sort panels by neighbours if prompted
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else: 
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights
        allns[:,i*p:(i+1)*p] = panel.n

    if layer == 'D':
        nodevals = dPhi_dn(x, allnodes, allns, k = k, nxory = 'y')*allzps*density
    elif layer == 'S':
        nodevals = Phi(x, allnodes, k = k)*allzps*density 
    else:
        print("The 'layer' parameter has to take values 'D' or 'S'!")
    
    u = np.einsum('ij,ij->i', nodevals, allweights)
    return u

def reconstruct_sol_periodic(panels, density, x, k, Sns, alpha, relcorner = False, layer = 'D'):
    """
    """
    # Sort panels by neighbours if prompted
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else: 
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights
        allns[:,i*p:(i+1)*p] = panel.n

    if layer == 'D':
        nodevals = dPhi_p_dn_offset(x, allnodes, allns, k, Sns, alpha, nxory = 'y')*allzps*density
    elif layer == 'S':
        nodevals = Phi_p_offset(x, allnodes, k, Sns, alpha)*allzps*density 
    else:
        print("The 'layer' parameter has to take values 'D' or 'S'!")
    
    u = np.einsum('ij,ij->i', nodevals, allweights)
    return u

def reconstruct_sol_dsol_periodic(panels, density, x, k, Sns, alpha, relcorner = False, layer = 'D'):
    """
    """
    # Sort panels by neighbours if prompted
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else: 
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights

    nx1 = np.array([[1], [0]])
    nx2 = np.array([[0], [1]])

    if layer == 'D':
        print("Second derivative of G_QP not implemented yet!")
        u = None
    elif layer == 'S':
        nodevals = Phi_p_offset(x, allnodes, k, Sns, alpha)*allzps*density 
        dnodevalsx1 = dPhi_p_dn_offset(x, allnodes, nx1, k, Sns, alpha, nxory = 'x')*allzps*density
        dnodevalsx2 = dPhi_p_dn_offset(x, allnodes, nx2, k, Sns, alpha, nxory = 'x')*allzps*density
    else:
        print("The 'layer' parameter has to take values 'D' or 'S'!")
    
    u = np.einsum('ij,ij->i', nodevals, allweights)
    dux1 = np.einsum('ij,ij->i', dnodevalsx1, allweights)
    dux2 = np.einsum('ij,ij->i', dnodevalsx2, allweights)

    return u, dux1, dux2




def reconstruct_sol_SLP(panels, sigmaN, x, k = 0.0, relcorner = False):
    """
    Reconstructs a function represented by a SLP `sigmaN`, evaluated on the
    boundary defined by `panels`, at points `x`.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else: 
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights

    nodevals = Phi(x, allnodes, k = k)*allzps*sigmaN
    u = np.einsum('ij,ij->i', nodevals, allweights)
    return u

def solve_Helm_ext_Dir(panels, f, k = 0.0):
    """

    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(panels):
        allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    K = nystromK_DLP_panels_gen_closed(panels, k = k)

    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allnodes)
    A = np.identity(N) + K
    tauN = np.linalg.solve(A, 2*fN)
    
    return tauN   

def solve_Helm_ext_Dir_polygon(panels, f, k = 0.0):
    """

    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(panels):
        allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    K = nystromK_DLP_panels_polygon(panels, k = k)

    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allnodes)
    A = np.identity(N) + K
    tauN = np.linalg.solve(A, 2*fN)
    
    return tauN   


def solve_Helm_ext_Neu_polygon(panels, f, k = 0.0, relcorner = False):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    K = nystromK_dSLP_panels_polygon(panels, k = k, relcorner = relcorner)

    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allns, allnodes)
    A = np.identity(N) + K
    sigmaN = np.linalg.solve(A, -2*fN)
    return sigmaN   

def solve_Helm_ext_Neu_stairs(panels, f, k, Sns, relcorner = True, sort = True):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))

    # Sort panels by neighbours if prompted
    panelinds = [panel.index for panel in panels]
    newpanels = [panels[panelinds.index(0)]]
    newpanelinds = [panel.index for panel in newpanels]
    if sort:
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break

    for i, panel in enumerate(newpanels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    K = nystromK_dSLP_periodic(newpanels, k, Sns, relcorner = relcorner)

    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allns, allnodes)
    A = np.identity(N) + K
    sigmaN = np.linalg.solve(A, -2*fN)
    return sigmaN   

def solve_Helm_ext_Neu_stairs_offset(panels, f, k, Sns, alpha, relcorner = True, sort = True):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))

    panelinds = [panel.index for panel in panels]
    cands = []
    for panel in panels:
        if len(panel.neighbors) == 1:
            cands.append(panel)
    if cands[0].index < cands[1].index:
        newpanels = [cands[0]]
    else:
        newpanels = [cands[1]]
    newpanelinds = [panel.index for panel in newpanels]
    if sort:
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break

    for i, panel in enumerate(newpanels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    K = nystromK_dSLP_periodic_offset(newpanels, k, Sns, alpha, relcorner = relcorner)

    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allns, allnodes)
    A = np.identity(N) + K

    sigmaN = np.linalg.solve(A, -2*fN)
    return sigmaN   


def solve_Helm_ext_Dir_stairs(panels, f, k, Sns, relcorner = True, sort = True):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))

    # Sort panels by neighbours if prompted
    panelinds = [panel.index for panel in panels]
    newpanels = [panels[panelinds.index(0)]]
    newpanelinds = [panel.index for panel in newpanels]
    if sort:
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break

    for i, panel in enumerate(newpanels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    K = nystromK_DLP_periodic(newpanels, k, Sns, relcorner = relcorner)

    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allnodes)
    A = np.identity(N) + K
    tauN = np.linalg.solve(A, 2*fN)
    return tauN   

def solve_Helm_ext_Dir_sp_offset(panels, f, k, Sns, alpha, relcorner = True):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))

    for i, panel in enumerate(panels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n

    #K = nystromK_DLP_periodic_offset(panels, k, Sns, alpha, relcorner = relcorner, sort = False)
    K = np.zeros((n*p, n*p), dtype = complex) # cut compt cost of tests, we know this is zero.
    #print(K)

    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allnodes)
    A = np.identity(N) + K
    tauN = np.linalg.solve(A, 2*fN)
    return tauN   


def solve_Helm_ext_Dir_stairs_offset(panels, f, k, Sns, alpha, relcorner = True, sort = True, manual = False):
    """
    Solves the Helmholtz equation on the exterior of the boundary defined by the
    input panels object, subject to Neumann boundary conditions specified
    through f. Returns the value of the SLP representation at the boundary
    nodes.
    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allns = np.zeros((2, n*p))

    if sort: 
        panelinds = [panel.index for panel in panels]
        cands = []
        for panel in panels:
            if len(panel.neighbors) == 1:
                cands.append(panel)
        if cands[0].index < cands[1].index:
            newpanels = [cands[0]]
        else:
            newpanels = [cands[1]]
        newpanelinds = [panel.index for panel in newpanels]
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break
    else:
        newpanels = panels

    for i, panel in enumerate(newpanels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else:
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allns[:,i*p:(i+1)*p] = panel.n


    K = nystromK_DLP_periodic_offset(newpanels, k, Sns, alpha, relcorner = relcorner, manual = manual)
#    fig, ax = plt.subplots(1,2)
#    CS1 = ax[0].imshow(K.real)
#    CS2 = ax[1].imshow(K.imag)
#    plt.show()
#    fig.colorbar(CS1)
#    fig.colorbar(CS2)
#    plt.savefig("nystromK_helm_fluxcheck.pdf")
#    plt.close()    


    # Solve Nystrom system
    N = K.shape[0]
    fN = f(allnodes)
    A = np.identity(N) + K
    tauN = np.linalg.solve(A, 2*fN)
    return tauN   

def reconstruct_sol_DLP(panels, tauN, x, k = 0.0):
    """

    """
    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(panels):
        allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights
        allns[:,i*p:(i+1)*p] = panel.n

    nodevals = dPhi_dn(x, allnodes, allns, nxory = 'y', k = k)*allzps*tauN
    u = np.einsum('ij,ij->i', nodevals, allweights)
    return u

def reconstruct_sol_DLP_periodic(panels, tauN, x, k, Sns, relcorner = True, sort = True):
    """
    Reconstructs a function represented by a SLP `sigmaN`, evaluated on the
    boundary defined by `panels`, at points `x`.
    """
    # Sort panels by neighbours if prompted
    if sort:
        panelinds = [panel.index for panel in panels]
        newpanels = [panels[panelinds.index(0)]]
        newpanelinds = [panel.index for panel in newpanels]
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break
    else:
        newpanels = panels

    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(newpanels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else: 
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights
        allns[:,i*p:(i+1)*p] = panel.n


    nodevals = dPhi_p_dn(x, allnodes, allns, k, Sns, nxory = 'y')*allzps*tauN
    u = np.einsum('ij,ij->i', nodevals, allweights)
    return u

def reconstruct_sol_DLP_periodic_offset(panels, tauN, x, k, Sns, alpha, relcorner = True,\
                                        sort = True, manual = False):
    """
    Reconstructs a function represented by a SLP `sigmaN`, evaluated on the
    boundary defined by `panels`, at points `x`.
    """
    # Sort panels by neighbours if prompted
    if sort:
        panelinds = [panel.index for panel in panels]
        cands = []
        for panel in panels:
            if len(panel.neighbors) == 1:
                cands.append(panel)
        if cands[0].index < cands[1].index:
            newpanels = [cands[0]]
        else:
            newpanels = [cands[1]]
        newpanelinds = [panel.index for panel in newpanels]
        for pa in range(1, len(panels)):
            for panel in panels:
                if panel.index not in newpanelinds and panel.index in newpanels[-1].neighbors:
                    newpanels.append(panel)
                    newpanelinds.append(panel.index)
                    break
    else:
        newpanels = panels

    n = len(panels)
    p = panels[0].nodes.shape[1]
    allnodes = np.zeros((2, n*p))
    allzps = np.zeros((1, n*p))
    allweights = np.zeros((1, n*p))
    allns = np.zeros((2, n*p))
    for i, panel in enumerate(newpanels):
        if relcorner:
            allnodes[:,i*p:(i+1)*p] = panel.nodes + panel.origin
        else: 
            allnodes[:,i*p:(i+1)*p] = panel.nodes
        allzps[:,i*p:(i+1)*p] = panel.zp
        allweights[:,i*p:(i+1)*p] = panel.weights
        allns[:,i*p:(i+1)*p] = panel.n

    if manual:
        nodevals = dPhi_dn_p_man(x, allnodes, allns, k, alpha, nxory = 'y',\
                                 p = 6, hx = 1e-3, hy = 1e-3)*allzps*tauN
    else:
        nodevals = dPhi_p_dn_offset(x, allnodes, allns, k, Sns, alpha, nxory = 'y')*allzps*tauN
    #print(nodevals.shape)
    u = np.einsum('ij,ij->i', nodevals, allweights)
    return u


def reconstruct_sol_periodic_vert(panels, density, k, Sns, kappa, relcorner = True, layer = 'S', res = 10, ncells = [-1, 0, 1]):
    """
    """
    alpha = np.exp(1j*kappa)
    d = 1.0 # TODO don't hard code this 
    nvcells = 6
    x1lo, x1hi, x2lo, x2hi, x1res, x2res = d*(-0.5 + ncells[0]), d*(0.5 + ncells[-1]),\
                                           -d/2, (nvcells + 1/2)*d, res*len(ncells) + 1, (nvcells+1)*res + 1
    x1i = np.linspace(x1lo, x1hi, x1res)[:-1]
    x2i = np.linspace(x2lo, x2hi, x2res)[:-1]
    us = np.zeros((len(ncells), res*(nvcells+1)*res), dtype = complex)
    origcellx1 = np.linspace(-d/2, d/2, res+1)[:-1]
    origcellx2 = origcellx1
    x1coords, x2coords = np.meshgrid(x1i, x2i, indexing = 'ij')
    origxvecs = np.zeros((2, res*res))
    origxvecs[0] = np.repeat(origcellx1, res)
    origxvecs[1] = np.repeat(origcellx2.reshape(-1, 1), res, axis = 1).flatten("F")

    ucell = reconstruct_sol_periodic(panels, density, origxvecs, k, Sns, alpha, relcorner = relcorner, layer = layer)
    ucellres = ucell.reshape((res, res))
    uy0 = ucellres[:, -1]
    y0 = origcellx2[-1]
    # Fourier transform to get field above unit cell
    fcoefs = 1/res*np.fft.fft(np.exp(-1j*kappa*origcellx1)*uy0)
    print(fcoefs)
    freqs = np.fft.fftfreq(res, d/res)
    kns = np.sqrt(k**2 - (freqs*2*np.pi + kappa)**2 + 0*1j)
    kns = kns.real*np.where(kns.imag < 0, -1, 1) + np.abs(kns.imag)*1j
    utop = np.zeros((res, (nvcells)*(res)), dtype = complex)
#    print(utop.shape, len(x2i[x2i > y0]))
#    print(y0, x2i)

    for j, y in enumerate(x2i[x2i > y0]):
        utop_oney = res*np.fft.ifft(fcoefs*np.exp(1j*kns*(y - y0)))*np.exp(1j*kappa*origcellx1)
        vecs = np.zeros((2, res))
        vecs[0] = origcellx1
        vecs[1] = y
        utop[:,j] = utop_oney
        print(utop_oney[0])
        utot = np.concatenate((ucellres, utop), axis = 1)

    for j, ncell in enumerate(ncells):
        us[j,:] = utot.flatten()*alpha**ncell

    us = us.reshape((x1res-1, x2res-1))

    return x1coords, x2coords, us


def reconstruct_arrayscan_fromfile_singlekappa(panels, files, res = 10, ncells = [-1, 0, 1], nvcells = 6, deriv = False, no_return_xs = False):
    """
    Parameters
    ----------
    panels: list [Panel]
        List of Panel objects defining the boundary geometry.
    files: list [string]
        List of three files in any order:
            - list of kappa values and weights for array scanning integral
            - density at the boundary at each kappa value
            - lattice sum coefficients at each kappa value
    res: int (optional)
        Resolution of each cell will be res x res
    ncells: list [int] (optional)
        Index of which cells to plot the solution across. 0 belongs to the original cell.

    Returns
    -------
    ncells x res x rex values to be plotted.
    """
    fieldfile = [f for f in files if "sigma" in f][0]
    snsfile = [f for f in files if "Sns" in f][0]
    kappafile = [f for f in files if "kappas" in f][0]
    paramfile = [f for f in files if "params" in f][0]

    # Read in kappa, tau/sigma on bdry, and lattice sum coeffs from file
    fields = np.loadtxt(fieldfile, dtype = complex)
    allsns = np.loadtxt(snsfile, dtype = complex)
    nkappas = 1
    with open(snsfile, 'r') as f:
        kappa = complex(f.readline().split(" = ")[-1])
        print("Kappa read in was:",kappa)
    fields = fields.reshape((nkappas, -1))
    allsns = allsns.reshape((nkappas, -1))
    paramsdata = pandas.read_csv(paramfile, sep = ', ')
    k = paramsdata['k'][0]
    print("Wavenumber value: ", k)

    # Generate grid 
    d = 1.0 
    x1lo, x1hi, x2lo, x2hi, x1res, x2res = d*(-0.5 + ncells[0]), d*(0.5 + ncells[-1]),\
                                           -d/2, (nvcells + 1/2)*d, res*len(ncells) + 1, (nvcells+1)*res + 1
    x1i = np.linspace(x1lo, x1hi, x1res)[:-1]
    x2i = np.linspace(x2lo, x2hi, x2res)[:-1]
    origcellx1 = np.linspace(-d/2, d/2, res+1)[:-1]
    origcellx2 = origcellx1
    x1coords, x2coords = np.meshgrid(x1i, x2i, indexing = 'ij')
    origxvecs = np.zeros((2, res*res))
    origxvecs[0] = np.repeat(origcellx1, res)
    origxvecs[1] = np.repeat(origcellx2.reshape(-1, 1), res, axis = 1).flatten("F")

    us = np.zeros((len(ncells), res*(nvcells+1)*res, nkappas), dtype = complex)

    for i in range(nkappas):
        time0 = time.time()
        fieldN = fields[i,:]
        Sns = allsns[i,:]
        alph = np.exp(1j*kappa)

        ucell = reconstruct_sol_periodic(panels, fieldN, origxvecs, k, Sns, alph, relcorner = True, layer = 'S')

        ucellres = ucell.reshape((res, res))
        xsource = np.array([[-0.2], [0.1]]) # TODO: read this in!
        ui = lambda x: Phi_p_offset(origxvecs, xsource, k, Sns, alph, self = False).flatten()
        nvec = np.array([[0.0], [1.0]])
        uifield = ui(origxvecs).reshape((res, res))
        # FA temporary
        ucellres += uifield
        
        uy0 = ucellres[:, -1]
        y0 = origcellx2[-1]
        # Fourier transform to get field above unit cell
        fcoefs = 1/res*np.fft.fft(np.exp(-1j*kappa*origcellx1)*uy0)
        freqs = np.fft.fftfreq(res, d/res)
        kns = np.sqrt(k**2 - (freqs*2*np.pi + kappa)**2)
        kns = kns.real*np.where(kns.imag < 0, -1, 1) + np.abs(kns.imag)*1j
        utop = np.zeros((res, (nvcells)*(res)), dtype = complex)

        for j, y in enumerate(x2i[x2i > y0]):
            utop_oney = res*np.fft.ifft(fcoefs*np.exp(1j*kns*(y - y0)))*np.exp(1j*kappa*origcellx1)
            vecs = np.zeros((2, res))
            vecs[0] = origcellx1
            vecs[1] = y
            utop_ana = reconstruct_sol_periodic(panels, fieldN, vecs, k, Sns, alph, relcorner = True, layer = 'S')
            utop[:,j] = utop_oney
        utot = np.concatenate((ucellres, utop), axis = 1)

        for j, ncell in enumerate(ncells):
            us[j,:,i] = utot.flatten()*alph**ncell

    field = us.reshape((x1res-1, x2res-1))
    if no_return_xs == False:
        return x1coords, x2coords, field
    else:
        return field




def reconstruct_arrayscan_fromfile(panels, files, res = 10, ncells = [-1, 0, 1], nvcells = 6, deriv = False, no_return_xs = False):
    """
    Parameters
    ----------
    panels: list [Panel]
        List of Panel objects defining the boundary geometry.
    files: list [string]
        List of three files in any order:
            - list of kappa values and weights for array scanning integral
            - density at the boundary at each kappa value
            - lattice sum coefficients at each kappa value
    res: int (optional)
        Resolution of each cell will be res x res
    ncells: list [int] (optional)
        Index of which cells to plot the solution across. 0 belongs to the original cell.

    Returns
    -------
    ncells x res x rex values to be plotted.
    """
    fieldfile = [f for f in files if "sigma" in f or "fieldN" in f][0]
    snsfile = [f for f in files if "Sns" in f or "sns" in f][0]
    kappafile = [f for f in files if "kappas" in f][0]
    weightfile = [f for f in files if "weights" in f][0]
    paramfile = [f for f in files if "params" in f][0]
    print("Reading density on the boundary from", fieldfile)
    print("Reading lattice sum coeffs (Sn-s) from", snsfile)
    print("Reading kappa values for array scanning from", kappafile) 
    print("Reading in weights for array scanning from", weightfile)
    print("Reading in params for array scanning from", weightfile)

    # Read in kappa, tau/sigma on bdry, and lattice sum coeffs from file
    kappas = np.loadtxt(kappafile, dtype = complex)
    fields = np.loadtxt(fieldfile, dtype = complex)
    weights = np.loadtxt(weightfile, dtype = complex)
    allsns = np.loadtxt(snsfile, dtype = complex)
    nkappas = kappas.shape[0]
    fields = fields.reshape((nkappas, -1))
    allsns = allsns.reshape((nkappas, -1))
    paramsdata = pandas.read_csv(paramfile, sep = ', ')
    k = paramsdata['k'][0]
    print("Wavenumber value:", k)

    # Generate grid 
    d = 1.0 # TODO don't hard code this 
    x1lo, x1hi, x2lo, x2hi, x1res, x2res = d*(-0.5 + ncells[0]), d*(0.5 + ncells[-1]),\
                                           -d/2, (nvcells + 1/2)*d, res*len(ncells) + 1, (nvcells+1)*res + 1
    x1i = np.linspace(x1lo, x1hi, x1res)[:-1]
    x2i = np.linspace(x2lo, x2hi, x2res)[:-1]
    us = np.zeros((len(ncells), res*(nvcells+1)*res, nkappas), dtype = complex)
    dusx1 = np.zeros((res*res, nkappas), dtype = complex)
    dusx2 = np.zeros((res*res, nkappas), dtype = complex)
    origcellx1 = np.linspace(-d/2, d/2, res+1)[:-1]
    origcellx2 = origcellx1
    x1coords, x2coords = np.meshgrid(x1i, x2i, indexing = 'ij')
    origxvecs = np.zeros((2, res*res))
    origxvecs[0] = np.repeat(origcellx1, res)
    origxvecs[1] = np.repeat(origcellx2.reshape(-1, 1), res, axis = 1).flatten("F")

    for i in range(nkappas):
        kappa = kappas[i]
        print("kappa:", kappa)
        fieldN = fields[i,:]
        Sns = allsns[i,:]
        alph = np.exp(1j*kappa)
        # TODO: relcorner to be a property of panels? Get rid of it altogether?
        if deriv:
            ucell, ducellx1, ducellx2 = reconstruct_sol_dsol_periodic(panels, fieldN, origxvecs, k, Sns, alph, relcorner = True, layer = 'S')
            #TODO: don't forget about x2 derivative
            ducellresx1 = ducellx1.reshape((res, res))
            ducellresx2 = ducellx2.reshape((res, res))
        else:
            ucell = reconstruct_sol_periodic(panels, fieldN, origxvecs, k, Sns, alph, relcorner = True, layer = 'S')

        ucellres = ucell.reshape((res, res))
        xsource = np.array([[-0.2], [0.1]])
        ui = lambda x: Phi_p_offset(origxvecs, xsource, k, Sns, alph, self = False).flatten()
        nvec = np.array([[0.0], [1.0]])
        uifield = ui(origxvecs).reshape((res, res))
        ucellres += uifield

        uy0 = ucellres[:, -1]
        y0 = origcellx2[-1]
        # Fourier transform to get field above unit cell
        fcoefs = 1/res*np.fft.fft(np.exp(-1j*kappa*origcellx1)*uy0)
        freqs = np.fft.fftfreq(res, d/res)
        kns = np.sqrt(k**2 - (freqs*2*np.pi + kappa)**2)
        kns = kns.real*np.where(kns.imag < 0, -1, 1) + np.abs(kns.imag)*1j
        utop = np.zeros((res, (nvcells)*(res)), dtype = complex)

        for j, y in enumerate(x2i[x2i > y0]):
            utop_oney = res*np.fft.ifft(fcoefs*np.exp(1j*kns*(y - y0)))*np.exp(1j*kappa*origcellx1)
            vecs = np.zeros((2, res))
            vecs[0] = origcellx1
            vecs[1] = y
            utop_ana = reconstruct_sol_periodic(panels, fieldN, vecs, k, Sns, alph, relcorner = True, layer = 'S')
            utop[:,j] = utop_oney
        utot = np.concatenate((ucellres, utop), axis = 1)

        for j, ncell in enumerate(ncells):
            us[j,:,i] = utot.flatten()*alph**ncell
        if deriv:
            dusx1[:,i] = ducellresx1.flatten()
            dusx2[:,i] = ducellresx2.flatten()
        time1 = time.time()
#        print("Elapsed time: ", time1 - time0, " seconds")

    field = 1/(2*np.pi)*np.einsum('kij,j->ki', us, weights) 
    field = field.reshape((x1res-1, x2res-1))
    if deriv:
        dfieldx1 = 1/(2*np.pi)*np.einsum('ij,j->i', dusx1, weights)
        dfieldx2 = 1/(2*np.pi)*np.einsum('ij,j->i', dusx2, weights)
        dfieldx1 = dfieldx1.reshape((res, res))
        dfieldx2 = dfieldx2.reshape((res, res))
        return x1coords, x2coords, field, dfieldx1, dfieldx2
    elif no_return_xs == False:
        return x1coords, x2coords, field
    else:
        return field

def reconstruct_asm_fromfile_x(panels, files, xvecs, deriv = False):
    """
    Parameters
    ----------
    panels: list [Panel]
        List of Panel objects defining the boundary geometry.
    files: list [string]
        List of three files in any order:
            - list of kappa values and weights for array scanning integral
            - density at the boundary at each kappa value
            - lattice sum coefficients at each kappa value
    res: int (optional)
        Resolution of each cell will be res x res
    ncells: list [int] (optional)
        Index of which cells to plot the solution across. 0 belongs to the original cell.

    Returns
    -------
    ncells x res x rex values to be plotted.
    """
    fieldfile = [f for f in files if "fieldN" in f][0]
    snsfile = [f for f in files if "sns" in f][0]
    kappafile = [f for f in files if "kappa" in f][0]
    weightfile = [f for f in files if "weight" in f][0]
    print("Reading density on the boundary from", fieldfile)
    print("Reading lattice sum coeffs (Sn-s) from", snsfile)
    print("Reading kappa values for array scanning from", kappafile) 
    print("Reading in weights for array scanning from", weightfile)

    # Read in kappa, tau/sigma on bdry, and lattice sum coeffs from file
    kappas = np.loadtxt(kappafile, dtype = complex)
    fields = np.loadtxt(fieldfile, dtype = complex)
    weights = np.loadtxt(weightfile, dtype = complex)
    allsns = np.loadtxt(snsfile, dtype = complex)
    nkappas = kappas.shape[0]
    fields = fields.reshape((nkappas, -1))
    allsns = allsns.reshape((nkappas, -1))
#    with open(kappafile, 'r') as f:
#        k = float(f.readline().split("=")[-1])
    k = 2.4
    print("Wavenumber value:", k)

    # Generate grid 
    novecs = xvecs.shape[1]
    us = np.zeros((novecs, nkappas), dtype = complex)
    dusx1 = np.zeros((novecs, nkappas), dtype = complex)
    dusx2 = np.zeros((novecs, nkappas), dtype = complex)
    print("Starting reconstruction")
    for i in range(nkappas):
        kappa = kappas[i]
        print("kappa:", kappa)
        fieldN = fields[i,:]
        Sns = allsns[i,:]
        alph = np.exp(1j*kappa)
        # TODO: relcorner to be a property of panels? Get rid of it altogether?
        if deriv:
#            print("xvecs yvals")
#            print(xvecs[0,:])
#            print("xvecs xvals")
            ucell, ducellx1, ducellx2 = reconstruct_sol_dsol_periodic(panels, fieldN, xvecs, k, Sns, alph, relcorner = True, layer = 'S')
            #TODO: don't forget about x2 derivative
#            return 0, 0, 0

        else:
            ucell = reconstruct_sol_periodic(panels, fieldN, xvecs, k, Sns, alph, relcorner = True, layer = 'S')
        
        us[:,i] = ucell.flatten()
        dusx1[:,i] = ducellx1.flatten()
        dusx2[:,i] = ducellx2.flatten()

    field = 1/(2*np.pi)*np.einsum('ij,j->i', us, weights) 
    dfieldx1 = 1/(2*np.pi)*np.einsum('ij,j->i', dusx1, weights)
    dfieldx2 = 1/(2*np.pi)*np.einsum('ij,j->i', dusx2, weights)
    if deriv:
        return field, dfieldx1, dfieldx2
    else:
        return field




#def solve_Helm_int_Dir(ss, ws, bdry, f, k = 0.0):
#    """
#
#    """
#    N = ss.shape[0]
#    fN = f(ss) 
#    K = bie.nystromK_DLP(ss, ws, bdry.z, bdry.zp, bdry.n, bdry.kappa, k = k)
##    fig, ax = plt.subplots(1,2)
##    ax[0].imshow(K.real)
##    ax[1].imshow(K.imag)
##    plt.savefig("nystromK_helm.pdf")
#    A = np.identity(N) - K
#    tauN = np.linalg.solve(A, -2*fN)
#    return tauN   
#   
#def solve_Laplace_int_Dir(ss, ws, bdry, f):
#    """
#
#    """
#    N = ss.shape[0]
#    fN = f(ss) 
#    K = bie.nystromK_DLP(ss, ws, bdry.z, bdry.zp, bdry.n, bdry.kappa)
#    A = np.identity(N) - K
#    tauN = np.linalg.solve(A, -2*fN)
#    return tauN   
#


def reconstruct_arrayscan_fromfile_OLD(panels, files, res = 10, ncells = [-1, 0, 1], nvcells = 6, deriv = False):
    """
    Parameters
    ----------
    panels: list [Panel]
        List of Panel objects defining the boundary geometry.
    files: list [string]
        List of three files in any order:
            - list of kappa values and weights for array scanning integral
            - density at the boundary at each kappa value
            - lattice sum coefficients at each kappa value
    res: int (optional)
        Resolution of each cell will be res x res
    ncells: list [int] (optional)
        Index of which cells to plot the solution across. 0 belongs to the original cell.

    Returns
    -------
    ncells x res x rex values to be plotted.
    """
    fieldfile = [f for f in files if "fieldN" in f][0]
    snsfile = [f for f in files if "sns" in f][0]
    kappafile = [f for f in files if "kappa" in f][0]
    weightfile = [f for f in files if "weight" in f][0]
    print("Reading density on the boundary from", fieldfile)
    print("Reading lattice sum coeffs (Sn-s) from", snsfile)
    print("Reading kappa values for array scanning from", kappafile) 
    print("Reading in weights for array scanning from", weightfile)

    # Read in kappa, tau/sigma on bdry, and lattice sum coeffs from file
    kappas = np.loadtxt(kappafile, dtype = complex)
    fields = np.loadtxt(fieldfile, dtype = complex)
    weights = np.loadtxt(weightfile, dtype = complex)
    allsns = np.loadtxt(snsfile, dtype = complex)
    nkappas = kappas.shape[0]
    fields = fields.reshape((nkappas, -1))
    allsns = allsns.reshape((nkappas, -1))
#    with open(kappafile, 'r') as f:
#        k = float(f.readline().split("=")[-1])
    k = 2.4
    print("Wavenumber value:", k)

    # Generate grid 
    d = 1.0 # TODO don't hard code this 
#    nvcells = 6
    x1lo, x1hi, x2lo, x2hi, x1res, x2res = d*(-0.5 + ncells[0]), d*(0.5 + ncells[-1]),\
                                           -d/2, (nvcells + 1/2)*d, res*len(ncells) + 1, (nvcells+1)*res + 1
    x1i = np.linspace(x1lo, x1hi, x1res)[:-1]
    x2i = np.linspace(x2lo, x2hi, x2res)[:-1]
    us = np.zeros((len(ncells), res*(nvcells+1)*res, nkappas), dtype = complex)
    dusx1 = np.zeros((res*res, nkappas), dtype = complex)
    dusx2 = np.zeros((res*res, nkappas), dtype = complex)
    origcellx1 = np.linspace(-d/2, d/2, res+1)[:-1]
    origcellx2 = origcellx1
    x1coords, x2coords = np.meshgrid(x1i, x2i, indexing = 'ij')
    origxvecs = np.zeros((2, res*res))
    origxvecs[0] = np.repeat(origcellx1, res)
    origxvecs[1] = np.repeat(origcellx2.reshape(-1, 1), res, axis = 1).flatten("F")

    print("Starting reconstruction")
    for i in range(nkappas):
        kappa = kappas[i]
        print("kappa:", kappa)
        fieldN = fields[i,:]
        Sns = allsns[i,:]
        alph = np.exp(1j*kappa)
        # TODO: relcorner to be a property of panels? Get rid of it altogether?
        if deriv:
            ucell, ducellx1, ducellx2 = reconstruct_sol_dsol_periodic(panels, fieldN, origxvecs, k, Sns, alph, relcorner = True, layer = 'S')
            #TODO: don't forget about x2 derivative
            ducellresx1 = ducellx1.reshape((res, res))
            ducellresx2 = ducellx2.reshape((res, res))

        else:
            ucell = reconstruct_sol_periodic(panels, fieldN, origxvecs, k, Sns, alph, relcorner = True, layer = 'S')

#        print("orig x vecs, x1:", origxvecs[0,:res])
#        print("orig x vecs, x2:", origxvecs[1,:res])
#        print("density:", fieldN[:5])
        ucellres = ucell.reshape((res, res))
        xsource = np.array([[-0.2], [0.1]])
        ui = lambda x: Phi_p_offset(origxvecs, xsource, k, Sns, alph, self = False).flatten()
        nvec = np.array([[0.0], [1.0]])
#        dui = lambda nvec, x: dPhi_p_dn_offset(x, xsource, nvec, k, Sns, alph, nxory = 'x', self = False).flatten()
#        duifield = dui(nvec, origxvecs)
        uifield = ui(origxvecs).reshape((res, res))
        ucellres += uifield
#        print("duifield:", duifield[:5])
       # print("ui:", uifield[:5, :5])
#        origx, origy = np.meshgrid(origcellx1, origcellx2, indexing = 'ij')
#        fig, ax = plt.subplots(2, 1)
##        CS1 = ax[0].pcolormesh(origx, origy, (ucellres+uifield).real, cmap = 'jet')
##        CS2 = ax[1].pcolormesh(origx, origy, (ucellres+uifield).imag, cmap = 'jet')
#        CS1 = ax[0].contourf(origx, origy, (ucellres+uifield).real, cmap = 'jet', levels = np.linspace(-0.5, 0.5, 50))
#        CS2 = ax[1].contourf(origx, origy, (ucellres+uifield).imag, cmap = 'jet', levels = np.linspace(-0.5, 0.5, 50))
#
#        fig.colorbar(CS1)
#        fig.colorbar(CS2)
#        ax[0].set_aspect('equal')
#        ax[1].set_aspect('equal')
#        plt.show()
#        return 0


        uy0 = ucellres[:, -1]
        y0 = origcellx2[-1]
        # Fourier transform to get field above unit cell
        fcoefs = 1/res*np.fft.fft(np.exp(-1j*kappa*origcellx1)*uy0)
        freqs = np.fft.fftfreq(res, d/res)
        kns = np.sqrt(k**2 - (freqs*2*np.pi + kappa)**2)
        kns = kns.real*np.where(kns.imag < 0, -1, 1) + np.abs(kns.imag)*1j
#        print("y0:", y0)
#        print("u_y0_kappa:", uy0)
#        print("ms:", freqs)
#        print("cm_kappa:", fcoefs)
#        print("kappa m-s:", kns) 
        utop = np.zeros((res, (nvcells)*(res)), dtype = complex)
#        print(utop.shape, len(x2i[x2i > y0]))
#        print(y0, x2i)

        for j, y in enumerate(x2i[x2i > y0]):
            utop_oney = res*np.fft.ifft(fcoefs*np.exp(1j*kns*(y - y0)))*np.exp(1j*kappa*origcellx1)
            vecs = np.zeros((2, res))
            vecs[0] = origcellx1
            vecs[1] = y
            utop_ana = reconstruct_sol_periodic(panels, fieldN, vecs, k, Sns, alph, relcorner = True, layer = 'S')
            utop[:,j] = utop_oney
        utot = np.concatenate((ucellres, utop), axis = 1)

        for j, ncell in enumerate(ncells):
            us[j,:,i] = utot.flatten()*alph**ncell
        if deriv:
            dusx1[:,i] = ducellresx1.flatten()
            dusx2[:,i] = ducellresx2.flatten()

    field = 1/(2*np.pi)*np.einsum('kij,j->ki', us, weights) 
    field = field.reshape((x1res-1, x2res-1))
    if deriv:
        dfieldx1 = 1/(2*np.pi)*np.einsum('ij,j->i', dusx1, weights)
        dfieldx2 = 1/(2*np.pi)*np.einsum('ij,j->i', dusx2, weights)
        dfieldx1 = dfieldx1.reshape((res, res))
        dfieldx2 = dfieldx2.reshape((res, res))
        return x1coords, x2coords, field, dfieldx1, dfieldx2
    else:
        return x1coords, x2coords, field






