#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.special as sp
import bie.bie as bie
import copy

def circle_path(poleloc, p = 16, eps = 0.1):
          
    # Trap on [-1, 1]
    trap_nodes = np.linspace(-1, 1, p).reshape((1, p))
    trap_weights = np.ones((1, p))*2/(p-1)
    trap_weights[0, 0] /= 2
    trap_weights[0,-1] /= 2

    # Panel around pole
    nodespole = poleloc + eps*np.exp(1j*(np.pi*trap_nodes + np.pi))
    nodespole2d = np.zeros((2, p))
    nodespole2d[0] = nodespole.real
    nodespole2d[1] = nodespole.imag
    zpvecpole = eps*1j*np.pi*np.exp(1j*(np.pi*trap_nodes + np.pi))
    zpvecpole2d = np.zeros((2, p))
    zpvecpole2d[0] = zpvecpole.real
    zpvecpole2d[1] = zpvecpole.imag
    zppvec = np.zeros((2, p))
    panelpole = bie.panel(nodespole2d, zpvecpole2d, zppvec, trap_weights, 2, [])
    
    return([panelpole])

def half_circle_path(poleloc, p = 16, eps = 0.1):
          
    # Trap on [-1, 1]
    trap_nodes = np.linspace(-1, 1, p).reshape((1, p))
    trap_weights = np.ones((1, p))*2/(p-1)
    trap_weights[0, 0] /= 2
    trap_weights[0,-1] /= 2

    # Panel around pole
    nodespole = poleloc + eps*np.exp(1j*(np.pi/2*trap_nodes - np.pi/2))
    nodespole2d = np.zeros((2, p))
    nodespole2d[0] = nodespole.real
    nodespole2d[1] = nodespole.imag
    zpvecpole = eps*1j*np.pi/2*np.exp(1j*(np.pi/2*trap_nodes - np.pi/2))
    zpvecpole2d = np.zeros((2, p))
    zpvecpole2d[0] = zpvecpole.real
    zpvecpole2d[1] = zpvecpole.imag
    zppvec = np.zeros((2, p))
    panelpole = bie.panel(nodespole2d, zpvecpole2d, zppvec, trap_weights, 2, [])
    
    return([panelpole])

def sine_path(A = 0.1, p = 16, quad = 'trap'):

    trap_nodes = np.linspace(-1, 1, p).reshape((1, p))
    trap_weights = np.ones((1, p))*2/(p-1)
    trap_weights[0, 0] /= 2
    trap_weights[0,-1] /= 2
    nodes2d = np.zeros((2, p))
    nodes2d[0] = trap_nodes*np.pi
    nodes2d[1] = -A*np.sin(nodes2d[0])
    zpvec2d = np.zeros((2, p))
    zpvec2d[0] = np.pi
    zpvec2d[1] = -A*np.pi*np.cos(nodes2d[0])
    zppvec = np.zeros((2, p))
    panel = bie.panel(nodes2d, zpvec2d, zppvec, trap_weights, 0, [])
    return([panel])
 
def perispecint(xp):
    """
    Taken from: https://github.com/ahbarnett/DPLS-demos/blob/master/singleinclusion/utils/perispecint.m
    """
    f = np.copy(xp)
    N = f.shape[0]
    fbar = np.mean(f); f -= fbar
    arr = np.zeros(N, dtype = complex)
    if N%2 == 0: 
        # even no. of samples
        arr[1:(N//2)] = 1/(1j*np.array([i for i in range(1,N//2)], dtype = complex)) 
        arr[-(N//2-1):] = 1/(1j*np.array([i for i in range(-N//2+1,0)], dtype = complex)) 
        g = np.fft.ifft(np.fft.fft(f)*arr, axis = 0)
    else:
        arr[1:(N-1)//2+1] = 1/(1j*np.array([i for i in range(1,(N-1)//2+1)], dtype = complex))
        arr[(N-1)//2+1:] = 1/(1j*np.array([i for i in range((1-N)//2,0)], dtype = complex)) 
        g = np.fft.ifft(np.fft.fft(f)*arr, axis = 0)
    g += np.array([i for i in range(1,N+1)], dtype = complex)*fbar*2*np.pi/N
    return g


def sine_path_expgrad_pi(A = 0.1, p = 16, beta = 1.0):
    """
    Expontential grading taken from:
    https://github.com/ahbarnett/DPLS-demos/blob/master/singleinclusion/utils/reparam_bunched.m
    """
    import matplotlib.pyplot as plt
    xpfun = lambda t: np.cosh(beta*np.sin(t/2))
    # p equally spaced trapezoidal nodes
    trap_nodes = np.pi*np.linspace(0, 2, p+1)[1:] # reshape later
    trap_weights = np.pi*np.ones((1, p))*2/p
    xp = xpfun(trap_nodes) 
    h = 2*np.pi/p
    I = sum(xp)*h
    aI = 2*np.pi/I
    xp *= aI # normalization
#    plt.figure()
#    plt.plot(range(p), xp, '.', color = 'C1')
#    plt.show()
    x = perispecint(xp)
    x -= (x[0] + np.pi) # start at -pi
    x = x.reshape((1, p))
    xp = xp.reshape((1, p))
    nodes2d = np.zeros((2, p))
    nodes2d[0] = x
    nodes2d[1] = -A*np.sin(nodes2d[0])
    zpvec2d = np.zeros((2, p))
    zpvec2d[0] = xp
    zpvec2d[1] = -A*np.cos(nodes2d[0])*xp
    zppvec = np.zeros((2, p))
    panel = bie.panel(nodes2d, zpvec2d, zppvec, trap_weights, 0, [])
    return([panel])

def sine_path_expgrad_zero(A = 0.1, p = 16, beta = 1.0):
    """
    Expontential grading taken from:
    https://github.com/ahbarnett/DPLS-demos/blob/master/singleinclusion/utils/reparam_bunched.m
    """
#    import matplotlib.pyplot as plt
    xpfun = lambda t: np.cosh(beta*np.sin((t-np.pi)/2))
    # p equally spaced trapezoidal nodes
    trap_nodes = np.linspace(0, 2*np.pi, p+1)[1:] # reshape later
    trap_weights = np.pi*np.ones((1, p))*2/p
    xp = xpfun(trap_nodes) 
    h = 2*np.pi/p
    I = sum(xp)*h
    aI = 2*np.pi/I
    xp *= aI # normalization
    x = perispecint(xp)
    x -= (x[-1] - np.pi) # start at -pi
    x = x.reshape((1, p))
#    print("min spacing:", min(abs(np.diff(x))[0]), "max spacing:", max(abs(np.diff(x))[0]), "avg:", 2*np.pi/p)
#    print("spacings:", abs(np.diff(x)))
    xp = xp.reshape((1, p))
    nodes2d = np.zeros((2, p))
    nodes2d[0] = x
    nodes2d[1] = -A*np.sin(nodes2d[0])
    zpvec2d = np.zeros((2, p))
    zpvec2d[0] = xp
    zpvec2d[1] = -A*np.cos(nodes2d[0])*xp
    zppvec = np.zeros((2, p))
    panel = bie.panel(nodes2d, zpvec2d, zppvec, trap_weights, 0, [])
    return([panel])
 


def least_absorption_path(woodloc, poleloc, psqrt = 16, ppole = 16, ptrap = 16, eps = 0.1):
          
    # Gauss-Legendre on [-1, 1]
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(psqrt)
    gauss_nodes = gauss_nodes.reshape((1, psqrt))
    gauss_weights = gauss_weights.reshape((1, psqrt))
    # Trap on [-1, 1]
    trap_nodes = np.linspace(-1, 1, ptrap).reshape((1, ptrap))
    trap_weights = np.ones((1, ptrap))*2/(ptrap-1)
    trap_weights[0, 0] /= 2
    trap_weights[0,-1] /= 2
    # Trap nodes around pole [-1, 1]
    trap_nodes_pole = np.linspace(-1, 1, ptrap).reshape((1, ppole))
    trap_weights_pole = np.ones((1, ppole))*2/(ppole-1)
    trap_weights_pole[0, 0] /= 2
    trap_weights_pole[0,-1] /= 2
    # Sqrt on [0, 1]
    sqrt_nodes = (1 + gauss_nodes)/2
    sqrt_weights = gauss_weights*sqrt_nodes
    sqrt_nodes *= sqrt_nodes
    
    # Panel from 0 to Wood anomaly with sqrt nodes
    nodes0wood = sqrt_nodes*(-woodloc) + woodloc
    nodes0wood[0,:] = nodes0wood[0,::-1]
    nodes0wood2d = np.zeros((2, psqrt))
    nodes0wood2d[0] = nodes0wood.real
    nodes0wood2d[1] = nodes0wood.imag
    zpvec0wood = np.ones((2, psqrt))
    zpvec0wood[0] *= woodloc.real
    zpvec0wood[1] *= woodloc.imag
    zppvec = np.zeros((2, psqrt))
    woodweights = np.zeros((1, psqrt))
    woodweights[0] = sqrt_weights[0,::-1]
    panel0wood = bie.panel(nodes0wood2d, zpvec0wood, zppvec, woodweights, 4, []) 

    # Panel from Wood anomaly to pole - eps with sqrt nodes
    nodeswoodpole = sqrt_nodes*(poleloc - eps - woodloc) + woodloc
#    nodeswoodpole = sqrt_nodes*(poleloc - woodloc) + woodloc
    nodeswoodpole2d = np.zeros((2, psqrt))
    nodeswoodpole2d[0] = nodeswoodpole.real
    nodeswoodpole2d[1] = nodeswoodpole.imag
    zpvecwoodpole = np.ones((2, psqrt))
    zpvecwoodpole[0] *= (poleloc - eps - woodloc).real
    zpvecwoodpole[1] *= (poleloc - eps - woodloc).imag
#    zpvecwoodpole[0] *= (poleloc - woodloc).real
#    zpvecwoodpole[1] *= (poleloc - woodloc).imag
    panelwoodpole = bie.panel(nodeswoodpole2d, zpvecwoodpole, zppvec, sqrt_weights, 5, [])

    # Panel around pole with ppole trap nodes
    nodespole = poleloc + eps*np.exp(1j*(np.pi/2*trap_nodes - np.pi/2))
    nodespole2d = np.zeros((2, ppole))
    nodespole2d[0] = nodespole.real
    nodespole2d[1] = nodespole.imag
    zpvecpole = eps*1j*np.pi/2*np.exp(1j*(np.pi/2*trap_nodes - np.pi/2))
    zpvecpole2d = np.zeros((2, ppole))
    zpvecpole2d[0] = zpvecpole.real
    zpvecpole2d[1] = zpvecpole.imag
    panelpole = bie.panel(nodespole2d, zpvecpole2d, zppvec, trap_weights, 2, [])
    
    # Panel from pole + eps to pi with trap nodes
    nodespolepi = (trap_nodes + 1)/2*(np.pi - poleloc - eps) + poleloc + eps
#    nodespolepi = (trap_nodes + 1)/2*(np.pi - poleloc) + poleloc
    nodespolepi2d = np.zeros((2, ptrap))
    nodespolepi2d[0] = nodespolepi.real
    nodespolepi2d[1] = nodespolepi.imag
    zpvecpolepi = np.ones((2, ptrap))
    zpvecpolepi[0] *= ((np.pi - poleloc- eps)/2).real
    zpvecpolepi[1] *= ((np.pi - poleloc- eps)/2).imag
#    zpvecpolepi[0] *= ((np.pi - poleloc)/2).real
#    zpvecpolepi[1] *= ((np.pi - poleloc)/2).imag
    zppvec = np.zeros((2, ptrap))
    panelpi = bie.panel(nodespolepi2d, zpvecpolepi, zppvec, trap_weights, 6, []) 
    
    # These panels, but reversed to cover [-pi, 0]
    panelmpi = copy.deepcopy(panelpi)
    panelmpi.nodes *= -1
    panelmpi.nodes[0] = panelmpi.nodes[0,::-1]
    panelmpi.weights[0] = panelmpi.weights[0,::-1]
    panelmpi.zpvec[0] = panelmpi.zpvec[0,::-1]
    panelmpi.index = 0
    panelmpole = copy.deepcopy(panelpole)
    panelmpole.nodes *= -1
    panelmpole.nodes[0] = panelmpole.nodes[0,::-1]
    panelmpole.weights[0] = panelmpole.weights[0,::-1]
    panelmpole.zpvec[0] = panelmpole.zpvec[0,::-1]
    panelmpole.zpvec[1] = panelmpole.zpvec[1,::-1]
    panelmpole.index = 1
    panelmwoodpole = copy.deepcopy(panelwoodpole)
    panelmwoodpole.nodes *= -1
    panelmwoodpole.nodes[0] = panelmwoodpole.nodes[0,::-1]
    panelmwoodpole.weights[0] = panelmwoodpole.weights[0,::-1]
    panelmwoodpole.zpvec[0] = panelmwoodpole.zpvec[0,::-1]
    panelmwoodpole.index = 2
    panelm0wood = copy.deepcopy(panel0wood)
    panelm0wood.nodes *= -1
    panelm0wood.nodes[0] = panelm0wood.nodes[0,::-1]
    panelm0wood.weights[0] = panelm0wood.weights[0,::-1]
    panelm0wood.zpvec[0] = panelm0wood.zpvec[0,::-1]
    panelm0wood.index = 3

    return([panelmpi, panelmpole, panelmwoodpole, panelm0wood,\
            panel0wood, panelwoodpole, panelpole, panelpi])
#    return([panelmpi, panelmwoodpole, panelm0wood,\
#            panel0wood, panelwoodpole, panelpi])

def single_panel(p, d = 1.0):
     
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_nodes = unsc_nodes.reshape((1, p))
    unsc_weights = unsc_weights.reshape((1, p))
    nodes = np.zeros((2, p))
    nodes[0] = unsc_nodes*d/2
    zpvec = np.ones((2, p))*d/2
    zpvec[1] = 0.0
    endpoints = np.array([[-d/2, d/2], [0.0, 0.0]])
    origin = np.zeros((2, 1))
    index = 0
    neighbors = []
    corneradj = np.array([False, False])
    colinears = []
    zppvec = np.zeros((2, p))
    panel = bie.panel(nodes, zpvec, zppvec, unsc_weights, index, neighbors,\
                      colinears = colinears, corneradj = corneradj,\
                      endpoints = endpoints, origin = origin)
    panel.n *= -1
    return [panel]
  

def gen_staircase(n, p, d, beta, nperside = 1):
    """
    Generates n=3 copies of the unit cell of a periodic staircase. The opening
    angle of each step is beta, the width of the unit cell is d. There will be nperside equal-size panels per
    side, with p quadrature nodes on each.
    """
    verty = d/2.0/np.tan(beta/2)
    verts = np.zeros((2, 1 + 2*n))
    verts[0] = np.linspace(-n/2*d, n/2*d, 1 + 2*n)
    verts[1,1::2] = verty
    verts[1,:] -= d/2
#    verts = np.array([[-1.5*d, -d, -0.5*d, 0.0, 0.5*d, d, 1.5*d],\
#                      [0.0, verty, 0.0, verty, 0.0, verty, 0.0]])
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_nodes = unsc_nodes.reshape((1, p))
    unsc_weights = unsc_weights.reshape((1, p))
    panels = []
    vert0 = verts[:,0].reshape((2, 1)) 
    for i in range(1, verts.shape[1]):
        vert1 = verts[:,i].reshape((2, 1))
        vert0side = vert0
        for j in range(nperside):
            index = (i-1)*nperside + j
            if index == 0:
                neighbors = [1]
            elif index == (2*nperside*n - 1):
                neighbors = [index-1]
            else:
                neighbors = [index - 1, index + 1]
            vert1side = vert0 + (vert1 - vert0)*(j+1)/nperside
            zppvec = np.zeros((2, p))
            origin = np.zeros((2, 1))
            colinears = [l for l in range((i-1)*nperside, i*nperside)]
            corneradj = np.array([False, False])
            # If panel is adjacent to a corner, have rel coords
            if j == 0:
                origin = vert0
                nodes = (unsc_nodes + 1)/2*(vert1side - vert0)
                zpvec = np.ones((2, p))*(vert1side - vert0)/2
                endpoints = np.concatenate((np.zeros((2, 1)), vert1side - vert0), axis = 1)
                if nperside == 1:
                    corneradj = np.array([True, True])
                else:
                    corneradj = np.array([True, False])
            elif j == (nperside - 1):
                origin = vert1
                nodes = (unsc_nodes + 1)/2*(vert1 - vert0side) + (vert0side - vert1)
                zpvec = np.ones((2, p))*(vert1 - vert0side)/2
                corneradj = np.array([False, True])
                endpoints = np.concatenate((vert0side - vert1, np.zeros((2, 1))), axis = 1)
            # Otherwise just normal coords
            else: 
                nodes = vert0side + (vert1side - vert0side)*(unsc_nodes + 1)/2
                zpvec = np.ones((2, p))*(vert1side - vert0side)/2
                endpoints = np.concatenate((vert0side, vert1side), axis = 1)
            panel = bie.panel(nodes, zpvec, zppvec, unsc_weights, index, neighbors,\
                               colinears = colinears, corneradj = corneradj,\
                               endpoints = endpoints, origin = origin)
            # Make sure normal points up
            panel.n *= -1
            panels.append(panel)
            vert0side = vert1side
        vert0 = vert1

    newpanels = bie.sort_panels(panels)
    return newpanels


def gen_staircase2(n, p, d, beta, nperside = 1):
    """
    Generates n copies of the unit cell of a periodic staircase. The opening
    angle of each step is beta, the width of the unit cell is d. There will be nperside equal-size panels per
    side, with p quadrature nodes on each.
    """
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_nodes = unsc_nodes.reshape((1, p))
    unsc_weights = unsc_weights.reshape((1, p))
    panels = []
    endl = np.array([[-d/2], [0.0]])
    endr = np.array([[d/2], [0.0]])
    cornerl = np.array([[-d/4], [-d/4*np.sqrt(1/np.sin(beta/2)**2 - 1)]])
    cornerr = np.array([[d/4], [d/4*np.sqrt(1/np.sin(beta/2)**2 - 1)]])
    for i in range(nperside//2):
        origin = cornerl
        corneradj = np.array([False, False])
        zppvec = np.zeros((2, p))
        if i == 0:
            # Corner-adjacent panels at left-hand corner
            nodes1 = (unsc_nodes + 1)/2*(cornerl - endl)/(nperside/2) + (endl - cornerl)/(nperside/2)
            zpvec1 = np.ones((2, p))*(-1/2)*(endl - cornerl)/(nperside/2)
            endpoints1 = np.concatenate(((endl - cornerl)/(nperside/2), np.zeros((2, 1))), axis = 1)
            if i != (nperside//2 - 1):
                neighbours = [nperside/2-i-2, nperside/2-i]
            else:
                print("found side")
                neighbours = [1]
            panel1 = bie.panel(nodes1, zpvec1, zppvec, unsc_weights, nperside/2-1, neighbours,\
                              colinears = [j for j in range(nperside//2)], corneradj = np.array([False, True]),\
                              endpoints = endpoints1, origin = origin)            
            nodes2 = (unsc_nodes + 1)/2*(- cornerl)/(nperside/2)
            zpvec2 = np.ones((2, p))*(-cornerl)/2/(nperside/2)
            endpoints2 = np.concatenate((np.zeros((2, 1)), -cornerl/(nperside/2)), axis = 1)
            panel2 = bie.panel(nodes2, zpvec2, zppvec, unsc_weights, nperside/2, [nperside/2-1, nperside/2+1],\
                              colinears = [j for j in range(nperside//2, 3*nperside//2)],\
                              corneradj = np.array([True, False]), endpoints = endpoints2, origin = origin)
            panels.append(panel1)
            panels.append(panel2)
        else:
            # All other panels on the left side
            vertl = (endl - cornerl)/(nperside/2)*(i+1)
            vertr = (endl - cornerl)/(nperside/2)*i
            nodes1 = (unsc_nodes + 1)/2*(cornerl - endl)/(nperside/2) + vertl
            zpvec1 = np.ones((2, p))/2*(cornerl - endl)/(nperside/2)
            endpoints1 = np.concatenate((vertl, vertr), axis = 1)
            if i != (nperside//2 - 1):
                neighbours = [nperside/2-i-2, nperside/2-i]
            else:
                print("found side")
                neighbours = [1]
            panel1 = bie.panel(nodes1, zpvec1, zppvec, unsc_weights, nperside/2-i-1,\
                              neighbours,\
                              colinears = [j for j in range(nperside//2)], corneradj = corneradj,\
                              endpoints = endpoints1, origin = origin)            
            nodes2 = (unsc_nodes + 1)/2*(- cornerl)/(nperside/2) -cornerl/(nperside/2)*i
            zpvec2 = np.ones((2, p))*(-cornerl)/2/(nperside/2)
            endpoints2 = np.concatenate((-cornerl/(nperside/2)*i, -cornerl/(nperside/2)*(i+1)), axis = 1)
            panel2 = bie.panel(nodes2, zpvec2, zppvec, unsc_weights, nperside/2+i,\
                              [nperside/2+i-1, nperside/2+i+1],\
                              colinears = [j for j in range(nperside//2, 3*nperside//2)],\
                              corneradj = corneradj, endpoints = endpoints2, origin = origin)
            panels.append(panel1)
            panels.append(panel2)
    # Right side
    for i in range(nperside//2):
        origin = cornerr
        corneradj = np.array([False, False])
        zppvec = np.zeros((2, p))
        if i == 0:

            # Corner-adjacent panels at right-hand corner
            nodes1 = (unsc_nodes + 1)/2*(endr - cornerr)/(nperside/2)
            zpvec1 = np.ones((2, p))/2*(endr - cornerr)/(nperside/2)
            endpoints1 = np.concatenate((np.zeros((2, 1)), (endr - cornerr)/(nperside/2)), axis = 1)
            if i != (nperside//2 - 1):
                neighbours = [3/2*nperside+i-1, 3/2*nperside+i+1]
            else:
                print("found side")
                neighbours = [2*nperside-2]
            panel1 = bie.panel(nodes1, zpvec1, zppvec, unsc_weights, 3/2*nperside,\
                              neighbours,\
                              colinears = [j for j in range(3*nperside//2, 2*nperside)],\
                              corneradj = np.array([True, False]),\
                              endpoints = endpoints1, origin = origin)            
            nodes2 = (unsc_nodes + 1)/2*(cornerr)/(nperside/2) -cornerr/(nperside/2)
            zpvec2 = np.ones((2, p))*cornerr/2/(nperside/2)
            endpoints2 = np.concatenate((-cornerr/(nperside/2), np.zeros((2, 1))), axis = 1)
            panel2 = bie.panel(nodes2, zpvec2, zppvec, unsc_weights, 3/2*nperside-1,\
                              [3/2*nperside-2, 3/2*nperside],\
                              colinears = [j for j in range(nperside//2, 3*nperside//2)],\
                              corneradj = np.array([False, True]), endpoints = endpoints2, origin = origin)
            panels.append(panel1)
            panels.append(panel2)
        else:
            # All other panels on the right side
            vertl = (endr - cornerr)/(nperside/2)*i
            vertr = (endr - cornerr)/(nperside/2)*(i+1)
            nodes1 = (unsc_nodes + 1)/2*(endr - cornerr)/(nperside/2) + vertl
            zpvec1 = np.ones((2, p))/2*(endr - cornerr)/(nperside/2) 
            endpoints1 = np.concatenate((vertl, vertr), axis = 1)
            if i != (nperside//2 - 1):
                neighbours = [3/2*nperside+i-1, 3/2*nperside+i+1]
            else:
                print("found side")
                neighbours = [2*nperside-2]
            panel1 = bie.panel(nodes1, zpvec1, zppvec, unsc_weights, 3/2*nperside+i,\
                              neighbours,\
                              colinears = [j for j in range(3*nperside//2, nperside*2)], corneradj = corneradj,\
                              endpoints = endpoints1, origin = origin)            
            nodes2 = (unsc_nodes + 1)/2*cornerr/(nperside/2) -cornerr/(nperside/2)*(i+1)
            zpvec2 = np.ones((2, p))*cornerr/2/(nperside/2)
            endpoints2 = np.concatenate((-cornerr/(nperside/2)*i, -cornerr/(nperside/2)*(i+1)), axis = 1)
            panel2 = bie.panel(nodes2, zpvec2, zppvec, unsc_weights,\
                              3/2*nperside-i-1, [3/2*nperside-i-2, 3/2*nperside-i],\
                              colinears = [j for j in range(nperside//2, 3*nperside//2)],\
                              corneradj = corneradj, endpoints = endpoints2, origin = origin)
            panels.append(panel1)
            panels.append(panel2)
    # Sort the panels
    retindex = lambda p: p.index
    panels.sort(key=retindex) 
    return panels


def gen_polygon(n, p, nperside = 1):
    """
    Generates an n-sided polynomial, with nperside panels on each side, and
    each panel having p Gauss--Legendre nodes.
    TODO: corners for refinement
    """
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_nodes = unsc_nodes.reshape((1, p))
    unsc_weights = unsc_weights.reshape((1, p))
    vert0 = np.array([[1], [0]])
    panels = []
    for i in range(n):
        vert1 = np.array([[np.cos(2*(i+1)*np.pi/n)], [np.sin(2*(i+1)*np.pi/n)]])
        vert0side = vert0
        # neighbours: everyone on the same side is a neighbour
        neighbors = [k for k in range(i*nperside, (i+1)*nperside)]
        for j in range(nperside):
            vert1side = vert0 + (vert1 - vert0)*(j+1)/nperside
            # nodes
            nodes = vert0side + (vert1side - vert0side)*(unsc_nodes + 1)/2
            # zpvec
            zpvec = np.ones((2, p))*(vert1side - vert0side)/2
            # zppvec
            zppvec = np.zeros((2, p))
            # build panel
            panel = bie.panel(nodes, zpvec, zppvec, unsc_weights, i*nperside+j, neighbors)
            panels.append(panel)
            vert0side = vert1side
        vert0 = vert1
    return panels

def gen_star(n, p, depth, nperside = 1):
    """
    Generates an n-vertex star, with nperside panels on each side, and
    each panel having p Gauss--Legendre nodes.
    TODO: corners
    """
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_nodes = unsc_nodes.reshape((1, p))
    unsc_weights = unsc_weights.reshape((1, p))
    vert0 = np.array([[1], [0]])
    panels = []
    for i in range(2*n):
        vert1 = np.array([[np.cos(2*(i+1)*np.pi/(2*n))], [np.sin(2*(i+1)*np.pi/(2*n))]])
        if i%2 == 0:
            vert1 *= depth
        vert0side = vert0
        # colinears: everyone on the same side is colinear 
        colinears = [k for k in range(i*nperside, (i+1)*nperside)]
        for j in range(nperside):
            vert1side = vert0 + (vert1 - vert0)*(j+1)/nperside
            # nodes
            nodes = vert0side + (vert1side - vert0side)*(unsc_nodes + 1)/2
            # zpvec
            zpvec = np.ones((2, p))*(vert1side - vert0side)/2
            # zppvec
            zppvec = np.zeros((2, p))
            # build panel
            neighbors = [(i*nperside+j-1)%(2*n*nperside), (i*nperside+j+1)%(2*n*nperside)]
            endpoints = np.concatenate((vert0side, vert1side), axis = 1)
            corneradj = np.array([False, False])
            if j == 0:
                corneradj[0] = True
            if j == nperside - 1:
                corneradj[1] = True
            panel = bie.panel(nodes, zpvec, zppvec, unsc_weights, i*nperside+j, neighbors,\
                              colinears = colinears, corneradj = corneradj, endpoints = endpoints)
            panels.append(panel)
            vert0side = vert1side
        vert0 = vert1
    return panels


def gen_star_relcoords(n, p, depth, nperside = 1):
    """
    Generates an n-vertex star, with nperside panels on each side, and
    each panel having p Gauss--Legendre nodes.
    Panels neighbouring corners have the nearest corner as their origin.
    """
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_nodes = unsc_nodes.reshape((1, p))
    unsc_weights = unsc_weights.reshape((1, p))
    vert0 = np.array([[1], [0]])
    panels = []
    for i in range(2*n):
#        print("index: ", i)
        vert1 = np.array([[np.cos(2*(i+1)*np.pi/(2*n))], [np.sin(2*(i+1)*np.pi/(2*n))]])
        if i%2 == 0:
            vert1 *= depth
        vert0side = vert0
        # colinears: everyone on the same side is colinear 
        colinears = [k for k in range(i*nperside, (i+1)*nperside)]
        for j in range(nperside):
#            print("j: ", j)
            vert1side = vert0 + (vert1 - vert0)*(j+1)/nperside
            corneradj = np.array([False, False])
            if j == 0:
#                print("corner")
                corneradj[0] = True
                origin = vert0side
                nodes = (unsc_nodes + 1)/2*(vert1side - vert0side)
                endpoints = np.concatenate((np.zeros((2, 1)), vert1side - vert0side), axis = 1)
            elif j == nperside - 1:
#                print("corner")
                corneradj[1] = True
                origin = vert1side
                nodes = (unsc_nodes + 1)/2*(vert1side - vert0side) + (vert0side - vert1side)
                endpoints = np.concatenate((vert0side - vert1side, np.zeros((2, 1))), axis = 1)
            else:
                origin = np.zeros((2, 1))
                nodes = vert0side + (vert1side - vert0side)*(unsc_nodes + 1)/2
                endpoints = np.concatenate((vert0side, vert1side), axis = 1)
            zpvec = np.ones((2, p))*(vert1side - vert0side)/2
            zppvec = np.zeros((2, p))
            neighbors = [(i*nperside+j-1)%(2*n*nperside), (i*nperside+j+1)%(2*n*nperside)]
            panel = bie.panel(nodes, zpvec, zppvec, unsc_weights, i*nperside+j, neighbors,\
                              colinears = colinears, corneradj = corneradj, endpoints = endpoints, origin = origin)
            panels.append(panel)
            vert0side = vert1side
        vert0 = vert1
    return panels




def gen_from_polar(n, p, R, Rp, Rpp):
    """
    Generates a closed curve defined by the polar function R(theta) and its
    derivatives. The curve is parametrized by the arc length s, which is
    2pi-periodic. The curve is then divided into n equi-s panels, each with p
    nodes.
    """
    z = lambda s: np.array([R(s)*np.cos(s), R(s)*np.sin(s)])
    zpvec = lambda s: np.array([Rp(s)*np.cos(s) - R(s)*np.sin(s), Rp(s)*np.sin(s) + R(s)*np.cos(s)])
    zppvec = lambda s: np.array([Rpp(s)*np.cos(s) - 2*Rp(s)*np.sin(s) - R(s)*np.cos(s),\
                                 Rpp(s)*np.sin(s) + 2*Rp(s)*np.cos(s) - R(s)*np.sin(s)])
    panels = []
    s_lo = 0.0
    for i in range(n):
        unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
        s_hi = 2*np.pi*(i+1)/n
        # nodes
        snodes = s_lo + (s_hi - s_lo)*(unsc_nodes + 1)/2
        sweights = unsc_weights*(s_hi - s_lo)/2
        sweights = sweights.reshape((1, p))
        nodes = z(snodes)
        # zpvec
        zpvec_arr = zpvec(snodes) 
        # zppvec
        zppvec_arr = zppvec(snodes)
        # build panel
        panel = bie.panel(nodes, zpvec_arr, zppvec_arr, sweights, i, ())
        panels.append(panel)
        s_lo = s_hi

    return panels

def refine(panels, subdiv = 1, rparam = 2):
    """
    Searches for corner-adjacent panels and subdivides them `subdiv` times
    using `rparam`-adic refinement.
    """
    indices = [panel.index for panel in panels]
    newpanels = []
    newindices = []
    brokenneighbs = []
    maxindex = max(indices)
    oldmax = maxindex + 1
    for panel in panels:
        p = panel.nodes.shape[1]
        if panel.corneradj.sum() == 1:
            # This panel will be split
            # Find the neighboring panel that is closest to the corner
#            print("current panel ",panel.index)
#            print("neighbors:",panel.neighbors)
            corner = panel.endpoints[:, panel.corneradj]
#            print("corner:",corner)
            neighborp = panels[indices.index(panel.neighbors[0])]
            if True in neighborp.corneradj:
#                print("cornerneighbor guess:",neighborp.index)
                neighborscorner = neighborp.endpoints[:, neighborp.corneradj]
#                print("neighbors endpoints:",neighborp.endpoints)
#                print("neighbors corneradj:", neighborp.corneradj)
#                print("of which corner:", neighborp.endpoints[:, neighborp.corneradj])
#                print(np.abs(neighborscorner - corner).sum())
                if np.abs(neighborscorner - corner).sum() < 5e-16:
                    cornerneighbori = neighborp.index
#                    print("guess was correct")
                else:
                    cornerneighbori = panel.neighbors[1]
                    neighborp = panels[indices.index(panel.neighbors[1])]
            else:
                cornerneighbori = panel.neighbors[1]
                neighborp = panels[indices.index(panel.neighbors[1])]
            # Generate new panels
            paneltosplit = panel
            colinears = panel.colinears
            for s in range(subdiv):
                panel1, panel2 = split_linear_panel(paneltosplit, rparam, maxindex + 1, (p, p), cornerneighbori)
                maxindex += 1
                paneltosplit = panel2
                newpanels.append(panel1)
                newindices.append(panel1.index)
                colinears += [maxindex]
                if s == subdiv - 1:
                    newpanels += [panel2]
                    newindices += [panel2.index]
                    brokenneighbs.append(panel2)
            # Update colinears
            for colinindex in panel.colinears:
                if colinindex in newindices:
                    newpanels[newindices.index(colinindex)].colinears = colinears
                elif colinindex in indices:
                    panels[indices.index(colinindex)].colinears = colinears
        elif panel.corneradj.sum() == 2:
            # Special case, both edges of panel are corners
            # TODO
            pass
        else:
            newpanels += [panel]
            newindices += [panel.index]
    # Fix broken neighbors
    for i, brokenp in enumerate(brokenneighbs):
#        print("broken neighbor panel:",brokenp.index)
#        print("old neighbors:",brokenp.neighbors)
        if i%2 == 1:
            nextbroken = brokenneighbs[(i+1)%len(brokenneighbs)]
        else:
            nextbroken = brokenneighbs[i-1]
        otherneighbor = brokenp.neighbors[0]
        if brokenp.index not in newpanels[newindices.index(otherneighbor)].neighbors:
            otherneighbor = brokenp.neighbors[1]
        newneighbors = [nextbroken.index] + [otherneighbor]
#        print("new neighbors:",newneighbors)
        newpanels[newindices.index(brokenp.index)].neighbors = newneighbors
    return newpanels

def split_linear_panel(panel, rparam, newindex, ps, cornerneighbor):
#    print("splitting panel:",panel.index)
#    print("neighbors:",panel.neighbors)
#    print("cornerneighbor:",cornerneighbor)
    endpoints = panel.endpoints
    corner = panel.endpoints[:, panel.corneradj]
    notcorner = panel.endpoints[:, panel.corneradj == False]
    divide = 1/rparam*notcorner + (1 - 1/rparam)*corner
    # New panel further away from corner
    p = ps[0]
    index = panel.index
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_weights = unsc_weights.reshape((1, -1))
    nodes = notcorner + (divide - notcorner)*(unsc_nodes + 1)/2
    zpvec = np.ones((2, p))*(divide - notcorner)/2
    # Check direction:
    if np.any(np.einsum('ij,ij->j', panel.zpvec, zpvec) < 0):
        zpvec *= -1
    zppvec = np.zeros((2, p))
    neighbors = [neighbor for neighbor in panel.neighbors if neighbor != cornerneighbor] + [newindex]
    panel1 = bie.panel(nodes, zpvec, zppvec, unsc_weights, index, neighbors, panel.colinears + [newindex],\
                      endpoints = np.concatenate((divide, notcorner), axis = 1))

    # New corner panel
    p = ps[1]
    index = newindex
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_weights = unsc_weights.reshape((1, -1))
    nodes = divide + (corner - divide)*(unsc_nodes + 1)/2
    zpvec = np.ones((2, p))*(corner - divide)/2
    if np.any(np.einsum('ij,ij->j', panel.zpvec, zpvec) < 0):
        zpvec *= -1
    zppvec = np.zeros((2, p))
    neighbors = [panel.index, cornerneighbor]
    panel2 = bie.panel(nodes, zpvec, zppvec, unsc_weights, newindex, neighbors, panel.colinears + [newindex],\
                      endpoints = np.concatenate((corner, divide), axis = 1),\
                      corneradj = np.array([True, False]))
#    print("created panels:",panel1.index, panel2.index)
#    print("neighbors:",panel1.neighbors, panel2.neighbors)
    return panel1, panel2


def refine_stairs(panels, subdiv = 1, rparam = 2):
    """
    Searches for corner-adjacent panels and subdivides them `subdiv` times
    using `rparam`-adic refinement.
    """
    indices = [panel.index for panel in panels]
    newpanels = []
    newindices = []
    brokenneighbs = []
    maxindex = max(indices)
    oldmax = maxindex + 1
    for panel in panels:
        p = panel.nodes.shape[1]
        if panel.corneradj.sum() == 1:
            # This panel will be split
            # Find the neighboring panel that is closest to the corner
#            print("current panel ",panel.index)
#            print("neighbors:",panel.neighbors)
            corner = panel.endpoints[:, panel.corneradj]
#            print("corner:",corner)
            neighborp = panels[indices.index(panel.neighbors[0])]
            if True in neighborp.corneradj:
#                print("cornerneighbor guess:",neighborp.index)
                neighborscorner = neighborp.endpoints[:, neighborp.corneradj]
#                print("neighbors endpoints:",neighborp.endpoints)
#                print("neighbors corneradj:", neighborp.corneradj)
#                print("of which corner:", neighborp.endpoints[:, neighborp.corneradj])
#                print("neighbborp origin:", neighborp.origin,"panel origin:",panel.origin)
#                print(np.abs(neighborscorner - corner).sum())
                if np.abs(neighborscorner).sum() == 0.0 and (np.abs(neighborp.origin - panel.origin)).sum() == 0.0:
                    cornerneighbori = neighborp.index
#                    print("guess was correct")
                else:
                    if len(panel.neighbors) > 1:
#                        print("corrected ini guess")
                        cornerneighbori = panel.neighbors[1]
                        neighborp = panels[indices.index(panel.neighbors[1])]
                    else:
#                        print("no neighbouring panel sharing a corner")
                        cornerneighbori = -1
            else:
                if len(panel.neighbors) > 1:
#                    print("corrected ini guess")
                    cornerneighbori = panel.neighbors[1]
                    neighborp = panels[indices.index(panel.neighbors[1])]
                else:
#                    print("no neighbouring panel sharing a corner")
                    cornerneighbori = -1
            # Generate new panels
            paneltosplit = panel
            colinears = panel.colinears
            for s in range(subdiv):
                panel1, panel2 = split_linear_panel_relcoords(paneltosplit, rparam, maxindex + 1, (p, p), cornerneighbori)
                maxindex += 1
                paneltosplit = panel2
                panel1.n *= -1
                newpanels.append(panel1)
                newindices.append(panel1.index)
                colinears += [maxindex]
                if s == subdiv - 1:
                    panel2.n *= -1
                    newpanels += [panel2]
                    newindices += [panel2.index]
                    if len(panel2.neighbors) > 1 :
                        brokenneighbs.append(panel2)
            # Update colinears
            for colinindex in panel.colinears:
                if colinindex in newindices:
                    newpanels[newindices.index(colinindex)].colinears = colinears
                elif colinindex in indices:
                    panels[indices.index(colinindex)].colinears = colinears
        elif panel.corneradj.sum() == 2:
            # Special case, both edges of panel are corners
            # TODO
            pass
        else:
            newpanels += [panel]
            newindices += [panel.index]
    # Fix broken neighbors
#    print([bp.index for bp in brokenneighbs])
    for i, brokenp in enumerate(brokenneighbs):
#        print("broken neighbor panel:",brokenp.index)
#        print("old neighbors:",brokenp.neighbors)
        if i%2 == 0:
            nextbroken = brokenneighbs[i+1]
        else:
            nextbroken = brokenneighbs[i-1]
        otherneighbor = brokenp.neighbors[0]
        if brokenp.index not in newpanels[newindices.index(otherneighbor)].neighbors:
            otherneighbor = brokenp.neighbors[1]
        newneighbors = [nextbroken.index] + [otherneighbor]
#        print("new neighbors:",newneighbors)
        newpanels[newindices.index(brokenp.index)].neighbors = newneighbors

    sortedpanels = bie.sort_panels(newpanels)
    return sortedpanels


def refine_relcoords(panels, subdiv = 1, rparam = 2):
    """
    Searches for corner-adjacent panels and subdivides them `subdiv` times
    using `rparam`-adic refinement.
    """
    indices = [panel.index for panel in panels]
    newpanels = []
    newindices = []
    brokenneighbs = []
    maxindex = max(indices)
    oldmax = maxindex + 1
    for panel in panels:
        p = panel.nodes.shape[1]
        if panel.corneradj.sum() == 1:
            # This panel will be split
            # Find the neighboring panel that is closest to the corner
            print("current panel ",panel.index)
            print("neighbors:",panel.neighbors)
            corner = panel.endpoints[:, panel.corneradj]
#            print("corner:",corner)
            neighborp = panels[indices.index(panel.neighbors[0])]
            if True in neighborp.corneradj:
#                print("cornerneighbor guess:",neighborp.index)
                neighborscorner = neighborp.endpoints[:, neighborp.corneradj]
#                print("neighbors endpoints:",neighborp.endpoints)
#                print("neighbors corneradj:", neighborp.corneradj)
#                print("of which corner:", neighborp.endpoints[:, neighborp.corneradj])
#                print(np.abs(neighborscorner - corner).sum())
                if neighborscorner.sum() == 0.0 and (neighborp.origin - panel.origin).sum() == 0.0:
                    cornerneighbori = neighborp.index
#                    print("guess was correct")
                else:
#                    print("corrected ini guess")
                    cornerneighbori = panel.neighbors[1]
                    neighborp = panels[indices.index(panel.neighbors[1])]
            else:
                cornerneighbori = panel.neighbors[1]
                neighborp = panels[indices.index(panel.neighbors[1])]
            # Generate new panels
            paneltosplit = panel
            colinears = panel.colinears
            for s in range(subdiv):
                panel1, panel2 = split_linear_panel_relcoords(paneltosplit, rparam, maxindex + 1, (p, p), cornerneighbori)
                maxindex += 1
                paneltosplit = panel2
                newpanels.append(panel1)
                newindices.append(panel1.index)
                colinears += [maxindex]
                if s == subdiv - 1:
                    newpanels += [panel2]
                    newindices += [panel2.index]
                    brokenneighbs.append(panel2)
            # Update colinears
            for colinindex in panel.colinears:
                if colinindex in newindices:
                    newpanels[newindices.index(colinindex)].colinears = colinears
                elif colinindex in indices:
                    panels[indices.index(colinindex)].colinears = colinears
        elif panel.corneradj.sum() == 2:
            # Special case, both edges of panel are corners
            # TODO
            pass
        else:
            newpanels += [panel]
            newindices += [panel.index]
    # Fix broken neighbors
    
    for i, brokenp in enumerate(brokenneighbs):
        print("broken neighbor panel:",brokenp.index)
        print("old neighbors:",brokenp.neighbors)
        if i%2 == 1:
            nextbroken = brokenneighbs[(i+1)%len(brokenneighbs)]
        else:
            nextbroken = brokenneighbs[i-1]
        otherneighbor = brokenp.neighbors[0]
        if brokenp.index not in newpanels[newindices.index(otherneighbor)].neighbors:
            otherneighbor = brokenp.neighbors[1]
        newneighbors = [nextbroken.index] + [otherneighbor]
        print("new neighbors:",newneighbors)
        newpanels[newindices.index(brokenp.index)].neighbors = newneighbors

    sortedpanels = bie.sort_panels(newpanels)
    return sortedpanels

def split_linear_panel_relcoords(panel, rparam, newindex, ps, cornerneighbor):
#    print("splitting panel:",panel.index)
#    print("neighbors:",panel.neighbors)
#    print("cornerneighbor:",cornerneighbor)
    endpoints = panel.endpoints
    corner = panel.endpoints[:, panel.corneradj]
    notcorner = panel.endpoints[:, panel.corneradj == False]
    divide = 1/rparam*notcorner + (1 - 1/rparam)*corner
    # New panel further away from corner
    p = ps[0]
    index = panel.index
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_weights = unsc_weights.reshape((1, -1))
    nodes = notcorner + (divide - notcorner)*(unsc_nodes + 1)/2
    zpvec = np.ones((2, p))*(divide - notcorner)/2
    # Check direction:
    if np.any(np.einsum('ij,ij->j', panel.zpvec, zpvec) < 0):
#        print("reversed", panel.index)
        zpvec *= -1
        nodes = divide + (notcorner - divide)*(unsc_nodes + 1)/2

    zppvec = np.zeros((2, p))
    neighbors = [neighbor for neighbor in panel.neighbors if neighbor != cornerneighbor] + [newindex]
    panel1 = bie.panel(nodes, zpvec, zppvec, unsc_weights, index, neighbors, panel.colinears + [newindex],\
                      endpoints = np.concatenate((divide, notcorner), axis = 1), origin = panel.origin)

    # New corner panel
    p = ps[1]
    index = newindex
    unsc_nodes, unsc_weights = np.polynomial.legendre.leggauss(p)
    unsc_weights = unsc_weights.reshape((1, -1))
    nodes = divide + (corner - divide)*(unsc_nodes + 1)/2
    zpvec = np.ones((2, p))*(corner - divide)/2
    if np.any(np.einsum('ij,ij->j', panel.zpvec, zpvec) < 0):
        zpvec *= -1
        nodes =  corner + (divide - corner)*(unsc_nodes + 1)/2
    zppvec = np.zeros((2, p))
    if cornerneighbor != -1:
        neighbors = [panel.index, cornerneighbor]
    else:
        neighbors = [panel.index]
    panel2 = bie.panel(nodes, zpvec, zppvec, unsc_weights, newindex, neighbors, panel.colinears + [newindex],\
                      endpoints = np.concatenate((corner, divide), axis = 1),\
                      corneradj = np.array([True, False]), origin = panel.origin)
#    print("created panels:",panel1.index, panel2.index)
#    print("neighbors:",panel1.neighbors, panel2.neighbors)
    return panel1, panel2
 

def gen_closed_curve(n = 3, p = 30):
    """
    Returns one panel representing a closed curve with 3 "vertices" and p periodic trap nodes
    """
    R = lambda s: 1 + 0.3*np.cos(n*s)
    Rp = lambda s: -0.3*n*np.sin(n*s)
    Rpp = lambda s: -0.3*n*n*np.cos(3*s)
    z = lambda s: np.array([R(s)*np.cos(s), R(s)*np.sin(s)])
    zp = lambda s: np.sqrt(Rp(s)**2 + R(s)**2)
    zpvec = lambda s: np.array([Rp(s)*np.cos(s) - R(s)*np.sin(s), Rp(s)*np.sin(s) + R(s)*np.cos(s)])
    zppvec = lambda s: np.array([Rpp(s)*np.cos(s) - 2*Rp(s)*np.sin(s) - R(s)*np.cos(s),\
                                 Rpp(s)*np.sin(s) + 2*Rp(s)*np.cos(s) - R(s)*np.sin(s)])
#    nvec = lambda s: 1/zp(s)*np.array([[Rp(s)*np.sin(s) + R(s)*np.cos(s)], [R(s)*np.sin(s) - Rp(s)*np.cos(s)]])
    ss = np.linspace(0, 2*np.pi, p+1)[:-1]
    unsc_nodes = ss.reshape((1, p))
    nodes = z(ss)
    zpvec = zpvec(ss)
    zppvec = zppvec(ss)
    unsc_weights = np.ones((1, p))*2*np.pi/p 
    print(nodes.shape, zpvec.shape, zppvec.shape)
    panel = bie.panel(nodes, zpvec, zppvec, unsc_weights, 0, (0, 0))
    return [panel]







