import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import numpy as np
import importlib
import bie.bie as bie
import geometry.geometry as geometry
import utils.utils as utils
import cmath
import re
import logging
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from multiprocessing import Pool
import pandas
import time
import os
from itertools import repeat

def plot(solnx1, solnx2, u, omega):
    solnmasked = np.where(solnx2 > np.abs(0.5 - 1/np.pi*np.arccos(np.cos(np.pi*solnx1))) - 0.5, u, np.nan + np.nan*1j)
    fig, ax = plt.subplots(2, 1)#, layout = 'constrained')
    fig.set_size_inches(10, 4)
    ax[0].clear()
    ax[1].clear()
#    CS1 = ax[0].contourf(solnx1, solnx2, solnmasked.real, cmap = 'jet', levels = 100, vmin = -0.6, vmax = 0.6, extend = 'both')
#    CS2 = ax[1].contourf(solnx1, solnx2, solnmasked.imag, cmap = 'jet', levels = 100, vmin = -0.6, vmax = 0.6, extend = 'both')
    CS1 = ax[0].pcolormesh(solnx1, solnx2, solnmasked.real, cmap = 'jet')
    CS2 = ax[1].pcolormesh(solnx1, solnx2, solnmasked.imag, cmap = 'jet')
    boundaryx = np.linspace(-7.5, 7.5, 2000)
    boundaryy = np.abs(0.5 - 1/np.pi*np.arccos(np.cos(np.pi*boundaryx))) - 0.5
    ax[0].plot(boundaryx, boundaryy, color = 'black')
    ax[1].plot(boundaryx, boundaryy, color = 'black')

    div1 = make_axes_locatable(ax[0])
    div2 = make_axes_locatable(ax[1])
    colbar_axes1 = div1.append_axes("right", size = '2%', pad = 0.1)
    colbar_axes2 = div2.append_axes("right", size = '2%', pad = 0.1)
    fig.colorbar(CS1, cax = colbar_axes1)
    fig.colorbar(CS2, cax = colbar_axes2)
    ax[0].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    ax[0].set_xlim((-7.5, 7.5))
    ax[1].set_xlim((-7.5, 7.5))
    ax[1].set_xlabel("$x_1$")
    ax[1].set_ylabel("$x_2$")
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    plt.savefig("./test-field2-omega-{}.pdf".format(str(omega).replace(".","")))


def reconstruct_solution(panels, files, coords = False):
    """
    Parameters
    ----------
    panels: list [Panel]
        List of Panel objects defining the boundary geometry.
    files: list [string]
        List of files in any order:
            - list of kappa values and weights for array scanning integral
            - density at the boundary at each kappa value
            - lattice sum coefficients at each kappa value
            - parameters file with k, x, and y position of source
    Returns
    -------
    Total radiated power from a point source.
    """
    fieldfile = [f for f in files if "sigma" in f][0]
    snsfile = [f for f in files if "Sns" in f][0]
    kappafile = [f for f in files if "kappas" in f][0]
    weightfile = [f for f in files if "weights" in f][0]
    paramfile = [f for f in files if "params" in f][0]
    # Read in kappa, tau/sigma on bdry, and lattice sum coeffs from file
    kappas = np.loadtxt(kappafile, dtype = complex)
    fields = np.loadtxt(fieldfile, dtype = complex)
    weights = np.loadtxt(weightfile, dtype = complex)
    allsns = np.loadtxt(snsfile, dtype = complex)
    nkappas = kappas.shape[0]
    fields = fields.reshape((nkappas, -1))
    allsns = allsns.reshape((nkappas, -1))
    # Read parameters (k, source position)
    data = pandas.read_csv(paramfile, sep = ', ')
    k = data['k'][0]
    xsource = np.array([[data['sourcex'][0]], [data['sourcey'][0]]])
    u_kappas = np.zeros((len(ncells)*res, (nvcells+1)*res, nkappas), dtype = complex)
    print("Shape of solution array: ", u_kappas.shape)
    for i in range(nkappas):
        time1 = time.time()
        kappa = kappas[i]
        log.warning("Reconstruction. kappa: "+str(kappa))
        fieldN = fields[i,:]
        Sns = allsns[i,:]
        alph = np.exp(1j*kappa)
        # Reconstruct field
        if i == 0 and coords == True:
            x1_arr, x2_arr, u_kappa = bie.reconstruct_arrayscan_fromfile(panels, files,\
                                      #ncells = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],\
                                      ncells = ncells,\
                                      res = res, nvcells = nvcells)
        else:
            u_kappa = bie.reconstruct_arrayscan_fromfile(panels, files,\
                      ncells = ncells,\
                      #ncells = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8],\
                      res = res, nvcells = nvcells, no_return_xs = True)

        u_kappas[:,:,i] = u_kappa        
        time2 = time.time()
        log.warning("Time elapsed for reconstructing kappa = : "+ str(kappa) + " was " + str(time2-time1) + " seconds.")
    u = 1/(2*np.pi)*np.einsum('ijk,k->ij', u_kappas, weights) 
    if coords == True:
        return x1_arr, x2_arr, u
    else:
        return u

def Helm_ext_Neu_stairs_ptsrc_field_wopanels(panels, k, cosphii, xsource = np.array([[-0.2], [0.1]])):
    # Calculate lattice sums
    kd = k
    a = 10.0
    eps = 1e-7
    a = max(-np.log(eps)/kd, a)
    print("a for latsum integral is", a)
    N = 1000#int(a/0.001) #0.0005
    nmax = 40
    alph = np.exp(1j*kd*cosphii)
    Sns = np.zeros(nmax + 1, dtype = complex)
    for i in range(nmax + 1):
        Sns[i] = bie.Sn_offset(i, kd, cosphii, a, N = N)
    # Periodic array of point sources
    ui = lambda x: bie.Phi_p_offset(x, xsource, k, Sns, alph, self = False).flatten()
    dui = lambda nvec, x: bie.dPhi_p_dn_offset(x, xsource, nvec, k, Sns, alph, nxory = 'x', self = False).flatten()
    # Calculate boundary condition
    f = lambda nvec, x: -dui(nvec, x) 
    # Solve
    tauN = bie.solve_bie_periodic(panels, f, k, Sns, alph, relcorner = True, bc = 'Neu')
    return tauN, np.array(Sns)

def asm_integral(k, xsource, A = 0.1, pkap = 200, n = 1, p = 16, rparam = 3, d = 1, beta = np.pi/2, nperside = 8, subdiv = 10, filename = "helmextneu-asm-fieldN-sine-1.txt", poleloc = 0.0, expgrad = False, b = 5):
    # Generate geometry once and for all
    print("Beta fed to asm_integral:", beta)
    panels = geometry.gen_staircase(1, p, d, beta, nperside)
    panels = geometry.refine_stairs(panels, subdiv = subdiv, rparam = rparam)
    # Generate path in complex kappa plane
    if expgrad:
        if k < 1:
            path = geometry.sine_path_expgrad_zero(A = A, p = pkap, beta = b)
            log.warning("Exp graded path near zero for kappa = {}".format(poleloc))
        else: 
            path = geometry.sine_path_expgrad_pi(A = A, p = pkap, beta = b)
            log.warning("Exp graded path near pi for kappa = {}".format(poleloc))
    else:
        path = geometry.sine_path(A = A, p = pkap)
    nodes = path[0].nodes[0] + path[0].nodes[1]*1j
    nkappa = nodes.shape[0]
    print(nkappa)
    snsfile = filename.replace("sigma", "Sn")
    with open(filename, 'w') as f:
        with open(snsfile, 'w') as snsf:
            for i, kappai in enumerate(nodes): 
                timeperkap0 = time.time()
                print("Array scanning. Kappa = ", kappai)
                cosphii = kappai/k
                field, sns = Helm_ext_Neu_stairs_ptsrc_field_wopanels(panels, k, cosphii, xsource = xsource)
                f.write("# kappa = {}\n".format(kappai))
                snsf.write("# kappa = {}\n".format(kappai))
                np.savetxt(f, field)
                np.savetxt(snsf, sns)
                timeperkap1 = time.time()
                log.warning("Solve time for kappa = "+str(kappai)+" was "+str(timeperkap1-timeperkap0)+" seconds.")
    kappafile = filename.replace("sigma", "kappa") 
    np.savetxt(kappafile, nodes)
    weights = np.zeros(nkappa, dtype = complex)
    for i, pathpanel in enumerate(path):
        weights[i*pkap:(i+1)*pkap] = pathpanel.weights*(pathpanel.zpvec[0,:] + 1j*pathpanel.zpvec[1,:])
    wfile = filename.replace("sigma", "weight") 
    np.savetxt(wfile, weights)
    pfile = filename.replace("sigma", "param")
    with open(pfile, 'w') as f:
        f.write("k, sourcex, sourcey, poleloc\n")
        f.write("{}, {}, {}, {}".format(k, xsource[0,0], xsource[1,0], poleloc))
    return panels

def run_loop(loop_instructions, input, pool):
    pool.starmap(loop_instructions, input)
   

def loop_over_omega(input_index, zipinputs):
    log = zipinputs['log'] 
    p = zipinputs['p']
    current_om = zipinputs['current_om']
    xsource = zipinputs['xsource']
    d = zipinputs['d']
    beta = zipinputs['beta']
    rparam = zipinputs['rparam']
    subdiv = zipinputs['subdiv']
    res = zipinputs['res']
    ncells = zipinputs['ncells']
    kappafile = zipinputs['kappafile'] 
    wfile = zipinputs['wfile'] 
    pfile = zipinputs['pfile'] 
    nvcells = zipinputs['nvcells'] 
    kappanodes = zipinputs['kappanodes']
    nperside = zipinputs['nperside']
    log.warning(str(input_index))
    try:
        kappa = kappanodes[input_index]
        k = current_om
        # Define geometry
        cosphii = kappa/k
        log.warning("k is {}".format(k))
        panels = geometry.gen_staircase(1, p, d, beta, nperside)
        panels = geometry.refine_stairs(panels, subdiv = subdiv, rparam = rparam)
        filename = "./data/thewell-omega-{}-x0-{}-{}-kappa-{}-sigmas.txt".format(str(k).replace(".",""), str(xsource[0,0]).replace(".",""), str(xsource[1,0]).replace(".",""), str(kappa.real).replace(".",""))
        snsfile = filename.replace("sigma", "Sn")
        # Solve BIE at this (complex) kappa
        timeperkap0 = time.time()
        field, sns = Helm_ext_Neu_stairs_ptsrc_field_wopanels(panels, k, cosphii, xsource = xsource)
        with open(filename, 'w') as f:
            f.write("# kappa = {}\n".format(kappa))
            np.savetxt(f, field)
        with open(snsfile, 'w') as snsf:
            snsf.write("# kappa = {}\n".format(kappa))
            np.savetxt(snsf, sns)
        timeperkap1 = time.time()
        log.warning("Solve time for kappa = "+str(kappa)+" was "+str(timeperkap1-timeperkap0)+" seconds.")
        reconstfile = filename.replace("sigmas", "reconsfield")
        x1file = "./data/thewell-coords-x1.txt"
        x2file = "./data/thewell-coords-x2.txt"
        # Reconstruct solution
        timerecon0 = time.time()
        files = [filename, snsfile, kappafile, wfile, pfile]        
        log.warning(str(files))
        if input_index == 0:
            x1_arr, x2_arr, u_kappa = bie.reconstruct_arrayscan_fromfile_singlekappa(panels, files,\
                                      ncells = ncells,\
                                      #ncells = [0],\
                                      res = res, nvcells = nvcells)
            log.warning("U shape: "+str(u_kappa.shape))
            with open(reconstfile, 'w') as f:
                f.write("# kappa = {}\n".format(kappa))
                np.savetxt(f, u_kappa)
            np.savetxt(x1file, x1_arr)
            np.savetxt(x2file, x2_arr)
        else:
            u_kappa = bie.reconstruct_arrayscan_fromfile_singlekappa(panels, files,\
                      #ncells = [0],\
                      ncells = ncells,\
                      res = res, nvcells = nvcells, no_return_xs = True)
            log.warning("U shape: "+str(u_kappa.shape))
            with open(reconstfile, 'w') as f:
                f.write("# kappa = {}\n".format(kappa))
                np.savetxt(f, u_kappa)

#        u_kappas[:,:,i] = u_kappa        
        timerecon1 = time.time()
        log.warning("Reconstructing field for kappa = "+ str(kappa) + " took " + str(timerecon1-timerecon0) + " seconds.")
#    u = 1/(2*np.pi)*np.einsum('ijk,k->ij', u_kappas, weights) 

    except Exception as e:
        log.warning(e)



def main():

    xsources = np.array([[-0.2], [0.1]])
    #xsources = np.array([[-0.4, -0.4, -0.4, -0.4, -0.3, -0.3, -0.3, -0.2, -0.2, -0.2, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4, 0.4, 0.4, -0.4, -0.3, 0.0, 0.0, 0.3, 0.4], [-0.2, 0.0, 0.2, 0.4, -0.1, 0.1, 0.3, 0.0, 0.2, 0.4, 0.1, 0.3, 0.2, 0.4, 0.1, 0.3, 0.0, 0.2, 0.4, -0.1, 0.1, 0.3, -0.2, 0.0, 0.2, 0.4, -0.1, 0.0, 0.1, 0.3, 0.0, -0.1]])
    lenx = xsources.shape[1]
    n = 1
    d = 1
    p = 8
    beta = np.pi/2
    nperside = 8
    subdiv = 10
    pkap = 64
    rparam = 3
    res = 64
    sineA = 1.0
    b = 5 #for exponential grading
    allkappasfile = "disprel.txt"
    allkappasdat = np.loadtxt(allkappasfile, delimiter = ',')
    #allkappas = allkappasdat[:-3:3,0]
    #allks = allkappasdat[:-3:3,1]
    allkappas = allkappasdat[:1,0]
    allks = allkappasdat[:1,1]
    ncells = [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    nvcells = 3
    logging.basicConfig(format='%(message)s')
    log = logging.getLogger(__name__)
    
    # Parallelize each omega, loop over kappa
    
    Nkappa = len(allkappas)
    
    # Loop over source position x0
    for l in range(lenx):
        xsource = np.array([[xsources[0, l]], [xsources[1, l]]])
        log.warning("xsource {}".format(xsource))
        # Alternatively, loop over kappa
        for current_om, current_kap in zip(allks, allkappas):
            poleloc = current_kap
            # Make relevant files
            filename = "./data/thewell-omega-{}-x0-{}-{}-sigmas.txt".format(str(current_om).replace(".",""), str(xsource[0,0]).replace(".",""), str(xsource[1,0]).replace(".",""))   
            ufile = filename.replace("sigmas", "u")
            try: 
                f = open(ufile)
                f.close()
                log.warning("{} exists.".format(ufile))
                x1file = "./data/thewell-coords-x1.txt"
                x2file = "./data/thewell-coords-x2.txt"
                solnx1 = np.loadtxt(x1file)
                solnx2 = np.loadtxt(x2file)
                u = np.loadtxt(filename.replace("sigmas", "u"), dtype = complex)
                plot(solnx1, solnx2, u, current_om)

            except FileNotFoundError:
                wfile = filename.replace("sigma", "weight") 
                pfile = filename.replace("sigma", "param")
                kappafile = filename.replace("sigma", "kappa") 
                with open(pfile, 'w') as f:
                    f.write("k, sourcex, sourcey, poleloc\n")
                    f.write("{}, {}, {}, {}".format(current_om, xsource[0,0], xsource[1,0], poleloc))
            
                # Create sinusoidal path
                expgrad = False
                if current_kap <= 0.5 or current_kap >= 2.5:
                    expgrad = True
                if expgrad:
                    if current_om < 1:
                        path = geometry.sine_path_expgrad_zero(A = sineA, p = pkap, beta = b)
                        log.warning("Exp graded path near zero for kappa = {}".format(poleloc))
                    else: 
                        path = geometry.sine_path_expgrad_pi(A = sineA, p = pkap, beta = b)
                        log.warning("Exp graded path near pi for kappa = {}".format(poleloc))
                else:
                    path = geometry.sine_path(A = sineA, p = pkap)
                kappanodes_re = path[0].nodes[0]
                kappanodes = path[0].nodes[0] + path[0].nodes[1]*1j
                Nkappa_asm = kappanodes_re.shape[0]
                weights = np.zeros(Nkappa_asm, dtype = complex)
                for i, pathpanel in enumerate(path):
                    weights[i*pkap:(i+1)*pkap] = pathpanel.weights*(pathpanel.zpvec[0,:] + 1j*pathpanel.zpvec[1,:])
                np.savetxt(wfile, weights)
                np.savetxt(kappafile, kappanodes)
                processes_pool = Pool(Nkappa_asm)
                inputs = {'p': p, 'd': d, 'xsource': xsource, 'beta': beta, 'nperside': nperside,\
                          'subdiv': subdiv, 'rparam': rparam, 'log': log, 'ncells': ncells, 'nvcells': nvcells,\
                          'wfile': wfile, 'pfile': pfile, 'current_om': current_om, 'res': res, 'kappafile': kappafile,\
                          'kappanodes': kappanodes}
                run_loop(loop_over_omega, zip(range(Nkappa_asm), repeat(inputs)), processes_pool)
            
                # Combine and plot
                u_tot = np.zeros((res, (nvcells+1)*res), dtype = complex)
                u_kappas = np.zeros((len(ncells)*res, (nvcells + 1)*res, Nkappa_asm), dtype = complex)
                rootdir = "./data/"
                for i in range(Nkappa_asm):
                    kappa = kappanodes[i]
                    print("kappa: ", kappa)
                    # Find the right file
                    for sfile in os.listdir(rootdir):
                        snsfile = rootdir+sfile 
                        #print("checking file", snsfile)
                        if snsfile.endswith("Sns.txt") and snsfile.startswith("./data/thewell-omega-{}-x0-{}-{}".format(str(current_om).replace(".",""), str(xsource[0,0]).replace(".",""), str(xsource[1,0]).replace(".",""))):
                            with open(snsfile, 'r') as f:
                                kappai = complex(f.readline().split(" = ")[-1])
                            if np.abs((kappa - kappai)/kappa) < 1e-6:
                                log.warning("found matching kappa: {}".format(kappai))
                                log.warning("Sns file: {}".format(snsfile))
                                fieldfile = snsfile.replace("Sns", "reconsfield")
                                u_kappa = np.loadtxt(fieldfile, dtype = complex, comments = '#')
                                u_kappas[:,:,i] = u_kappa
                log.warning("u kappas: {}".format(u_kappas))
                u = 1/(2*np.pi)*np.einsum('ijk,k->ij', u_kappas, weights) 
                np.savetxt(ufile, u)
                x1file = "./data/thewell-coords-x1.txt"
                x2file = "./data/thewell-coords-x2.txt"
                solnx1 = np.loadtxt(x1file)
                solnx2 = np.loadtxt(x2file)
        #        u = np.loadtxt(filename.replace("sigmas", "u"), dtype = complex)
        #        plot(solnx1, solnx2, u, current_om)
   
   
if __name__ == "__main__":
    main()

   
