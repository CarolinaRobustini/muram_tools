import numpy as np
import ISPy.io.lapalma as lp
#from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import scipy.io as sc
import scipy.interpolate
from scipy.interpolate import interp1d
import scipy.ndimage
import math

ee = 1.602189e-12  # electron volt [erg]
hh = 6.626176e-27  # Planck's constant [erg s]
cc = 2.99792458e10 # Speed of light [cm/s]
em = 9.109534e-28  # Electron mass [g] 
uu = 1.6605655e-24 # Atomic mass unit [g]
bk = 1.380662e-16  # Boltzman's cst. [erg/K]
pi = 3.14159265359 # Pi
ec = 4.80325e-10   # electron charge [statcoulomb]

hc2     = hh * cc**2
h_e     = hh / ee
hc_e    = hh * cc / ee
twoh_c2 = 2.0 * hh / cc**2
c2_2h   = cc**2 / 2.0 / hh
twohc   = 2.0 * hh * cc
hc_k    = hh * cc / bk 
e_k      = ee/bk
h_k     = hh / bk
h_4pi    = hh / 4.0 / pi

f_to_Aji = 8.0 * pi**2 * ec**2 / em / cc
KM_TO_CM =1e5
CM_TO_KM =1e-5
AA_TO_CM =1e-8
CM_TO_AA =1e8
grph = 2.380491e-24   

def read_cube(fil0,fil1):

    nx, ny, dum, ns, dtype, ndim = lp.head(fil0)
    nw, nt, dum, ns, dtype, ndim = lp.head(fil1)

    io = np.memmap(fil0, mode='r', shape=(nt,ns,nw,ny,nx), dtype=dtype,
                offset=512)
    return io
    

# def read_lp_cube(file):
#     nx, ny, nt, dum, dtype, ndim = lp.head(file)
#     io = np.memmap(file, mode='r', shape=(nt,ny,nx), dtype=dtype,
#                 offset=512)
#    return io
    
def get_limbdarkening(mu, line):
    A={'6302':[0.33644, 1.30590, -1.79238, 2.45040, -1.89979, 0.59943, 0.8290],
       '6563':[0.34685, 1.37539, -2.04425, 2.70493, -1.94290, 0.55999, 0.8360],
       '8542':[0.45396, 1.25101, -2.02958, 2.75410, -2.02287, 0.59338, 0.8701],
       '3950':[0.12995, 0.91836, -0.07566, 0.19149, -0.28712, 0.12298, 0.7204]}
   
    ptot = 0.0

    for kk in range(6):
        ptot += (A[line][kk]*mu**(kk))


    return ptot



def derivative(x,y):

    n= x.shape[0]
    yp= np.zeros(n)
    
    dx  = x[1:n-1] -x[0:n-2]
    dx1 = x[2:] - x[1:n-1]

    der = (y[1:n-1] - y[0:n-2]) / dx
    der1 = (y[2:] - y[1:n-1]) / dx1

    idx= np.where(der*der1 > 0.0)
    if (((der*der1 > 0.0).nonzero() >0) == True):
        lambdad = (1. + dx1[idx] / (dx1[idx] + dx[idx])) / 3.
        yp[np.asarray(idx)+1] = (der[idx] * der1[idx]) / (lambdad * der1[idx] + (1. - lambdad) * der[idx])
    else:
        yp[1:n-1] = (dx1 * der + dx * der1) / (dx1+dx)
        
    yp[0] = der[0]  
    yp[n-1] = der1[der1.shape[0]-1]
    

    return yp



# def my_cluster(f0, f1, nclu, t_step=26, s=0, w0=0,w1=21, y0=50, x0=120):
#     cube=read_polcube(f0,f1)
#     nt=cube[:,s,w0:w1,y0:,x0:].shape[0]
#     nw=cube[:,s,w0:w1,y0:,x0:].shape[1]
#     ny=cube[:,s,w0:w1,y0:,x0:].shape[2]
#     nx=cube[:,s,w0:w1,y0:,x0:].shape[3]
#     mm= np.zeros((ny*nx))
#     dat1=cube[t_step,s,w0:w1,y0:,x0:]
#     dat =dat1.reshape(nw,nx*ny)
#     mm=KMeans(n_clusters=nclu).fit(np.transpose(dat)).labels_
#     return mm.reshape(ny,nx)
    
    
    
def readfits(name, ext=0):
    print ("loading -> {0}".format(name))
    io = fits.open(name)
    res = np.float32(io[ext].data[:])

    return res


def writefits(data, filename):
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    print ("writing -> " + filename)
    hdul.writeto(filename)
    
def readz_bifrost(name, ext=1):
    print ("loading -> {0}".format(name))
    io = fits.open(name)
    res = np.float32(io[ext].data[:])

    return res


def read_header(name, ext=0):
    print ("loading -> {0}".format(name))
    io = fits.open(name)
    res = io[ext].header

    return res

def planck(tg,l):
    """
    Planck function, tg in K, l in Angstrom, Bnu in erg/(s cm2 Hz ster)
    """ 
    nu = cc/(l*AA_TO_CM)
    return  2.0 * hh * nu * (nu/cc)**2 / (np.exp((hh/bk)*(nu/tg))-1.0)


def trad(inu,l):
    """
    inverse Planck function, Inu in  erg/(s cm2 Hz ster), l in Angstrom
    output in K
    """
    ll = l*AA_TO_CM
    return (hh/bk) * (cc/ll) / np.log(2.0 * cc * hh / ll**3 / inu +1.0) 

def restore(file):
    return sc.readsav(file,verbose=True,python_dict=True)



def fw6_deriv(x,y):
    """First derivative with 6th order accuracy
    result= fw6_deriv(x,y)
    x,y: input 1-D array
    x and y must have the same size 
    x has to be a uniform grid
    """
    
    nx = x.shape[0]
    ny = y.shape[0]
    dd = np.zeros(nx)
    if nx!=ny:
        raise Exception("x and y must have the same size and regular grid")
        
    coeff = [-49/20.,6.,-15/2.,20/3.,-15/4.,6/5.,-1/6.]
    for i in range(nx-len(coeff)):
        for c in range(len(coeff)):
            dd[i] += coeff[c]*y[i+c]
        dd[i] /= (x[i+c]-x[i])/6.  
    return(dd)
        

def interp_mia(x,y,xx):
    f= interp1d(x,y, fill_value = 'extrapolate')
    yy= f(xx)
    return yy

def check_nan(data):

    nan_val_pos= []
    
    if(len(data.shape) == 4):
        nx = data.shape[0]
        ny = data.shape[1]
        nz = data.shape[2]
        nw = data.shape[3]
        for xx in range(0,nx):
            for yy in range(0,ny):
                for zz in range(0,nz):
                    for ww in range(0,nw):
                        if(math.isnan(data[xx,yy,zz,ww])):
                            #print(xx,yy,zz,ww)
                            nan_val_pos.append((xx,yy,zz,ww))
                    
    if(len(data.shape) == 3):
        nx = data.shape[0]
        ny = data.shape[1]
        nz = data.shape[2]
        for xx in range(0,nx):
            for yy in range(0,ny):
                for zz in range(0,nz):
                    if(math.isnan(data[xx,yy,zz])):
                        #print(xx,yy,zz)
                        nan_val_pos.append((xx,yy,zz))                           
                    
    if(len(data.shape) == 2):
        nx = data.shape[0]
        ny = data.shape[1]
        for xx in range(0,nx):
            for yy in range(0,ny):
                if(math.isnan(data[xx,yy])):
                    #print(xx,yy)
                    nan_val_pos.append((xx,yy))
                    
    if(len(data.shape) == 1):
        nx = data.shape[0]
        for xx in range(0,nx):
            if(math.isnan(data[xx])):
                #print(xx)
                nan_val_pos.append((xx))
                    
    return nan_val_pos
    
   

def check_inf(data):

    inf_val_pos= []
    
    if(len(data.shape) == 4):
        nx = data.shape[0]
        ny = data.shape[1]
        nz = data.shape[2]
        nw = data.shape[3]
        for xx in range(0,nx):
            for yy in range(0,ny):
                for zz in range(0,nz):
                    for ww in range(0,nw):
                        if(math.isinf(data[xx,yy,zz,ww])):
                            print(xx,yy,zz,ww)
                            inf_val_pos.append((xx,yy,zz,ww))
                    
    if(len(data.shape) == 3):
        nx = data.shape[0]
        ny = data.shape[1]
        nz = data.shape[2]
        for xx in range(0,nx):
            for yy in range(0,ny):
                for zz in range(0,nz):
                    if(math.isinf(data[xx,yy,zz])):
                        print(xx,yy,zz)
                        inf_val_pos.append((xx,yy,zz))                           
                    
    if(len(data.shape) == 2):
        nx = data.shape[0]
        ny = data.shape[1]
        for xx in range(0,nx):
            for yy in range(0,ny):
                if(math.isinf(data[xx,yy])):
                    print(xx,yy)
                    inf_val_pos.append((xx,yy))
                    
    if(len(data.shape) == 1):
        nx = data.shape[0]
        for xx in range(0,nx):
            if(math.isinf(data[xx])):
                print(xx)
                inf_val_pos.append((xx))
                    
    return inf_val_pos


def rjo(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]



def log10_sign(data):
    if(np.isscalar(data)):
        if (data <0):
            res = -np.log10(np.abs(data))
        else:
            res = np.log10(np.abs(data))
    else:      
        if(len(data.shape) == 1):
            nx = data.shape[0]
            res = np.zeros(nx)
            for xx in range(0,nx):
                 if (data[xx] <0):
                     res[xx] = -np.log10(np.abs(data[xx]))
                 else:
                     res[xx] = np.log10(np.abs(data[xx]))
                     
        if(len(data.shape) == 2):
            nx = data.shape[0]
            ny = data.shape[1]
            res = np.zeros((nx,ny))
            for xx in range(0,nx):
                for yy in range(0,ny):
                 if (data[xx,yy] <0):
                     res[xx,yy] = -np.log10(np.abs(data[xx,yy]))
                 else:
                     res[xx,yy] = np.log10(np.abs(data[xx,yy]))        

        
    return res



def value_locate(vect,values): 
    if (type(values) != list): values = [values]  
    print(values, type(values))             
    #vector must be monotonic
    nlv = len(vect)
    index = np.zeros(len(values))
    if (vect[0]< vect[1]):
        for value in values:
            print("value= ",value)
            j = values.index(value)
            print(j)
            for i in range(0,nlv):
                if (value < vect[0]):   
                    index[j] = -1
                    break
                else:
                    if (value >= vect[i]) and (value < vect[i+1]):
                        index[j] = i
                        break
                    else:
                        if (value >= max(vect)):
                            index[j]= np.where(np.array(vect)==max(vect))[0]   
                            break
                        else:
                            index[j] = i-1        
    if (vect[0]> vect[1]):
        for value in values:
            print("value= ",value)
            j = values.index(value)
            print(j)
            for i in range(0,nlv):
                if (value >= vect[0]):   
                    index[j] = -1
                    break
                else:
                    if (value < vect[i]) and (value >= vect[i+1]):
                        index[j] = i
                        break
                    else:
                        if (value < min(vect)):
                            index[j]=np.where(np.array(vect)==max(vect))[0] 
                            break
                        else:
                            index[j] = i-1
                            
                     


    return index



def read_time_muram(snapshots, dir = "/scratch/crobu/fake_peacocks/lowres/"):
    snapshots = list(snapshots)
    times = []
    dum=[]
    dt = np.zeros(len(snapshots))
    time_file= dir+"BC.log"
    print("reading time log")
    coloumn1 = []
    coloumn2 = []
    with open(dir+"BC.log", "r") as f:
        data = f.readlines()
        #print dataexit
        for line in data:
            coloumn1.append(line.strip().split()[0])
            coloumn2.append(float(line.strip().split()[1]))
    index=0
    for element in coloumn1:
        if (int(element) in snapshots):
            if (element not in coloumn1[0:index-1]) :
                dum.append(coloumn1[index])
                times.append(coloumn2[index])
        index +=1
    print("computing dt")
    for tt in range(0, len(snapshots)-1):
        dt[tt]= times[tt+1]-times[tt]
    dt[-1]=dt[-2]
    dd= dict(zip(dum, list(np.arange(0,len(dum)))))        
    return(times)

def calc_curv(x,y,z):

    k = np.zeros(x.shape[0])
    
    ix = (np.where(x == 0))[0]
    iy = (np.where(y == 0))[0]
    iz = (np.where(z == 0))[0]
    
    ind = min(ix.min(),iy.min(),iz.min())
    
    
    x = x[0:ind]
    y = y[0:ind]
    z = z[0:ind]
    
    
    #first derivatives 
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
        
    #second derivatives 
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    d2z = np.gradient(dz)
    
    up   = np.sqrt((d2z*dy-d2y*dz)**2+(d2x*dz-d2z*dx)**2+(d2y*dx-d2x*dy)**2) 
    down = (dx**2+dy**2+dz**2)**(3./2.)
    
    k[0:ind] = up /down

    return k

def add_colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar
    
                    