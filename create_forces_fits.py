from My_functions import *
import m3d
import math
import simul
from ipdb import set_trace as stop
import gc
import helita.sim.muram as chcode
simulation = 3

#0 ---> low cadence  high res
#1 ---> high cadence high res
#2 ---> high cadence low res

################################################################# 

if (simulation == 0):
    shx = 1024
    shy = 1024
    shz = 1024
    snap= np.arange(557000,603000,1000, dtype=int)
    hpx = 368
    dir3D = "peacock/3D/"
    odir="forces/lowcad/"
    print("LOW CAD")

elif (simulation == 1):
    shx = 1024
    shy = 1024
    shz = 1024
    hpx = 368
    snap = np.arange(580000,585600,200, dtype=int)
    dir3D ="peacock/3D/hi_cad/"
    odir="forces/hires/"
    print("HIGH CADENCE")

elif(simulation == 2):
    shx = 512
    shy = 512
    shz = 512
    hpx = 180
    snap= np.arange(540000,589800,200,dtype=int)
    dir3D ='lowres/3D/'
    odir  ='forces_s/'
    print("LOWRES")
    
elif(simulation == 3):
    shx = 512
    shy = 512
    shz = 512
    hpx = 180
    snap= np.arange(544000,550500,50,dtype=int)
    dir3D ='hi_cad_snaps'
    odir  ='/scratch/crobu/fake_peacocks/forces_helita/'
    print("LOWRES hi cad")
 ###################################################################  


coeff = np.round(1024/shx).astype(int)


dx  = 40000./shx *1e5 #cm/px
dy  = 40000./shy *1e5
dz  = 22000./shz *1e5
sol_g = 274*1e2 #cm/s
b = simul.simul(dir3D)

# t0=1
# t1=1


for s in snap[0:]:
    print(s)
    
    x,y,z = 0,1,2

    # atmos = b.read_atmos(s,shx=shx,shy=shy,shz=shz, read_ord='C', template=(2,0,1))

    # By  = atmos["Bz"]
    # Bx  = atmos["Bx"]
    # Bz  = atmos["By"]
    # rho = atmos["rho"]
    # P   = atmos["P"]
    
    snapname = '.{:6d}'.format(s)

    dd=chcode.MuramAtmos(template=snapname,prim=snapname,fdir= dir3D, sel_units='cgs')

    Bx  = dd.trans2comm('bx')
    By  = dd.trans2comm('by')
    Bz  = dd.trans2comm('bz') 
    # rho = dd.trans2comm('rho') 
    # P   = dd.trans2comm('pg') 


    btot2 = By**2+Bx**2+Bz**2

    # dxBx = np.gradient(Bx,dx,axis=x)
    # dyBy = np.gradient(By,dy,axis=y)
    # dzBz = np.gradient(Bz,dz,axis=z)
        
    dyBx = np.gradient(Bx,dy,axis=y)
    dzBx = np.gradient(Bx,dz,axis=z)
    
    dyBz = np.gradient(Bz,dy,axis=y)
    dxBz = np.gradient(Bz,dx,axis=x)
    
    dxBy = np.gradient(By,dx,axis=x)
    dzBy = np.gradient(By,dz,axis=z)
    


    # Tmx = (Bx*dxBx+By*dyBx+Bz*dzBx)/(4*math.pi)
    # Tmy = (Bx*dxBy+By*dyBy+Bz*dzBy)/(4*math.pi)
    # Tmz = (Bx*dxBz+By*dyBz+Bz*dzBz)/(4*math.pi)
    # aaa= writefits(np.array((Tmx,Tmy,Tmz)), odir+"Tm_"+str(s)+".fits") 
    
    # del Tmx
    # del Tmy
    # del Tmz
    # gc.collect()
    
    # Pmx = np.gradient(btot2,dx,axis=x)/(8*math.pi) 
    # Pmy = np.gradient(btot2,dy,axis=y)/(8*math.pi)
    # Pmz = np.gradient(btot2,dz,axis=z)/(8*math.pi)
    # aaa= writefits(np.array((Pmx,Pmy,Pmz)), odir+"Pm_"+str(s)+".fits")
    
    # del Pmx
    # del Pmy
    # del Pmz 
    # gc.collect()
    
    # Pgx = np.gradient(P,dx,axis=x)
    # Pgy = np.gradient(P,dy,axis=y)
    # Pgz=  np.gradient(P,dz,axis=z)
    # aaa= writefits(np.array((Pgx,Pgy,Pgz)), odir+"Pg_"+str(s)+".fits")
    
    # del Pgx
    # del Pgy
    # del Pgz
    # gc.collect()
    
    Jx = dyBz - dzBy
    Jy = dzBx - dxBz
    Jz = dxBy - dyBx
    
    #CL= np.sqrt(Jx**2+Jy**2+Jz**2)/np.sqrt(btot2)
    aaa= writefits(np.array((Jx,Jy,Jz)), odir+"J_"+str(s)+".fits")
    #aaa= writefits(CL, odir+"CL_"+str(s)+".fits")
    
    # Lfx = (Jy*Bz - Jz*By)/(4*math.pi)
    # Lfy = (Jz*Bx - Jx*Bz)/(4*math.pi)
    # Lfz = (Jx*By - Jy*Bx)/(4*math.pi)
    # aaa= writefits(np.array((Lfx,Lfy,Lfz)), odir+"Lf_"+str(s)+".fits")


    # q= Tmz
    # h= 250
    # mx = (q[:,:,h].max()-10*q[:,:,h].max()/100.)
    # mn = -mx
    # plt.imshow(q[:,:,h], origin="lower", cmap="RdBu")

