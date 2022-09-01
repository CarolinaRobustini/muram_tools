from My_functions import *
from b_line import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from matplotlib.gridspec import GridSpec
from matplotlib import rc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import style
import matplotlib.colors as colors
import m3d
import math
import simul
import glob, os
import re



snap= np.arange(544000,550500,50,dtype=int)
dir3D ='hi_cad_snaps/'
shx = 512
shy = 512
shz = 512
b = simul.simul(dir3D)    

dx  = 40000./shx *1e-3#Mm/px
dy  = 40000./shy *1e-3
dz  = 22000./shz *1e-3

if(0):
    idir = "corks_flines/"
    files = []
    
    #os.chdir(idir)
    for file in sorted(glob.glob(idir+"*.fits")):
        files.append(file)
        
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    r = readfits(files[2])
    nt, npoint, nline, ncoord = r.shape
    
    
    for file in files: 
        start = int(file.split('/')[1].split('_')[0])
        fin = start+50
        r[:,:,start:fin,:] = readfits(file)[:,:,0:50,:]
    
    aaa = writefits(r, "corks_flines/tot_traj_rho300-350_flines.fits")
    
    curvs = np.zeros((nt,npoint,nline))
    
    for t in range (0,nt):
        print(t,nt)
        for l in range(0, fin):
            
            x = r[t,:,l,0]
            y = r[t,:,l,1]
            z = r[t,:,l,2]
            
            curvs[t,:,l] = calc_curv(x,y,z)
    
    a=np.where(np.isnan(curvs) == True)
    curvs[a[0],a[1],a[2]] = -1
    
    aaa = writefits(curvs, "corks_flines/tot_curvs.fits")
    
else:
    curvs = readfits("corks_flines/tot_curvs.fits")
    r = readfits("corks_flines/tot_traj_rho300-350_flines.fits")
    nt, npoint, nline = curvs.shape


# labels_orig = readfits('labels_incl.fits')
# labels = labels_orig[::100]

# in_l = np.where(labels == 0)[0]#[0:-1:3]

# # max_curvs= np.where(curvs[:,150:-150,in_l] > 280)
# # in_l = list(set(list(max_curvs[2])))



# in_l = np.arange(0,420,50)
in_box= np.where(r[:,:,:,0]<150)

# in_l = [0,50,100,200,350]

r[in_box[0],in_box[1],in_box[2],:]=[0,0,0]

index= readfits("indx_corks.fits")

in_l = index[0:50].astype(int)
outd ="label/jumpLf/"


for t in range(0,nt):
    print(t,nt)
    rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r"\usepackage{tgheros} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}"
    plt.rcParams["font.size"] = 9
    fig= plt.figure(figsize=(8.8*2,8.8*2))
    fig,((ax1, ax2), (ax4, ax3)) = plt.subplots(1 1,figsize=(8.8*2,8.8*2))
    
   
    # gs = matplotlib.gridspec.GridSpec(4,1, figure=fig)
    # ax1 = fig.add_subplot(gs[0, :])
    # ax2 = fig.add_subplot(gs[1, :])
    # ax4 = fig.add_subplot(gs[2, :])
    # ax3 = fig.add_subplot(gs[3, :])

    # rho = b.read_atmos_rho(snap[t],shx=shx,shy=shy,shz=shz,template=(2,0,1))
    # T = b.read_atmos_T(snap[t],shx=shx,shy=shy,shz=shz,template=(2,0,1))
    # Bz = b.read_atmos_By(snap[t],shx=shx,shy=shy,shz=shz,template=(2,0,1))
    # vz = b.read_atmos_vy(snap[t],shx=shx,shy=shy,shz=shz,template=(2,0,1))
    # vx = b.read_atmos_vx(snap[t],shx=shx,shy=shy,shz=shz,template=(2,0,1))
    # vy = b.read_atmos_vz(snap[t],shx=shx,shy=shy,shz=shz,template=(2,0,1))
    # Qres = b.read_atmos_Qres(snap[t],shx=shx,shy=shy,shz=shz,template=(2,0,1))
    # # Qtot = b.read_atmos_Qtot(snap[t],shx=shx,shy=shy,shz=shz,template=(2,0,1))
    
    
    va = np.arange(0,shx)
    vb = np.arange(0,shy)      
    vc = np.arange(0,shz)
    fvz = regi((va, vb, vc), vz)
    
    
    colormap = plt.cm.viridis #or any other colormap
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=len(in_l))
    
    hcut  = 190
    xcut  = 0
    ycut  = 58
    ycut1 = 380
    
    p1=(405,280)
    m = 0.85
    q = p1[1]-m*p1[0]
    x0c = int((ycut-q)/m)
    x1c =int((512-q)/m)
    
    qt = p1[1]+m*p1[0]
    xtc = (qt -ycut1-ycut)/m
    xt = np.arange(xtc+1,512, dtype=int)
    yt = (-m*xt+qt).astype(int)
    
    
    xp = np.arange(x0c,512, dtype=int)
    yp = (xp*m + q).astype(int)
    alpha = np.arctan(m)
    
    zr = np.arange(0,shz-hcut)*dz
    xr = np.arange(0,shx)*dx
    yr = np.arange(0,ycut1-ycut)*dy
    zdiag = np.arange(0,xp.shape[0])*dx/np.cos(alpha)
    zdiagt = np.arange(0,xt.shape[0])*dx/np.cos(alpha)
    
    
   
    ax1.imshow(np.transpose(np.log10(rho[xcut:,p1[1],hcut:])), origin="lower", cmap="gist_gray")  
    #ax1.imshow(np.transpose(vz[xcut:,p1[1],hcut:]*1e-5), origin="lower", cmap="RdBu",vmin=-100, vmax=100)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    for nline in in_l:
        ax1.scatter(r[t,:,nline,0]-xcut,r[t,:,nline,2]+120-hcut,  c= np.ones(npoint)*nline,cmap=colormap, norm=normalize, s=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Z')   
    
    ax1.pcolormesh(zdiag,zr,np.transpose(np.log10(rho[xp,yp,hcut:])), origin="lower", cmap="gist_gray")  
    
    
    # module = np.sqrt((xp[-1]-xp[0])**2+(yp[-1]-yp[0])**2)
    # xx = ((r[t,:,:,0]-xcut-x0c)*np.cos(alpha)- np.sin(alpha)*(r[t,:,:,1]-ycut1-ycut))*(shx-x0c)/module
    # ax1.contour(zdiag,zr,np.transpose(np.log10(T[xp,yp,hcut:])),levels=[4], colors="black")
    # im1 = ax1.pcolormesh(zdiag,zr,np.transpose(Bz[xp,yp,hcut:]), cmap="RdBu",vmin=-200, vmax=200, shading="auto")
    # ax1.set_ylim(bottom=0)
    # ax1.set_xlim(left=0)
    # i=0
    # for nline in in_l:
    #     ax1.scatter(xx[:,nline]*dx/np.cos(alpha),(r[t,:,nline,2]+120-hcut)*dz,  c= np.ones(npoint)*i,cmap=colormap, norm=normalize, s=0.5)
    #     i+=1
    # ax1.set_xlabel('Line 1 [Mm]')
    # ax1.set_ylabel('Z [Mm]')  
    # divider1 = make_axes_locatable(ax1)
    # cax1 = divider1.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im1, cax=cax1, orientation='vertical', label="Bz [G]")
    
    # #ax2.imshow(np.transpose(np.log10(rho[p1[0],:,hcut:])), origin="lower", cmap="gist_gray")
    # coeff = np.cos(np.arctan(-m) - np.arctan2(vy[xt,yt,hcut:],vx[xt,yt,hcut:]))
    # vh = np.sqrt(vx[xt,yt,hcut:]**2+vy[xt,yt,hcut:]**2)*coeff
    
    # modulet = np.sqrt((xt[-1]-xt[0])**2+(yt[-1]-yt[0])**2)
    # xxt = ((r[t,:,:,0]-xcut-xtc)*np.cos(alpha)- np.sin(alpha)*(r[t,:,:,1]-ycut1-ycut))*(shx-xtc)/modulet
    # # coeff = np.cos(math.pi/2 - np.arctan2(vy[p1[0],:,hcut:],vx[p1[0],:,hcut:]))
    # # vh = np.sqrt(vx[p1[0],:,hcut:]**2+vy[p1[0],:,hcut:]**2)*coeff
    # im2 = ax2.pcolormesh(zdiagt,zr,np.transpose(vh*1e-5), cmap="RdBu", vmin=-100,vmax=100,shading="auto")
    # ax2.contour(zdiagt,zr,np.transpose(np.log10(T[xt,yt,hcut:])),levels=[4], colors="black")
    # ax2.set_ylim(bottom=0)
    # ax2.set_xlim(right=zdiagt[-1])
    # i=0
    # for nline in in_l:
    #     ax2.scatter(xxt[:,nline]*dx/np.cos(alpha),(r[t,:,nline,2]+120-hcut)*dz, c= np.ones(npoint)*i,cmap=colormap, norm=normalize, s=0.5)
    #     i+=1
    # ax2.set_xlabel('Line 2 [Mm]')
    # ax2.set_ylabel('Z [Mm]')  

    # divider2 = make_axes_locatable(ax2)
    # cax2 = divider2.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im2, cax=cax2, orientation='vertical',label= r'vh [km s$^{-1}$]')
  
    
    # hrho=210
    # im3 = ax3.pcolormesh(xr,yr,np.transpose(Bz[xcut:,ycut:ycut1,hrho]), cmap="gist_gray",vmin=-500,vmax=500, shading="auto")  
    # i=0
    # for nline in in_l:
    #     in_rho= np.where((r[t,:,nline,2]+120)> hrho)
    #     ax3.scatter((r[t,in_rho[0],nline,0]-xcut)*dx,(r[t,in_rho[0],nline,1]-ycut)*dy,  c= np.ones(in_rho[0].shape[0])*i, cmap=colormap, norm=normalize, s=0.5)
    #     i+=1
    # # ax3.vlines(p1[0]-xcut, ymin=0, ymax = ycut1-ycut, colors= "red" )
    # # ax3.hlines(p1[1]-ycut, xmin=0, xmax = 512-xcut, colors="red")
    # ax3.plot((xt-xcut)*dx,(yt-ycut)*dy,color="red",label = "Line 2", linestyle="dashed")
    # ax3.plot((xp-xcut)*dx,(yp-ycut)*dy,color="red",label = "Line 1")
    # ax3.set_ylim(top= (ycut1-ycut)*dy, bottom=0)
    # ax3.set_xlim(right= (512-xcut)*dx)
    # ax3.set_xlim(left=0)
    # ax3.set_xlabel('X [Mm]')
    # ax3.set_ylabel('Y [Mm]')
    # ax3.legend()
    # ax3.text(1,1,"Z= "+str('{:3.2f}'.format((hrho-hcut)*dz))+" Mm",backgroundcolor="white")
    # divider3 = make_axes_locatable(ax3)
    # cax3 = divider3.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im3, cax=cax3, orientation='vertical',label= r'Bz [G]')
    

    
 

    # im4 = ax4.pcolormesh(zdiag,zr,np.transpose(vz[xp,yp,hcut:]*1e-5), cmap="RdBu",vmin=-100, vmax=100, shading="auto")  
    # ax4.contour(zdiag,zr,np.transpose(np.log10(T[xp,yp,hcut:])),levels=[4], colors="black")
    # ax4.contour(zdiag,zr,np.transpose(Qres[xp,yp,hcut:]/rho[xp,yp,hcut:]),levels=[0.3], colors="purple")
    # ax4.set_ylim(bottom=0)
    # ax4.set_xlim(left=0)
    # i=0
    # for nline in in_l:
    #     ax4.scatter(xx[:,nline]*dx/np.cos(alpha),(r[t,:,nline,2]+120-hcut)*dz,  c= np.ones(npoint)*i,cmap=colormap, norm=normalize, s=0.5)
    #     i+=1
    # ax4.set_xlabel('Line 1 [Mm]')
    # ax4.set_ylabel('Z [Mm]')  
    # divider4 = make_axes_locatable(ax4)
    # cax4 = divider4.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(im4, cax=cax4, orientation='vertical', label= r'vz [km s$^{-1}$]')
    
    

    
    plt.tight_layout(pad=0.4, w_pad=0.9, h_pad=1.0)
    
    plt.savefig("corks_flines/"+outd+"/"+str(snap[t])+".png",bbox_inches='tight', pad_inches=0.02)