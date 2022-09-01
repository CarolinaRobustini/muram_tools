import numpy as np
from numpy import linalg as LA
from sys import exit
from ipdb import set_trace as stop
from scipy.interpolate import RegularGridInterpolator as regi

class b_line:
    """
    Methods:
    constructor(): 
    read_atmos():
    """
    def __init__(self,bx,by,bz,x,y,z,r0,ds,xyperiodic,niter, direction):
        self.bx         = bx
        self.by         = by
        self.bz         = bz
        self.x          = x
        self.y          = y
        self.z          = z
        self.r0         = r0
        self.ds         = ds
        self.xyperiodic = xyperiodic
        self.niter      = niter
        self.direction  = direction

        # if 2D box force r0 to x,y,z coordinate
        if (len(self.r0) == 2):
            self.r0 = np.array([self.r0[0],self.y[0],self.r0[1]])    
        
    def get_box_dims(self):
        if (len(self.bx.shape) == 3):
            nx = self.bx.shape[0]      
            ny = self.bx.shape[1]
            nz = self.bx.shape[2]
        else:
            nx = self.bx.shape[0]
            ny = self.bx.shape[1]
            nz = 1
            print(self.bx)

        return (nx,ny,nz)
         
    def get_d(self,nx,ny,nz):
        dx = self.x[1]-self.x[0]
        dy = self.y[1]-self.y[0]
        dz = np.zeros(nz-1)
        for i in range (0,nz-1):
            dz[i]=self.z[i+1]-self.z[i]
        if (nx != len(self.x)):
            print('nx != n_elements(x)')
        if (ny != len(self.y)):
            print('ny != n_elements(y)')
        if (nx != len(self.x)):
            print('nz != n_elements(z)')
        if (self.ds == 0):
            dds = min(dx,dy,min(dz))
            if (self.direction < 0): dds = -dds
            return (dx,dy,dz,dds)   
        else:
            if (self.direction < 0): self.ds = self.ds*(-1.)
            return (dx,dy,dz,self.ds)  

    def reshape_b(self):
        self.bx  = self.bx.reshape(self.nx,self.ny,self.nz)
        self.by  = self.by.reshape(self.nx,self.ny,self.nz)
        self.bz  = self.bz.reshape(self.nx,self.ny,self.nz)
        return(self.bx,self.by,self.bz)
    
    def check_boundaries(self,nx,ny):
        if (self.xyperiodic):    
            try:
                if (self.bx[0,0,0] != self.bx[nx-1,0,0]):
                    raise ValueError('bx[0,0,0] ne bx[nx-1,0,0]')
            except (ValueError, IndexError):
                exit('Could not complete request.')
            try:
                if (self.by[0,0,0] != self.by[0,ny-1,0]):
                    raise ValueError('by[0,0,0] ne by[0,ny-1,0]')
            except (ValueError, IndexError):
                exit('Could not complete request.')
            try:
                if (self.bz[0,0,0] != self.bz[nx-1,ny-1,0]):
                    raise ValueError('bx[0,0,0] ne bx[nx-1,ny-1,0]')
            except (ValueError, IndexError):
                exit('Could not complete request.')
            
    def xymax(self,nx,ny):
        xmax = self.x[nx-1]
        ymax = self.y[ny-1]
        xymax = [xmax,ymax]
        return xymax
    
    def zminmax(self):
        zmin = np.min(self.z)
        zmax = np.max(self.z)  
        zminmax = [zmin,zmax]
        return zminmax
        

    def check_seed_initial(self, nx,ny):
        xmax = self.xymax(nx,ny)[0]
        ymax = self.xymax(nx,ny)[1]
        zmin = self.zminmax()[0]
        zmax = self.zminmax()[1]

        r2   = self.r0      
        try:
            if ((r2[0] < self.x[0]) or (r2[0] > xmax) or (r2[1] < self.y[0]) or (r2[1] > ymax) or (r2[2] < zmin) or (r2[2] > zmax)):
                print(r2)
                print(self.x[0],xmax, self.y[0],ymax,zmin,zmax)
                raise ValueError('Starting point is outside the spatial domain')
        except (ValueError, IndexError):
            exit('Could not complete request.')
        
        return r2
    
    
    def check_seed(self, nx,ny,r2, count):
        xmax = self.xymax(nx,ny)[0]
        ymax = self.xymax(nx,ny)[1]
        zmin = self.zminmax()[0]
        zmax = self.zminmax()[1]
        if (self.xyperiodic):
            inside = (r2[2] > zmin) and (r2[2] < zmax) and (count < self.niter-2)
        else:  
            inside = True
            if (r2[0] < self.x[0]) or (r2[0] > xmax): 
                inside = False
            if (r2[1] < self.y[0]) or (r2[1] > ymax):
                inside = False
            if (r2[2] < zmin) or (r2[2] > zmax):
                inside = False 
            if (count >= self.niter-2): 
                inside = False
        return inside

        
    def value_locate(self,vect,values): 
        if (type(values) != list): values = [values]              
        #vector must be monotonic
        nlv = len(vect)
        index = np.zeros(len(values))
        if (vect[0]< vect[1]):
            for value in values:
                j = values.index(value)
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
                j = values.index(value)
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
 
   
    def trilinear(self, data,xx,yy,zz):
        x2=xx
        y2=yy
        z2=zz
        if(self.z[1]> self.z[0]):
            zn =self.z
        else:
            zn=-self.z
            z2=-z2
        
        dx=self.x[1]-self.x[0]
        dy=self.y[1]-self.y[0]
    
        nx=len(self.x)
        ny=len(self.y)
        nz=len(zn)
    
        #fractional coordinates , force out of bounds to edge
    
        fx = (x2-min(self.x)) / dx 
        fy = (y2-min(self.y)) / dy 
    
        if(len(fx.shape)>0):
            for i in range(0,len([fx])):
                if (fx[i] < 0): fx[i] = 0
                if (fx[i] > nx-1): fx[i] = nx-1
            for i in range(0,len([fy])):
                if (fy[i] < 0): fy[i] = 0
                if (fy[i] > ny-1): fy[i] = ny-1            
        else:
            if (fx < 0): fx = 0
            if (fx > nx-1): fx = nx-1
            if (fy < 0): fy = 0
            if (fy > ny-1): fy = ny-1     
            
         
        zl = self.value_locate(zn,z2).astype(int)
        if (type(xx) == list):
            nl = len(xx)
            fz = np.zeros(nl)
            for i in range(0,nl):
                if (zl[i] == -1):
                    fz[i]=0 
                else:
                    if (zl[i] == nz-1):
                        fz[i] = nz-1 
                    else:
                        fz[i] = zl[i] + (z2[i]- zn[zl[i]]) / (zn[zl[i]+1]-zn[zl[i]])  
                
        else:
            if (zl == -1):
                fz = 0
            else:
                if (zl == nz-1):
                    fz = nz-1
                else:
                    fz = zl + (z2- zn[zl]) / (zn[zl[0]+1]-zn[zl[0]]) 
    
                    
        dimx = data.shape[0]
        dimy = data.shape[1]
        dimz = data.shape[2]
        
        va = np.arange(0,dimx)
        vb = np.arange(0,dimy)      
        vc = np.arange(0,dimz)
        f = regi((va, vb, vc), data)
        pts = (fx,fy,fz)
        intdat = f(pts)
     
        return intdat


    def check_if_periodic(self):
        return self.xyperiodic     
    
    

def fun_b_line(bx,by,bz,x,y,z,r0,ds=0,xyperiodic = False,niter=500, direction=1):
    """

  
    Parameters: 
    bx: data cube of the x-component of B field
    by: data cube of the y-component of B field
    bz: data cube of the z-component of B field
    x,y,z: coordinate axes for each cuve dimension
    r0: seeds
    ds:increment value of the arc length along the field lines( default=min(Delta x, Delta y, Delta z)
    xyperiodic: Default is False. If True the box is periodic in x and y
    niter = number of steps a field line is traced. default is 500
    direction: default=1. If any negative number the integration occurs along the opposite direction of magnetic field
  
    Returns: 
    r: array of coordinates that show the path of your field lines.
    """
    
    a = b_line(bx,by,bz,x,y,z,r0,ds,xyperiodic,niter, direction)
    
    nx = a.get_box_dims()[0]
    ny = a.get_box_dims()[1]
    nz = a.get_box_dims()[2]

    # dx = a.get_d(nx,ny,nz)[0]
    # dy = a.get_d(nx,ny,nz)[1]
    # dz = a.get_d(nx,ny,nz)[2]
    ds = a.get_d(nx,ny,nz)[3]

    
    xmax = a.xymax(nx,ny)[0]
    ymax = a.xymax(nx,ny)[1]
    # zmin = a.zminmax()[0]
    # zmax = a.zminmax()[1]
    

    if (len(bx.shape) != 3):
        bx = a.reshape_b[0]
        by = a.reshape_b[1]
        bz = a.reshape_b[2]


    dum = a.check_boundaries(nx,ny)

    r2 = a.check_seed_initial(nx,ny)
    r = np.array(r2).reshape(3)

    b2 = np.array([a.trilinear(bx,r2[0],r2[1],r2[2]),a.trilinear(by,r2[0],r2[1],r2[2]),a.trilinear(bz,r2[0],r2[1],r2[2])])
    b2 = np.transpose(b2/ LA.norm(b2))
    
    count = 0
    
    seed_ok = a.check_seed(nx,ny,r2, count)
    
    while seed_ok:
        r1=r2               
        # b1=b2     
        # iter2=0     

        # 2nd order runge kutta     
        rhalf=(r1+0.5*ds*b2).reshape(3)
        bhalf=[a.trilinear(bx,rhalf[0],rhalf[1],rhalf[2]),a.trilinear(by,rhalf[0],rhalf[1],rhalf[2]),a.trilinear(bz,rhalf[0],rhalf[1],rhalf[2])]    
        bhalf= np.transpose(bhalf/LA.norm(bhalf)).reshape(3)
        r2=r1+ds*bhalf

        if (xyperiodic):
            r2[0]= (r2[0]+xmax) % xmax
            r2[1]= (r2[1]+ymax) % ymax

        r = np.concatenate((r.T, r2.T), axis=0)
        
    
        seed_ok = a.check_seed(nx,ny,r2, count)
        
        count=count+1
        
    r = r.reshape(count+1,3)
    return r


    
 

 
