from My_functions import *
import math
import glob, os
import re
from ipdb import set_trace as stop

class simul:
    """
    Methods:
    constructor(dir): sets initial folder
    read_atmos(time_stamp):reads T P rho vx vy vz e Bx By Bz at a given time
    read_tseries(param): reads the parameter from all the snapshot
    """
    def __init__(self,dirc):
        self.coef = np.sqrt(4*math.pi)
        self.dirc  = dirc

    def read_atmos(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        """
        read_atmos(time_stamp)
        time_stamp must be the number appearing in the snapshot name
        """
        T   = np.memmap(self.dirc+'eosT.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)                     # Temperature in K
        P   = np.memmap(self.dirc+'eosP.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)                     # pressure in dyne/cm^2
        rho = np.memmap(self.dirc+'result_prim_0.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)           # density in g/cm^3
        vx  = np.memmap(self.dirc+'result_prim_1.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)             # vx in cm/s
        vz  = np.memmap(self.dirc+'result_prim_3.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)            # vz in cm/s
        vy  = np.memmap(self.dirc+'result_prim_2.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)             # vy in cm/s !VERTICAL!
        e   = np.memmap(self.dirc+'result_prim_4.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)            # uint in erg/cm^3
        Bx  = np.memmap(self.dirc+'result_prim_5.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)*self.coef   # bx in G - 
        Bz  = np.memmap(self.dirc+'result_prim_7.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)*self.coef   # bz in G
        By  = np.memmap(self.dirc+'result_prim_6.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)*self.coef   # by in G ! VERTICAL !
        # F1  = np.memmap(self.dirc+'result_prim_8.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)
        # F2  = np.memmap(self.dirc+'result_prim_9.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)
        # F3  = np.memmap(self.dirc+'result_prim_10.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)
        
        return({"T":T, "P":P, "rho":rho, "vx":vx, "vy":vy, "vz":vz, "e":e, "Bx":Bx, "By":By, "Bz":Bz})
    
    
    def read_atmos_T(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        T   = np.memmap(self.dirc+'eosT.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)                      # Temperature in K
        return(T)
                
    def read_atmos_P(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        P   = np.memmap(self.dirc+'eosP.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)                      # pressure in dyne/cm^2
        return(P)
    
    def read_atmos_rho(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        rho = np.memmap(self.dirc+'result_prim_0.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)             # density in g/cm^3
        return(rho)
    
    def read_atmos_vx(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        vx  = np.memmap(self.dirc+'result_prim_1.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)             # vx in cm/s
        return(vx)
    
    def read_atmos_vz(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        vz  = np.memmap(self.dirc+'result_prim_3.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)             # vz in cm/s
        return(vz)
    
    def read_atmos_vy(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        vy  = np.memmap(self.dirc+'result_prim_2.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)             # vy in cm/s !VERTICAL!
        return(vy)
    
    def read_atmos_Bx(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        Bx  = np.memmap(self.dirc+'result_prim_5.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)*self.coef   # bx in G - 
        return(Bx)
    
    def read_atmos_By(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        By  = np.memmap(self.dirc+'result_prim_6.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)*self.coef   # by in G ! VERTICAL !
        return(By)
    
    def read_atmos_Bz(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        Bz  = np.memmap(self.dirc+'result_prim_7.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)*self.coef   # bz in G
        return(Bz)
    
    def read_atmos_Qres(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        Qres  = np.memmap(self.dirc+'Qres.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)  
        return(Qres)
    
    def read_atmos_Qtot(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        Qtot  = np.memmap(self.dirc+'Qtot.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)  
        return(Qtot)
    
    def read_atmos_Qvis(self,time_stamp,shx=1024,shy=1024,shz=1024,read_ord='C', template=(0,1,2)):
        Qvis  = np.memmap(self.dirc+'Qvis.'+str(time_stamp), mode='r', shape=(shx,shy,shz), dtype=np.float32, order=read_ord).transpose(template)  
        return(Qvis)
    
    def read_tseries(self,param,x0,x1,z0,z1,shx=1024,shy=1024,shz=1024):
        file_root= {'T':'eosT.',
               'P':'eosP.',
               'rho':'result_prim_0.',
               'vx':'result_prim_1.',
               'vz':'result_prim_2.',
               'vy':'result_prim_3.',
               'e':'result_prim_4.',
               'Bx':'result_prim_5.',
               'By':'result_prim_7.',
               'Bz':'result_prim_6.'}
        
        #os.chdir(self.dirc)
        file_list = []
        time_steps = []

        i=0
        for f  in glob.glob(self.dirc+file_root[param]+"*"):
            file_list.append(str(f))
            # m = re.search('.(.+?)000', str(f))
            # time_steps.append(np.int(m.group(1))

        file_list.sort()
        nt = len(file_list)
        
        res = np.zeros((x1-x0,shz,z1-z0,nt))
        for i in range(0, nt):
            print("t= "+str(i)+"/"+str(nt)+"  file =  "+file_list[i]+"   param= "+param)
            res[:,:,:,i] = np.memmap(file_list[i], mode='r', shape=(shy,shz,shx), dtype=np.float32, order=read_ord).transpose(template)[x0:x1,:,z0:z1]
            
        return(res)



