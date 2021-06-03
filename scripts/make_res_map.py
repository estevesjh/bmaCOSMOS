import numpy
import h
import sys
sys.path.append('/home/s1/jesteves/git/ccopa/python/bma')

import loadPopColors
from smass import *

libPath= '/data/des61.a/data/pbarchi/galaxyClusters/simha_miles_Nov2016/'
splines, zmet = loadPopColors.doAll(libPath, lib="miles")

dz   = 0.01
zvec = np.arange(0.1,1.+dz,dz)

spline_list_Dict = [get_spline(splines[sp],zmet[sp],zvec) for sp in range(len(splines))]

def _get_spline(spline,zmet,zed):
    # for speed
    sgr  = float(spline[0](zed))
    sri  = float(spline[1](zed))
    siz  = float(spline[2](zed))
    sgrr = float(spline[4](zed)) ;# restframe g-r
    sgir = float(spline[5](zed)) ;# restframe g-i
    skii = float(spline[6](zed)) ;# kcorrection: i_o - i_obs
    skri = float(spline[7](zed)) ;# kcorrection: r_o - i_obs
    sml  = float(spline[8](zed)) ;# log(mass/light)  (M_sun/L_sun)
    ssfr = float(spline[9](zed))
    if (ssfr<-20.): ssfr=-20.
    sage_cosmic = float(spline[10](zed))
    sage = float(spline[11](zed))
    szmet = float(zmet)
    return np.array([sgr,sri,siz,sgrr,sgir,skii,skri,sml,ssfr,sage,szmet])


def get_color_min_z(zp,x):
    dgr,dri,diz = [],[],[]
    for i in range(len(splines)):
        out = _get_spline(splines[i],zmet[i],zp)
        dgr.append(x - out[0])
        dri.append(x - out[1])
        diz.append(x - out[2])

    cmin_gr = np.min(np.abs(dgr),axis=0)
    cmin_ri = np.min(np.abs(dri),axis=0)
    cmin_iz = np.min(np.abs(diz),axis=0)
    cmin_c3 = np.min(np.sqrt(np.array(dgr)**2+np.array(dri)**2+np.array(diz)**2) ,axis=0)
    return cmin_gr,cmin_ri,cmin_iz,cmin_c3

zvec = np.linspace(0.1,1.,100)
xvec = np.linspace(-1.,3,3000)

dgr,dri,diz,d3 = [],[],[],[]
for z in zvec:
    c1,c2,c3,ct = get_color_min_z(z,xvec)
    dgr.append(c1)
    dri.append(c2)
    diz.append(c3)
    d3.append(ct)

res_map = dict()
res_map['gr'] = np.array(dgr)
res_map['ri'] = np.array(dri)
res_map['iz'] = np.array(diz)
res_map['d3'] = np.array(d3)
res_map['zvec']    = zvec
res_map['cvec']    = xvec
res_map['indices'] = np.arange(0,zvec.size,1,dtype=np.int)

def get_res(zi,ci,label):
    idx = int(np.interp(zi,res_map['zvec'],res_map['indices']))
    err = np.interp(ci,res_map['cvec'],res_map['d3'][idx,:])
    return err



def save_map(outfile,mymap):
    columns = mymap.keys()
    hf = h5py.File(outfile,'w')
    for col in columns:
        hf.create_dataset('%s/'%col,data=mymap[col][:])

save_map('/data/des61.a/data/johnny/COSMOS/BMA/res_map.hdf5',res_map)
