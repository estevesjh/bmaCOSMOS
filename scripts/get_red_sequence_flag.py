# This code loads the columns is_on_red_sequence_gr/ri
# into the tables bma_cosmodc2_cluster/field_galaxies.fits

import healpy
import numpy as np
import GCRCatalogs

from time import time, sleep
import pandas as pd
import matplotlib.pyplot as plt

from astropy.table import Table, vstack, join
from astropy.io.fits import getdata

import esutil

## Output Table
outfile = 'matched_table.fits'
## jid is the index of input tables
## for galaxy index -> jid
## for fields index -> jid-len(gal)

## Output Columns
columns = ['galaxy_id','ra','dec','photoz_mask','stellar_mass','redshift','baseDC2/is_on_red_sequence_gr', 'baseDC2/is_on_red_sequence_ri','is_central']
columns+= ['mag_true_r_sdss', 'mag_r_sdss', 'Mag_true_r_sdss_z0'] 
columns+= ['mag_true_i_sdss', 'mag_i_sdss', 'Mag_true_i_sdss_z0'] 
columns+= ['Mag_true_g_lsst_z0','Mag_true_r_lsst_z0','Mag_true_i_lsst_z0'] 
columns+= ['mag_true_g_lsst','mag_true_r_lsst','mag_true_i_lsst'] 

######################################################################
## Auxiliary Functions
def match_samples(gals,data,sep=0.15/3600):
    ra_cluster,dec_cluster = gals['RA'], gals['DEC']
    ra_galaxy ,dec_galaxy  = data['ra'], data['dec']    
    match = match_sky_coordinates(ra_cluster,dec_cluster,ra_galaxy,dec_galaxy,radius=sep)
    ndata = data[match[1]]
    
    ndata['GID'] = gals['GID'][match[0]]
    ndata['ran'] = gals['RA'][match[0]]
    ndata['decn']= gals['DEC'][match[0]]
    ndata['jid'] = gals['jid'][match[0]]  ## indexes of the original sample
    return ndata

def match_sky_coordinates(ra_cluster,dec_cluster,ra_galaxy,dec_galaxy,radius=0.2/3600):
    sep = (0.25/60/60) # 1 arcsec
    depth=10
    h=esutil.htm.HTM(depth)
    #Inner match
    m1i,m2i,disti=h.match(ra_cluster,dec_cluster,ra_galaxy,dec_galaxy,radius=sep,maxmatch=1)
    return [m1i,m2i]

def apply_cuts(data):
    smass= data['stellar_mass']
    zgal = data['redshift']
    mask = (smass>=7e9)
    mask&= (zgal>=0.098)&(zgal<=1.02)
    return data[mask]

def apply_photoz_mask(data,columns):
    mask = (data['photoz_mask']).copy()
    nall = len(mask)
    for col in columns:
        if len(data[col])==nall:
            data[col] = data[col][mask]
    return Table(data)

def get_galaxy_given_healpix_pixel(hpx,columns):
    ### getting healpix number neighbours
    print(5*'---')
    print('Healpixel: %i'%(hpx))
    data = gc.get_quantities(columns,native_filters=['healpix_pixel==%s'%(str(hpx))])
    data = apply_photoz_mask(data,columns) ## astropy table
    return data


######################################################################
## Load the tables bma_cosmodc2_cluster/field_galaxies.fits
gal = Table(getdata('./data/cosmoDC2_bma_cluster_galaxies.fits'))
galb= Table(getdata('./data/cosmoDC2_bma_field_galaxies.fits'))

gal['jid']  = np.arange(len(gal),dtype=np.int64)
galb['jid'] = np.arange(len(gal),len(gal)+len(galb),dtype=np.int64)

## GCR Catalog
gc = GCRCatalogs.load_catalog('cosmoDC2_v1.1.4_image_with_photozs_v1')
tiles = gc._healpix_files.keys()
tiles_array = np.unique(np.array([toto[1] for toto in tiles]))
nfields = len(tiles_array)

## Creating Sky Coordinates Catalog
g = vstack([gal,galb])
# _, uindex = np.unique(g0['RA','DEC'],return_index=True)
# g = g0[uindex]

## fraction of repeated objects on sky coordinates
# frep = (1-1.*len(g)/len(g0))*100
# print('percentage of repeated objects on input tables: %.2f %%'%(frep))


## Loading cosmoDC2 files
t0   = time()
out  = []
nsize= 0
for count,hpx in enumerate(tiles_array):
    data = get_galaxy_given_healpix_pixel(hpx,columns)
    data = apply_cuts(data)
    data = match_samples(g,data,sep=0.2/3600)
    
    partial_time= (time()-t0)/60.
    nsize+= len(data)
    print('Field %i/%i'%(count+1,nfields))
    print('recovered percentage: %.2f %%'% (100.*nsize/len(g) ))
    print('partial time: %.2f min \n'%(partial_time))
    if len(data)>0:
        out.append(data)
    else:
        print('Error: %i empty field'%hpx)
    
    if nsize>=1.2*len(g):
        break
        
outtable = vstack(out)
_,index  = np.unique(outtable['jid'],return_index=True)
data     = outtable[index]
print('recovered percentage: %.2f %%'% (100.*len(data)/len(g) ))

data.write(outfile,format='fits',overwrite=True)

total_time = (time()-t0)/60.
print('total time: %2.f s'%(total_time))

