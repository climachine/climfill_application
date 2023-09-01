"""
NAMESTRING
"""

import argparse
from datetime import datetime
import numpy as np
import xarray as xr

from climfill.postprocessing import to_latlon

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
parser.add_argument('--set', '-s', dest='veriset', type=str)
parser.set_defaults(veriset=None)
args = parser.parse_args()
testcase = args.testcase
veriset = args.veriset

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
if veriset is None:
    filepath = f'{esapath}{testcase}/clusters/'
    savepath = f'{esapath}{testcase}/'
    maskorig = f'{esapath}mask_orig.nc'
    maskinit = f'{esapath}{testcase}/mask_initguess.nc'
    mask_orig = xr.open_dataset(maskorig)
    mask_init = xr.open_dataset(maskinit)
else:
    filepath = f'{esapath}{testcase}/verification/set{veriset}/clusters/'
    savepath = f'{esapath}{testcase}/verification/set{veriset}/'

# read data
print(f'{datetime.now()} read data...')
data = xr.open_mfdataset(f'{filepath}datacluster_iter_c*.nc', 
                         combine='nested', concat_dim='datapoints').to_array().load()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# unstack
print(f'{datetime.now()} unstack...')
data = data.set_index(datapoints=('time','landpoints')).unstack('datapoints')
data = to_latlon(data, landmask)

# create flag
flag = xr.full_like(mask_orig, np.nan)
flag = flag.where(np.logical_not((mask_orig == 0) & (mask_init == 0)), 0) # is obs
flag = flag.where(np.logical_not(mask_orig == 1), 2) # is gap-filled
flag = flag.where(np.logical_not((flag == 2) & (mask_init == 0)), 1) # is init
varnames = list(mask_orig.keys())
varnames_flag = [varname + '_flag' for varname in varnames]
flag = flag.rename(dict(zip(varnames, varnames_flag)))
flag.attrs = {'flag legend': '0 is observed, 1 is observed in less than 15d in this month and therefore gap-filled, 2 is not observed and therefore gap-filled'}

# save
print(f'{datetime.now()} save...')
data = data.to_dataset('variable')
data = data.merge(flag)
#data.to_netcdf(f'{savepath}/data_climfilled.nc')
#flag.to_netcdf(f'{savepath}/flag_climfilled.nc')

# save individual year names
for year in np.unique(data['time.year']):
    tmp = data.where(data['time.year'] == year, drop=True)
    tmp.to_netcdf(f'{savepath}/data_climfilled_{year}.nc')

