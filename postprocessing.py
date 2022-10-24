"""
NAMESTRING
"""

import argparse
from datetime import datetime
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

def to_latlon(data, landmask): # TODO to climfill package
    shape = landmask.shape
    landlat, landlon = np.where(landmask)
    tmp = xr.DataArray(np.full((data.coords['variable'].size,data.coords['time'].size,shape[0],shape[1]),np.nan), 
                       coords=[data.coords['variable'], data.coords['time'], 
                               landmask.coords['lat'], landmask.coords['lon']], 
                       dims=['variable','time','lat','lon'])
    tmp.values[:,:,landlat,landlon] = data
    return tmp

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
print(f'{datetime.now()} read data...')
data = xr.open_mfdataset(f'{esapath}{testcase}/clusters/datacluster_iter_c*.nc', 
                         combine='nested', concat_dim='datapoints').to_array().load()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# unstack
print(f'{datetime.now()} unstack...')
data = data.set_index(datapoints=('time','landpoints')).unstack('datapoints')
data = to_latlon(data, landmask)

# save
print(f'{datetime.now()} save...')
data = data.to_dataset('variable')
data.to_netcdf(f'{esapath}{testcase}/data_climfilled.nc')
