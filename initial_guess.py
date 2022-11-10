"""
initial gapfill
"""

import numpy as np
import xarray as xr
import argparse
from datetime import datetime
from climfill.interpolation import gapfill_thin_plate_spline, gapfill_kriging

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data, NEW: without crossval holes
print(f'{datetime.now()} read data...')
data = xr.open_dataset(f'{esapath}data_orig.nc')
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# xarray/dask issue https://github.com/pydata/xarray/issues/3813
# value assignment only works if non-dask array
data = data.to_array().load()

# read initial guess
print(f'{datetime.now()} read initial guess data ...')
dtr_init = xr.open_dataset(f'{esapath}diurnal_temperature_range_init.nc').to_array()
lst_init = xr.open_dataset(f'{esapath}surface_temperature_init.nc').to_array()
sm_init = xr.open_dataset(f'{esapath}soil_moisture_init.nc').to_array()
t2m_init = xr.open_dataset(f'{esapath}temperature_obs_init.nc').to_array()
pre_init = xr.open_dataset(f'{esapath}precipitation_obs_init.nc').to_array()

# fill 
print(f'{datetime.now()} fill initial guess...')
data.loc['diurnal_temperature_range'] = data.loc['diurnal_temperature_range'].fillna(dtr_init).squeeze()
data.loc['surface_temperature'] = data.loc['surface_temperature'].fillna(lst_init).squeeze()
data.loc['soil_moisture'] = data.loc['soil_moisture'].fillna(sm_init).squeeze()
data.loc['temperature_obs'] = data.loc['temperature_obs'].fillna(t2m_init).squeeze()
data.loc['precipitation_obs'] = data.loc['precipitation_obs'].fillna(pre_init).squeeze()

# save to netcdf
print(f'{datetime.now()} save initial guess...')
data.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/data_initguess.nc')
