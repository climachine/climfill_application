"""
NAMESTRING
"""

from datetime import datetime
from glob import glob

import numpy as np
import xarray as xr

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','burned_area','temperature_obs',
            'precipitation_obs','snow_cover_fraction',
            'diurnal_temperature_range','landcover']

# read data
print(f'{datetime.now()} read data...')
filenames = [esapath + varname + '.nc' for varname in varnames]
data = xr.open_mfdataset(filenames)
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# create mask
print(f'{datetime.now()} create mask...')
mask = np.isnan(data)

# set ocean to nan
print(f'{datetime.now()} set ocean to nan...')
data = data.where(landmask, np.nan)
mask = mask.where(landmask, np.nan)

# save original cube for crossvalidation and verification
print(f'{datetime.now()} save orig data...')
data.to_netcdf(f'{esapath}data_orig.nc')
mask.to_netcdf(f'{esapath}mask_orig.nc')
