"""
NAMESTRING
"""

from datetime import datetime
import argparse

import numpy as np
import xarray as xr

from climfill.verification import delete_minicubes

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# crossval settings
frac_mis = 0.1 
ncubes = 20
verification_year = '2004'
crossvalidation_year = '2005'

# read data cube
print(f'{datetime.now()} read data...')
data = xr.open_dataset(f'{esapath}/data_orig.nc').to_array()
mask = xr.open_dataset(f'{esapath}/mask_orig.nc').to_array()
#varnames =  list(data.keys()) # unsorted!
varnames = data.coords['variable'].values

# delete values for verification
for varname in varnames:
    print(f'{datetime.now()} verification {varname}...')
    tmp = delete_minicubes(mask.sel(time=verification_year, variable=varname).drop('variable').load(),
                           frac_mis, ncubes)
    mask.loc[dict(variable=varname, time=verification_year)] = tmp

data = data.where(np.logical_not(mask))

# delete values for cross-validation
for varname in varnames:
    print(f'{datetime.now()} cross-validation {varname}...')
    tmp = delete_minicubes(mask.sel(time=crossvalidation_year, variable=varname).drop('variable').load(),
                           frac_mis, ncubes)
    mask.loc[dict(variable=varname, time=crossvalidation_year)] = tmp

data = data.where(np.logical_not(mask))

# save crossval cube for gap-filling
print(f'{datetime.now()} save...')
data.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/data_crossval.nc')
mask.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/mask_crossval.nc')
