"""
NAMESTRING
"""

from datetime import datetime
import argparse

import numpy as np
import xarray as xr

#from climfill.verification import delete_minicubes
from climfill.verification import create_minicubes

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# crossval settings
frac_mis = 0.1 
ncubes = 24 # needs to be >= nt
vf_year = '2004'
cv_year = '2005'

# read data cube
print(f'{datetime.now()} read data...')
data = xr.open_dataset(f'{esapath}data_orig.nc').to_array()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# mask needs to be computed separately
mask = np.isnan(data)
mask = mask.where(landmask, np.nan) # is needed for calc of n_obs

# get varnames
#varnames =  list(data.keys()) # unsorted, but works on dataset
varnames = list(data.coords['variable'].values)
varnames.remove('landcover')

# select year
mask = mask.sel(time=slice(vf_year,cv_year)).load()
data = data.sel(time=slice(vf_year,cv_year)).load()

# copy 10 times
datasets = []
for i in range(10):
    datasets.append(data.copy(deep=True))

# create minicubes (on land)
minicubes = create_minicubes(mask.sel(variable='soil_moisture').drop('variable'),
                             ncubes)

# divide cubes on land into ten roughly equally sized sets
cubes_on_land = np.unique(minicubes)
cubes_on_land = cubes_on_land[~np.isnan(cubes_on_land)]

for varname in varnames:

    np.random.shuffle(cubes_on_land)
    random_sets = np.array_split(cubes_on_land, 10)

    for dat, cubeset in zip(datasets, random_sets):

        dat.loc[varname,:,:,:] = dat.loc[varname,:,:,:].where(np.logical_not(minicubes.isin(cubeset)))
        print((np.isnan(dat).sum(dim=('lat','lon','time')) / np.isnan(dat).count(dim=('lat','lon','time'))).values)

# save
for d, dat in enumerate(datasets):
    mask = np.isnan(dat)
    dat.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/verification/set{d}/data_crossval.nc')
    mask.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/verification/set{d}/mask_crossval.nc')

# delete values for verification
#for varname in varnames:
#    print(f'{datetime.now()} verification {varname}...')
#    tmp = delete_minicubes(mask.sel(time=vf_year, variable=varname).drop('variable').load(),
#                           frac_mis, ncubes)
#    mask.loc[dict(variable=varname, time=vf_year)] = tmp
#
#data = data.where(np.logical_not(mask))
#
## delete values for cross-validation
#for varname in varnames:
#    print(f'{datetime.now()} cross-validation {varname}...')
#    tmp = delete_minicubes(mask.sel(time=cv_year, variable=varname).drop('variable').load(),
#                           frac_mis, ncubes)
#    mask.loc[dict(variable=varname, time=cv_year)] = tmp
#
#data = data.where(np.logical_not(mask))

# save crossval cube for gap-filling
#print(f'{datetime.now()} save...')
#data.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/data_crossval.nc')
#mask.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/mask_crossval.nc')
