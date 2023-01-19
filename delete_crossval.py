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

# same cubes every time
np.random.seed(0)

# crossval settings
frac_mis = 0.1 
ncubes = 24 # needs to be >= nt
vf_year = '2004'
cv_year = '2005'

# read data cube
print(f'{datetime.now()} read data...')
data_all = xr.open_dataset(f'{esapath}data_orig.nc').to_array()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# mask needs to be computed separately
mask_all = np.isnan(data_all)
mask_all = mask_all.where(landmask, np.nan) # is needed for calc of n_obs

# get varnames
#varnames =  list(data.keys()) # unsorted, but works on dataset
varnames = list(data_all.coords['variable'].values)
varnames.remove('landcover')

# select year
mask = mask_all.sel(time=slice(vf_year,cv_year)).load().copy(deep=True)
data = data_all.sel(time=slice(vf_year,cv_year)).load().copy(deep=True)

# copy 10 times
datasets = []
for i in range(10):
    datasets.append(data.copy(deep=True))

# create minicubes (on land)
minicubes = create_minicubes(mask.sel(variable='soil_moisture').drop('variable'),
                             ncubes)

# divide cubes on land into ten roughly equally sized sets
print(f'{datetime.now()} create and apply cubes...')
cubes_on_land = np.unique(minicubes)
cubes_on_land = cubes_on_land[~np.isnan(cubes_on_land)]

cubes_save = []
for varname in varnames:

    np.random.shuffle(cubes_on_land)
    random_sets = np.array_split(cubes_on_land, 10)

    for d, (dat, cubeset) in enumerate(zip(datasets, random_sets)):

        cubemask = np.logical_not(minicubes.isin(cubeset))
        dat.loc[varname,:,:,:] = dat.loc[varname,:,:,:].where(cubemask)
        #print((np.isnan(dat).sum(dim=('lat','lon','time')) / np.isnan(dat).count(dim=('lat','lon','time'))).values)
        cubemask = cubemask.expand_dims(veriset=[d])
        cubemask = cubemask.rename(varname)
        cubes_save.append(cubemask)

print(f'{datetime.now()} merge...')
cubes_save = xr.merge(cubes_save)
cubes_save.to_netcdf(f'{esapath}{testcase}/verification/mask_cubes.nc')

# save
print(f'{datetime.now()} save...')
for d, dat in enumerate(datasets):
    mask = np.isnan(dat)

    # add other years again
    mask_all.loc[dict(time=slice(vf_year,cv_year))] = mask
    data_all.loc[dict(time=slice(vf_year,cv_year))] = dat

    data_all.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/verification/set{d}/data_crossval.nc')
    mask_all.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/verification/set{d}/mask_crossval.nc')

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
