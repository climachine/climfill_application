"""
NAMESTRING
"""

import numpy as np
import xarray as xr

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
data = xr.open_mfdataset(f'{esapath}*.nc')

# extract landmask
landmask = data.landmask
data = data.drop('landmask') 

# xarray/dask issue https://github.com/pydata/xarray/issues/3813
# value assignment only works if non-dask array
data = data.to_array().load()

# create mask of missing values
mask = np.isnan(data)

# set ocean to nan
data = data.where(landmask, np.nan)
mask = mask.where(landmask, np.nan)

# crossval settings
frac_mis = 0.1
ncubes = 20

#def delete_minicubes(mask, frac_mis, ncubes):
#
#    varnames = mask.coords["variable"].values
#
#    for varname in varnames:
#        tmp = delete_minicubes_one_variable(mask.sel(variable=varname).drop('variable'),
#                                            frac_mis, ncubes)
#        mask.loc[dict(variable=varname)] = tmp
#
#    return mask

def delete_minicubes(mask, frac_mis, ncubes):
    #### args 
    # args: mask (3dims called time, lat lon, xarray); 
    #          with observed True, unobserved False, outside (ocean) NaN
    #       ncubes: int, number of cubes along each axis
    #       frac_mis: fraction of missing values, float

    # calculate number of observed and missing values (on land)
    n_mis = mask.sum().item()
    n_land = np.isnan(mask).sum().item()
    n_obs = n_land - n_mis

    # create minicubes of observed data for cross-validation
    nt = len(mask.time) 
    nx, ny = len(mask.lon), len(mask.lat)
    ncubes = ncubes # along each axis
    a = np.arange(ncubes**3).reshape(ncubes,ncubes,ncubes)
    b = a.repeat(np.ceil(nt/ncubes),0).repeat(np.ceil(ny/ncubes),1).repeat(np.ceil(nx/ncubes),2)
    b = b[:nt,:ny,:nx] # trim

    # wrap around xarray
    minicubes = xr.full_like(mask, np.nan) # to xarray for .isin fct
    minicubes[:] = b

    # check only those on land for faster convergence
    minicubes = minicubes.where(~np.isnan(mask)) # only consider cubes on land
    cubes_on_land = np.unique(minicubes)
    cubes_on_land = cubes_on_land[~np.isnan(cubes_on_land)]

    # delete randomly 10% of the minicubes
    mask_verification = mask.copy(deep=True)
    exitflag = False
    while True:
        selected_cube = np.random.choice(cubes_on_land)
        mask_verification = mask_verification.where(minicubes != selected_cube, True)
        n_cv = mask_verification.sum().load().item()
        frac_cv = n_cv / n_obs
        print(f'fraction crossval data from observed data: {frac_cv}')
        if frac_cv > 1+frac_mis: # cannot break outer loop
            break

    return mask_verification

verification_year = '2004'
crossvalidation_year = '2005'

varnames = mask.coords["variable"].values

for varname in varnames:
    tmp = delete_minicubes(mask.sel(time=verification_year, variable=varname).drop('variable'),
                           frac_mis, ncubes)
    mask.loc[dict(variable=varname, time=verification_year)] = tmp

data.to_netcdf # save orig
data = data.where(mask)
