"""
create ESA-CCI landmask.
"""

import numpy as np
import xarray as xr
import regionmask

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.75,90, 0.5)), # same as cdo_weights.nc
                     'lon': (['lon'], np.arange(-180, 180,0.5))})

# get basic landmask from regionmask
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(ds_out.lon,ds_out.lat)

#datacube mask ocean
landmask = landmask.where(landmask!=0,1) # boolean mask: land is True, ocean is False 
landmask = landmask.where(~np.isnan(landmask),0)
landmask = landmask.astype(bool)

# mask greenland (permanently glaciated)
n_greenland = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.map_keys('Greenland')
mask = regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(landmask)
landmask = landmask.where(mask != n_greenland, False)

# mask antarctica (permanently glaciated)
landmask = landmask.where(landmask.coords["lat"] > -60, False)

# mask deserts from precip threshold
precip_obs = xr.open_dataset(f'{esapath}precipitation_obs.nc')
isdesert = precip_obs.precipitation_obs.resample(time='YS').sum().mean(dim='time') < 100
landmask = landmask.where(~isdesert, False)

# save
landmask = landmask.to_dataset(name='landmask')
landmask.to_netcdf(f'{esapath}landmask.nc')
