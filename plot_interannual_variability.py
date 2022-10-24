"""
NAMESTRING
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import regionmask

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
orig = xr.open_dataset(f'{esapath}{testcase}/data_orig.nc')
intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
era5 = xr.open_dataset(f'{esapath}data_era5land.nc')

# calc interannual variability
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
intp = intp.groupby('time.month') - intp.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()
era5 = era5.groupby('time.month') - era5.groupby('time.month').mean()

# aggregate per ar6 region
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))

orig = orig.groupby(regions).mean()
intp = intp.groupby(regions).mean()
fill = fill.groupby(regions).mean()
era5 = era5.groupby(regions).mean()

# plot
region = 17
orig.surface_temperature.sel(mask=region).plot()
intp.surface_temperature.sel(mask=region).plot()
fill.surface_temperature.sel(mask=region).plot()
era5.skt.sel(mask=region).plot()
plt.show()
