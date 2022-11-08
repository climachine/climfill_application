"""
NAMESTRING
"""

import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
orig = xr.open_dataset(f'{esapath}{testcase}/data_orig.nc')
intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# select variable
varname = 'precipitation'
orig = orig[varname]
intp = intp[varname]
fill = fill[varname]

# intp remove ocean
intp = intp.where(landmask)

# plot
month = '1996-10'

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(131, projection=proj)
ax2 = fig.add_subplot(132, projection=proj)
ax3 = fig.add_subplot(133, projection=proj)

orig.sel(time=month, lat=slice(0,80), lon=slice(50,100)).plot(ax=ax1)
intp.sel(time=month, lat=slice(0,80), lon=slice(50,100)).plot(ax=ax2)
fill.sel(time=month, lat=slice(0,80), lon=slice(50,100)).plot(ax=ax3)

ax1.set_title('gappy satellite data')
ax2.set_title('init interp gapfill')
ax3.set_title('climfill gapfill')

plt.show()
