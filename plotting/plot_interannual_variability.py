"""
NAMESTRING
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import regionmask
import cartopy.crs as ccrs

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','temperature_obs','precipitation_obs',
            'snow_cover_fraction']

# read data
orig = xr.open_dataset(f'{esapath}data_orig.nc') 
intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
era5 = xr.open_dataset(f'{esapath}data_era5land.nc')

# calc interannual variability
# DEBUG
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
intp = intp.groupby('time.month') - intp.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()
era5 = era5.groupby('time.month') - era5.groupby('time.month').mean()

# plot correlation
#levels = np.arange(0,1.2,0.1)
#proj = ccrs.Robinson()
#transf = ccrs.PlateCarree()
#for varname in varnames:
#    fig = plt.figure(figsize=(25,5))
#    ax1 = fig.add_subplot(1,3,1, projection=proj)
#    ax2 = fig.add_subplot(1,3,2, projection=proj)
#    ax3 = fig.add_subplot(1,3,3, projection=proj)
#    xr.corr(orig[varname],era5[varname], dim='time').plot(ax=ax1, 
#            transform=transf, vmin=0, vmax=1, cmap='Greens', levels=levels)
#    xr.corr(intp[varname],era5[varname], dim='time').plot(ax=ax2, 
#            transform=transf, vmin=0, vmax=1, cmap='Greens', levels=levels)
#    xr.corr(fill[varname],era5[varname], dim='time').plot(ax=ax3, 
#            transform=transf, vmin=0, vmax=1, cmap='Greens', levels=levels)
#    plt.show()

# aggregate per ar6 region
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))

orig = orig.groupby(regions).mean()
intp = intp.groupby(regions).mean()
fill = fill.groupby(regions).mean()
era5 = era5.groupby(regions).mean()

# plot per region
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()

for varname in varnames:
    #for region in orig.mask.values:
    region=22# DEBUG

    orig_tmp = orig[varname].sel(mask=region)
    intp_tmp = intp[varname].sel(mask=region)
    fill_tmp = fill[varname].sel(mask=region)
    era5_tmp = era5[varname].sel(mask=region)

    fig = plt.figure(figsize=(20,5))
    ax = fig.add_subplot(111)

    corr_orig = np.round(xr.corr(orig_tmp,era5_tmp).item(),2)
    corr_intp = np.round(xr.corr(intp_tmp,era5_tmp).item(),2)
    corr_fill = np.round(xr.corr(fill_tmp,era5_tmp).item(),2)

    orig_tmp.plot(ax=ax, label=f'orig {corr_orig}')
    intp_tmp.plot(ax=ax, label=f'intp {corr_intp}')
    fill_tmp.plot(ax=ax, label=f'fill {corr_fill}')
    era5_tmp.plot(ax=ax, label='era5land')

    ax.legend()
    plt.show()

