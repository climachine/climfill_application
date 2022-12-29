"""
NAMESTRING
"""

import numpy as np
import xarray as xr
import regionmask
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
orig = xr.open_dataset(f'{esapath}/data_orig.nc')
intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# create anomalies
orig = orig.groupby('time.month') - fill.groupby('time.month').mean()
intp = intp.groupby('time.month') - fill.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# select month and place
month = '2020-10'
orig = orig.sel(time=month, lat=slice(0,80), lon=slice(50,100))
intp = intp.sel(time=month, lat=slice(0,80), lon=slice(50,100))
fill = fill.sel(time=month, lat=slice(0,80), lon=slice(50,100))

# intp remove ocean
orig = orig.where(landmask)
intp = intp.where(landmask)
fill = fill.where(landmask)

# make ocean grey
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon,orig.lat)
oceanmask = ~np.isnan(oceanmask)
orig = orig.where(oceanmask, -300)
intp = intp.where(oceanmask, -300)
fill = fill.where(oceanmask, -300)

# plot
proj = ccrs.Robinson()
proj = ccrs.PlateCarree()
transf = ccrs.PlateCarree()

cmap = plt.get_cmap('coolwarm')   
cmap.set_under('aliceblue')
cmap.set_bad('lightgrey')
cmap_r = plt.get_cmap('coolwarm_r')
cmap_r.set_under('aliceblue')
cmap_r.set_bad('lightgrey')

fig = plt.figure(figsize=(10,10))
fig.suptitle('Oct 2020, anomalies')
ax1 = fig.add_subplot(331, projection=proj)
ax2 = fig.add_subplot(332, projection=proj)
ax3 = fig.add_subplot(333, projection=proj)
ax4 = fig.add_subplot(334, projection=proj)
ax5 = fig.add_subplot(335, projection=proj)
ax6 = fig.add_subplot(336, projection=proj)
ax7 = fig.add_subplot(337, projection=proj)
ax8 = fig.add_subplot(338, projection=proj)
ax9 = fig.add_subplot(339, projection=proj)


plt_kwargs = {'transform': transf,  'cmap': cmap,
              'add_colorbar': False}
plt_kwargs_r = {'transform': transf,  'cmap': cmap_r,
              'add_colorbar': False}

levels = np.arange(-10,10,2)
im = orig.surface_temperature.plot(ax=ax1, **plt_kwargs, levels=levels)
intp.surface_temperature.plot(ax=ax2, **plt_kwargs, levels=levels)
fill.surface_temperature.plot(ax=ax3, **plt_kwargs, levels=levels)

cbar_ax = fig.add_axes([0.90, 0.70, 0.02, 0.2]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$^\circ$C')

levels = np.arange(-0.05,0.05,0.01)
im = orig.soil_moisture.plot(ax=ax4, **plt_kwargs_r, levels=levels)
intp.soil_moisture.plot(ax=ax5, **plt_kwargs_r, levels=levels)
fill.soil_moisture.plot(ax=ax6, **plt_kwargs_r, levels=levels)

cbar_ax = fig.add_axes([0.90, 0.40, 0.02, 0.2]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$m^3\;m^{-3}$')

levels = np.arange(-100,100,20)
im = orig.precipitation.plot(ax=ax7, **plt_kwargs_r, levels=levels)
intp.precipitation.plot(ax=ax8, **plt_kwargs_r, levels=levels)
fill.precipitation.plot(ax=ax9, **plt_kwargs_r, levels=levels)

cbar_ax = fig.add_axes([0.90, 0.10, 0.02, 0.2]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('$mm\;month^{-1}$')

ax1.set_title('With Gaps')
ax2.set_title('Interpolation Gap-Fill')
ax3.set_title('CLIMFILL Gap-Fill')

ax1.text(-1.1, 0.5,'surface \ntemperature',transform=ax1.transAxes, va='center')
ax4.text(-1.1, 0.5,'soil moisture',transform=ax4.transAxes, va='center')
ax7.text(-1.1, 0.5,'precipitation',transform=ax7.transAxes, va='center')

for ax in (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9):
    ax.coastlines()

for ax in (ax4,ax5,ax6,ax7,ax8,ax9):
    ax.set_title('')

plt.savefig('twosteps.png')
