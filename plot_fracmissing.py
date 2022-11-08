"""
plot maps of fraction missing on daily resolution
"""

import numpy as np
import xarray as xr
import regionmask

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
varnames = ['soil_moisture','surface_temperature','precipitation', #for order of plots
            'terrestrial_water_storage','snow_water_equivalent',
            'temperature_obs','precipitation_obs','burned_area']

# TODO
# weird months with high obs in lst?
# include dtr

# open data
data = xr.open_dataset(f'{esapath}/data_orig.nc')
mask = xr.open_dataset(f'{esapath}/mask_orig.nc')
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# get landmask for plotting (land: w/o deserts and glacier)
latmask = landmask.sum(dim='lon')

# get oceanmask from regionmask
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(data.lon,data.lat)
oceanmask = ~np.isnan(oceanmask)

# calculate fraction of missing values per grid point
maskmap = mask.sum(dim='time') / len(data.time)

# calculate fraction of missing values per month x latitude
masktimeline = mask.sum(dim='lon') / latmask
masktimeline = masktimeline.where(~np.isinf(masktimeline), np.nan)

# calculate fraction of missing values overall
n_relevant = landmask.sum() * len(data.time)
frac_mis = mask.sum() / n_relevant

# remove uncovered high latitudes
masktimeline = masktimeline.sel(lat=slice(-62,82))

# set non-relevant land to nan
maskmap = maskmap.where(landmask, np.nan)

# set ocean to negative
maskmap = maskmap.where(oceanmask, -10)

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('YlOrBr_r')
cmap.set_under('aliceblue')

fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(4,4,1, projection=proj)
ax2 = fig.add_subplot(4,4,2, projection=proj)
ax3 = fig.add_subplot(4,4,3, projection=proj)
ax4 = fig.add_subplot(4,4,4, projection=proj)
ax5 = fig.add_subplot(4,4,5)
ax6 = fig.add_subplot(4,4,6)
ax7 = fig.add_subplot(4,4,7)
ax8 = fig.add_subplot(4,4,8)
ax9 = fig.add_subplot(4,4,9, projection=proj)
ax10 = fig.add_subplot(4,4,10, projection=proj)
ax11 = fig.add_subplot(4,4,11, projection=proj)
ax12 = fig.add_subplot(4,4,12, projection=proj)
ax13 = fig.add_subplot(4,4,13)
ax14 = fig.add_subplot(4,4,14)
ax15 = fig.add_subplot(4,4,15)
ax16 = fig.add_subplot(4,4,16)

im = maskmap.soil_moisture.plot(ax=ax1, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.surface_temperature.plot(ax=ax2, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.precipitation.plot(ax=ax3, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.terrestrial_water_storage.plot(ax=ax4, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.snow_water_equivalent.plot(ax=ax9, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.temperature_obs.plot(ax=ax10, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.precipitation_obs.plot(ax=ax11, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.burned_area.plot(ax=ax12, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)

masktimeline.soil_moisture.T.plot(ax=ax5, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.surface_temperature.T.plot(ax=ax6, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.precipitation.T.plot(ax=ax7, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.terrestrial_water_storage.T.plot(ax=ax8, cmap=cmap, vmin=0, 
                           vmax=1, add_colorbar=False)
masktimeline.snow_water_equivalent.T.plot(ax=ax13, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.temperature_obs.T.plot(ax=ax14, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.precipitation_obs.T.plot(ax=ax15, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.burned_area.T.plot(ax=ax16, cmap=cmap, vmin=0, 
                           vmax=1, add_colorbar=False)

for varname, ax in zip(varnames, (ax1,ax2,ax3,ax4,ax9,ax10,ax11,ax12)):
    frac = int(np.round(frac_mis[varname].item(), 2)*100)
    ax.set_title(f'{varname}: {frac}% missing')

cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('fraction of missing values')

ax5.set_xlabel('')
ax6.set_xlabel('')
ax7.set_xlabel('')
ax8.set_xlabel('')

ax6.set_ylabel('')
ax7.set_ylabel('')
ax8.set_ylabel('')
ax14.set_ylabel('')
ax15.set_ylabel('')
ax16.set_ylabel('')

ax5.set_xticklabels([])
ax6.set_xticklabels([])
ax7.set_xticklabels([])
ax8.set_xticklabels([])

ax1.set_facecolor('lightgrey')
ax2.set_facecolor('lightgrey')
ax3.set_facecolor('lightgrey')
ax4.set_facecolor('lightgrey')
ax5.set_facecolor('lightgrey')
ax6.set_facecolor('lightgrey')
ax7.set_facecolor('lightgrey')
ax8.set_facecolor('lightgrey')
ax9.set_facecolor('lightgrey')
ax10.set_facecolor('lightgrey')
ax11.set_facecolor('lightgrey')
ax12.set_facecolor('lightgrey')
ax13.set_facecolor('lightgrey')
ax14.set_facecolor('lightgrey')
ax15.set_facecolor('lightgrey')
ax16.set_facecolor('lightgrey')
plt.show()
