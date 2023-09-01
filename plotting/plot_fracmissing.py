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
            'terrestrial_water_storage', 'temperature_obs','precipitation_obs',
            'snow_cover_fraction', 'diurnal_temperature_range', 'burned_area']
#varnames = ['soil_moisture','surface_temperature','temperature_obs',
#            'diurnal_temperature_range','burned_area',
#            'precipitation','precipitation_obs','snow_cover_fraction',
#            'terrestrial_water_storage', 'land_cover']

# TODO
# include burned area

# control text sizes plot
SMALL_SIZE = 15
MEDIUM_SIZE = SMALL_SIZE+2
BIGGER_SIZE = SMALL_SIZE+4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

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
cmap = plt.get_cmap('YlOrBr')
cmap.set_under('aliceblue')

fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(6,3,1, projection=proj)
ax2 = fig.add_subplot(6,3,2, projection=proj)
ax3 = fig.add_subplot(6,3,3, projection=proj)
ax4 = fig.add_subplot(6,3,4)
ax5 = fig.add_subplot(6,3,5)
ax6 = fig.add_subplot(6,3,6)
ax7 = fig.add_subplot(6,3,7, projection=proj)
ax8 = fig.add_subplot(6,3,8, projection=proj)
ax9 = fig.add_subplot(6,3,9, projection=proj)
ax10 = fig.add_subplot(6,3,10)
ax11 = fig.add_subplot(6,3,11)
ax12 = fig.add_subplot(6,3,12)
ax13 = fig.add_subplot(6,3,13, projection=proj)
ax14 = fig.add_subplot(6,3,14, projection=proj)
ax15 = fig.add_subplot(6,3,15, projection=proj)
ax16 = fig.add_subplot(6,3,16)
ax17 = fig.add_subplot(6,3,17)
ax18 = fig.add_subplot(6,3,18)

im = maskmap.soil_moisture.plot(ax=ax1, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.surface_temperature.plot(ax=ax2, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.precipitation.plot(ax=ax3, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)

maskmap.terrestrial_water_storage.plot(ax=ax7, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.temperature_obs.plot(ax=ax8, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.precipitation_obs.plot(ax=ax9, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)

maskmap.snow_cover_fraction.plot(ax=ax13, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.diurnal_temperature_range.plot(ax=ax14, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)
maskmap.burned_area.plot(ax=ax15, cmap=cmap, vmin=0, vmax=1, 
                           transform=transf, add_colorbar=False)

masktimeline.soil_moisture.T.plot(ax=ax4, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.surface_temperature.T.plot(ax=ax5, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.precipitation.T.plot(ax=ax6, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)

masktimeline.terrestrial_water_storage.T.plot(ax=ax10, cmap=cmap, vmin=0, 
                           vmax=1, add_colorbar=False)
masktimeline.temperature_obs.T.plot(ax=ax11, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.precipitation_obs.T.plot(ax=ax12, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)

masktimeline.snow_cover_fraction.T.plot(ax=ax16, cmap=cmap, vmin=0, vmax=1, 
                           add_colorbar=False)
masktimeline.diurnal_temperature_range.T.plot(ax=ax17, cmap=cmap, vmin=0, 
                           vmax=1, add_colorbar=False)
masktimeline.burned_area.T.plot(ax=ax18, cmap=cmap, vmin=0, 
                           vmax=1, add_colorbar=False)

varnames_plot = ['SM','LST','PSAT', #for order of plots
            'TWS', 'T2M','P2M',
            'SCF', 'DTR', 'BA']
#varnames_plot = ['surface layer \n soil moisture','land surface temperature',
#                 'precipitation \n(satellite)', 
letters = ['a','b','c','d','e','f','g','h','i']
for v, (letter, varname, ax) in enumerate(zip(letters,varnames, (ax1,ax2,ax3,ax7,ax8,ax9,ax13,ax14,ax15))):
    #frac = np.around(frac_mis[varname].item()*100, decimals=0)
    frac = round(frac_mis[varname].item()*100)
    ax.set_title(f'{letter}) {varnames_plot[v]}: {frac}% missing')

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('fraction of missing values')

ax4.set_xlabel('')
ax5.set_xlabel('')
ax6.set_xlabel('')
ax10.set_xlabel('')
ax11.set_xlabel('')
ax12.set_xlabel('')

ax4.set_xticklabels([])
ax5.set_xticklabels([])
ax6.set_xticklabels([])
ax10.set_xticklabels([])
ax11.set_xticklabels([])
ax12.set_xticklabels([])

ax5.set_ylabel('')
ax6.set_ylabel('')
ax11.set_ylabel('')
ax12.set_ylabel('')
ax17.set_ylabel('')
ax18.set_ylabel('')

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
ax17.set_facecolor('lightgrey')
ax18.set_facecolor('lightgrey')
#plt.show()
plt.savefig('frac_missing.jpeg', dpi=300, bbox_inches='tight')
