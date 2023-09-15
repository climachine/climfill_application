"""
NAMESTRING
"""

import argparse
import numpy as np
import xarray as xr
import regionmask

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
location = 'Europe'

orig = xr.open_dataset(f'{esapath}data_orig.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')

# select location 
if location == 'California':
    maxlat = 45
    minlat = 30
    maxlon = -105
    minlon = -125
    time = '2020-08-01'
elif location == 'Australia':
    maxlat = -11
    minlat = -45
    maxlon = 156
    minlon = 111
    time = '2019-12-01'
elif location == 'Europe':
    maxlat = 55
    minlat = 35
    maxlon = 27
    minlon = -10
    time = '2017-08-01'
else:
    raise AttributeError('location not defined')

orig = orig.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon))
fill = fill.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon)) 

# calc corr(sm,t) for all aug in eu
pcorr_orig = xr.corr(orig.sel(time=orig['time.month'] == 8).surface_temperature, 
                     orig.sel(time=orig['time.month'] == 8).soil_moisture, dim='time')
pcorr_fill = xr.corr(fill.sel(time=orig['time.month'] == 8).surface_temperature, 
                     fill.sel(time=orig['time.month'] == 8).soil_moisture, dim='time')

# burned area absolute values not ranks
ba_orig = orig.burned_area
ba_fill = fill.burned_area

# calculate quantiles
orig = orig.rank(dim='time', pct=True)
fill = fill.rank(dim='time', pct=True)

# sel time period
orig = orig.sel(time=time)
fill = fill.sel(time=time)
ba_orig = ba_orig.sel(time=time)
ba_fill = ba_fill.sel(time=time)

# get ocean different color
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon,orig.lat)
oceanmask = ~np.isnan(oceanmask)
orig = orig.where(oceanmask, -10)
fill = fill.where(oceanmask, -10)
ba_orig = ba_orig.where(oceanmask, -10)
ba_fill = ba_fill.where(oceanmask, -10)

# get specific color for rank 1, i.e. unprecendented event
orig = orig.where(orig != 1, 10)
fill = fill.where(fill != 1, 10)

# plot pre 
varnames_plot = ['burned area','2m temperature \n(in-situ)','land surface \ntemperature',
                 'diurnal temperature \nrange','precipitation \n(in-situ)',
                 'precipitation \n(satellite)','surface layer soil \nmoisture',
                 'terrestrial water \nstorage'] 
levels = np.arange(0,1.1,0.1)

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
proj = ccrs.Orthographic(central_longitude=(maxlon+minlon)/2,
                           central_latitude=(maxlat+minlat)/2)
#transf = ccrs.NearsidePerspective(central_longitude=(maxlon+minlon)/2,
#                                  central_latitude=(maxlat+minlat)/2)
cmap = plt.get_cmap('coolwarm')   
cmap_r = plt.get_cmap('coolwarm_r')
cmap_hot = plt.get_cmap('hot_r')
cmap.set_under('whitesmoke')
cmap_r.set_under('whitesmoke')
cmap_hot.set_under('whitesmoke')
cmap.set_bad('lightgrey')
cmap_r.set_bad('lightgrey')
cmap_hot.set_bad('lightgrey')
#cmap.set_over('maroon')
#cmap_r.set_over('maroon')

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(5,4,1, projection=proj)
ax2 = fig.add_subplot(5,4,2, projection=proj)
ax3 = fig.add_subplot(5,4,3, projection=proj)
ax4 = fig.add_subplot(5,4,4, projection=proj)
ax5 = fig.add_subplot(5,4,5, projection=proj)
ax6 = fig.add_subplot(5,4,6, projection=proj)
ax7 = fig.add_subplot(5,4,7, projection=proj)
ax8 = fig.add_subplot(5,4,8, projection=proj)
ax9 = fig.add_subplot(5,4,9, projection=proj)
ax10 = fig.add_subplot(5,4,10, projection=proj)
ax11 = fig.add_subplot(5,4,11, projection=proj)
ax12 = fig.add_subplot(5,4,12, projection=proj)
ax13 = fig.add_subplot(5,4,13, projection=proj)
ax14 = fig.add_subplot(5,4,14, projection=proj)
ax15 = fig.add_subplot(5,4,15, projection=proj)
ax16 = fig.add_subplot(5,4,16, projection=proj)
ax17 = fig.add_subplot(5,4,17, projection=proj)
ax18 = fig.add_subplot(5,4,18, projection=proj)
ax19 = fig.add_subplot(5,4,19, projection=proj)
ax20 = fig.add_subplot(5,4,20, projection=proj)

cbar_kwargs = {'label': ''}
plt_kwargs = {'vmin': 0, 'vmax': 1, 'transform': transf, 'levels': levels,
              'add_colorbar': False}

im0 = pcorr_orig.plot(ax=ax2, vmin=-1, vmax=0, transform=transf, add_colorbar=False)
pcorr_fill.plot(ax=ax3, vmin=-1, vmax=0, transform=transf, add_colorbar=False)

im1 = ba_orig.plot(ax=ax5, cmap=cmap_hot, vmin=0, vmax=10000000, 
                   transform=transf, add_colorbar=False)
ba_fill.plot(ax=ax9, cmap=cmap_hot, vmin=0, vmax=10000000, transform=transf, 
             add_colorbar=False)

im2 = orig.temperature_obs.plot(ax=ax6, cmap=cmap, **plt_kwargs)
fill.temperature_obs.plot(ax=ax10, cmap=cmap, **plt_kwargs)

orig.surface_temperature.plot(ax=ax7, cmap=cmap, **plt_kwargs)
fill.surface_temperature.plot(ax=ax11, cmap=cmap, **plt_kwargs)

orig.diurnal_temperature_range.plot(ax=ax8, cmap=cmap, **plt_kwargs)
fill.diurnal_temperature_range.plot(ax=ax12, cmap=cmap, **plt_kwargs)

orig.precipitation_obs.plot(ax=ax13, cmap=cmap_r, **plt_kwargs)
fill.precipitation_obs.plot(ax=ax17, cmap=cmap_r, **plt_kwargs)

orig.precipitation.plot(ax=ax14, cmap=cmap_r, **plt_kwargs)
fill.precipitation.plot(ax=ax18, cmap=cmap_r, **plt_kwargs)

orig.soil_moisture.plot(ax=ax15, cmap=cmap_r, **plt_kwargs)
fill.soil_moisture.plot(ax=ax19, cmap=cmap_r, **plt_kwargs)

orig.terrestrial_water_storage.plot(ax=ax16, cmap=cmap_r, **plt_kwargs)
fill.terrestrial_water_storage.plot(ax=ax20, cmap=cmap_r, **plt_kwargs)

#states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
for ax in (ax2,ax3,ax5,ax6,ax7,ax8,ax9,ax10,
           ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19,ax20):
    ax.set_title('')
    ax.coastlines()
    #states.plot(ax=ax, add_label=False) #DEBUG plots all US area
ax1.set_visible(False)
ax4.set_visible(False)
ax2.set_title('a) Corr(sm,t) With Gaps')
ax3.set_title('b) Corr(sm,t) Gap-Filled')

#ax1.set_title('With Gaps')
#ax2.set_title('Gap-Filled')
#ax1.text(0.5,1.4,f'{location}, {time}',transform=ax1.transAxes, va='center', fontsize=14)
ax5.text(-0.5, 0.5,'With Gaps',transform=ax5.transAxes, va='center')
ax9.text(-0.5, 0.5,'Gap-Filled',transform=ax9.transAxes, va='center')
ax13.text(-0.5, 0.5,'With Gaps',transform=ax13.transAxes, va='center')
ax17.text(-0.5, 0.5,'Gap-Filled',transform=ax17.transAxes, va='center')

letters = ['c','d','e','f','g','h','i','j','k']
for i, (varname, ax) in enumerate(zip(varnames_plot, (ax5,ax6,ax7,ax8,ax13,ax14,ax15,ax16))):
    #ax.text(-1.1, 0.5,varname,transform=ax.transAxes, va='center')
    l = letters[i]
    ax.set_title(f'{l}) {varname}')

cbar_ax = fig.add_axes([0.715, 0.76, 0.015, 0.12]) # left bottom width height
cbar = fig.colorbar(im0, cax=cbar_ax)
cbar.set_label('correlation')

cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.58]) # left bottom width height
cbar = fig.colorbar(im2, cax=cbar_ax)
cbar.set_label('percentile hottest/driest')

cbar_ax = fig.add_axes([0.125, 0.595, 0.17, 0.010]) # left bottom width height
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
cbar.set_label('$km^2$',labelpad=-5)

fig.suptitle(f'{time} {location}: Fire, Heat and Water Percentiles')
#plt.subplots_adjust(left=0.3, right=0.8)
plt.savefig(f'extrememaps_{location}.jpeg', dpi=300)
