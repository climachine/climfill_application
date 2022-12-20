"""
NAMESTRING
"""

import argparse
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

orig = xr.open_dataset(f'{esapath}data_orig.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')

# select California 
maxlat = 42
minlat = 32
maxlon = -115
minlon = -125

orig = orig.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon))
fill = fill.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon)) 

# calculate quantiles
orig = orig.rank(dim='time', pct=True)
fill = fill.rank(dim='time', pct=True)

# calculate anomalies
#orig = (orig.groupby('time.month') - orig.groupby('time.month').mean())# / orig.groupby('time.month').std()
#era5 = (era5.groupby('time.month') - era5.groupby('time.month').mean())# / era5.groupby('time.month').std()
#fill = (fill.groupby('time.month') - fill.groupby('time.month').mean())# / fill.groupby('time.month').std()

# calculate quantiles
#quantiles = fill.groupby('time.month').quantile(np.arange(0.1,1,0.1)).mean(dim=('lat','lon'))

# select 2020
timemin = '2020-03-01'
timemax = '2020-11-01'

orig = orig.sel(time=slice(timemin,timemax))
fill = fill.sel(time=slice(timemin,timemax))
#quantiles = quantiles.sel(month=slice(3,11))

# mean over all months
orig = orig.sel(time='2020-08-01')
fill = fill.sel(time='2020-08-01')

# plot pre 
varnames_plot = ['burned area [$km^2$]','2m temperature \nanomalies [$K$]','surface temperature \nanomalies [$K$]',
                 'diurnal temperature \nrange sfc anomalies [$K$]','precipitation (ground) \n[$mm\;month^{-1}$]',
                 'precipitation (sat) \n[$mm\;month^{-1}$]','surface layer soil \nmoisture [$m^3\;m^{-3}$]',
                 'terrestrial water \nstorage [$cm$]'] 
levels = np.arange(0,1.1,0.1)

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,17))
ax1 = fig.add_subplot(8,2,1, projection=proj)
ax2 = fig.add_subplot(8,2,2, projection=proj)
ax3 = fig.add_subplot(8,2,3, projection=proj)
ax4 = fig.add_subplot(8,2,4, projection=proj)
ax5 = fig.add_subplot(8,2,5, projection=proj)
ax6 = fig.add_subplot(8,2,6, projection=proj)
ax7 = fig.add_subplot(8,2,7, projection=proj)
ax8 = fig.add_subplot(8,2,8, projection=proj)
ax9 = fig.add_subplot(8,2,9, projection=proj)
ax10 = fig.add_subplot(8,2,10, projection=proj)
ax11 = fig.add_subplot(8,2,11, projection=proj)
ax12 = fig.add_subplot(8,2,12, projection=proj)
ax13 = fig.add_subplot(8,2,13, projection=proj)
ax14 = fig.add_subplot(8,2,14, projection=proj)
ax15 = fig.add_subplot(8,2,15, projection=proj)
ax16 = fig.add_subplot(8,2,16, projection=proj)

cbar_kwargs = {'label': ''}
plt_kwargs = {'vmin': 0, 'vmax': 1, 'transform': transf, 'levels': levels}

orig.burned_area.plot(ax=ax1, cmap='coolwarm', **plt_kwargs, add_colorbar=False)
fill.burned_area.plot(ax=ax2, cmap='coolwarm', **plt_kwargs, cbar_kwargs=cbar_kwargs)

orig.temperature_obs.plot(ax=ax3, cmap='coolwarm', **plt_kwargs, add_colorbar=False)
fill.temperature_obs.plot(ax=ax4, cmap='coolwarm', **plt_kwargs, cbar_kwargs=cbar_kwargs)

orig.surface_temperature.plot(ax=ax5, cmap='coolwarm', **plt_kwargs, add_colorbar=False)
fill.surface_temperature.plot(ax=ax6, cmap='coolwarm', **plt_kwargs, cbar_kwargs=cbar_kwargs)

orig.diurnal_temperature_range.plot(ax=ax7, cmap='coolwarm', **plt_kwargs, add_colorbar=False)
fill.diurnal_temperature_range.plot(ax=ax8, cmap='coolwarm', **plt_kwargs, cbar_kwargs=cbar_kwargs)

orig.precipitation_obs.plot(ax=ax9, cmap='coolwarm_r', **plt_kwargs, add_colorbar=False)
fill.precipitation_obs.plot(ax=ax10, cmap='coolwarm_r', **plt_kwargs, cbar_kwargs=cbar_kwargs)

orig.precipitation.plot(ax=ax11, cmap='coolwarm_r', **plt_kwargs, add_colorbar=False)
fill.precipitation.plot(ax=ax12, cmap='coolwarm_r', **plt_kwargs, cbar_kwargs=cbar_kwargs)

orig.soil_moisture.plot(ax=ax13, cmap='coolwarm_r', **plt_kwargs, add_colorbar=False)
fill.soil_moisture.plot(ax=ax14, cmap='coolwarm_r', **plt_kwargs, cbar_kwargs=cbar_kwargs)

orig.terrestrial_water_storage.plot(ax=ax15, cmap='coolwarm_r', **plt_kwargs, add_colorbar=False)
fill.terrestrial_water_storage.plot(ax=ax16, cmap='coolwarm_r', **plt_kwargs, cbar_kwargs=cbar_kwargs)

for ax in (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10,ax11,ax12,ax13,ax14,ax15,ax16):
    ax.coastlines()
    ax.set_title('')

for varname, ax in zip(varnames_plot, (ax1,ax3,ax5,ax7,ax9,ax11,ax13,ax15)):
    ax.text(-0.9, 0.5,varname,transform=ax.transAxes, va='center')

plt.savefig('extrememaps.png')
