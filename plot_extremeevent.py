"""
NAMESTRING
"""

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

orig = xr.open_dataset(f'{esapath}data_orig.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
era5 = xr.open_dataset(f'{esapath}data_era5land.nc')

# select California 
maxlat = 42
minlat = 32
maxlon = -115
minlon = -125

orig = orig.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon))
fill = fill.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon)) 
era5 = era5.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon)) 

# test
orig['surface_temperature'] = orig.surface_temperature.groupby('time.month') - orig.surface_temperature.groupby('time.month').mean()
fill['surface_temperature'] = fill.surface_temperature.groupby('time.month') - fill.surface_temperature.groupby('time.month').mean()
era5['surface_temperature'] = era5.surface_temperature.groupby('time.month') - era5.surface_temperature.groupby('time.month').mean()

orig['temperature_obs'] = orig.temperature_obs.groupby('time.month') - orig.temperature_obs.groupby('time.month').mean()
fill['temperature_obs'] = fill.temperature_obs.groupby('time.month') - fill.temperature_obs.groupby('time.month').mean()
era5['temperature_obs'] = era5.temperature_obs.groupby('time.month') - era5.temperature_obs.groupby('time.month').mean()

orig['diurnal_temperature_range'] = orig.diurnal_temperature_range.groupby('time.month') - orig.diurnal_temperature_range.groupby('time.month').mean()
fill['diurnal_temperature_range'] = fill.diurnal_temperature_range.groupby('time.month') - fill.diurnal_temperature_range.groupby('time.month').mean()
era5['diurnal_temperature_range'] = era5.diurnal_temperature_range.groupby('time.month') - era5.diurnal_temperature_range.groupby('time.month').mean()

# calculate quantiles
quantiles = fill.groupby('time.month').quantile(np.arange(0.1,1,0.1)).mean(dim=('lat','lon'))

# calculate anomalies
#orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
#era5 = era5.groupby('time.month') - era5.groupby('time.month').mean()
#fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# select 2020
timemin = '2020-03-01'
timemax = '2020-11-01'

orig = orig.sel(time=slice(timemin,timemax))
fill = fill.sel(time=slice(timemin,timemax))
era5 = era5.sel(time=slice(timemin,timemax))
quantiles = quantiles.sel(month=slice(3,11))

# compute regional averages
orig = orig.mean(dim=('lat','lon'))
fill = fill.mean(dim=('lat','lon'))
era5 = era5.mean(dim=('lat','lon'))

# get greyscale colorscale
cmap = plt.get_cmap('Greys')
cols = cmap(np.linspace(0.1,0.8,9))

def plot_quantiles(quantiles,xaxis,cols,ax,zero=0):
    for qu, col in zip(quantiles.coords['quantile'][:-9:-1], cols[:-9:-1]):
        ax.fill_between(xaxis, quantiles.sel(quantile=qu).values,
                        quantiles.sel(quantile=qu-0.1, method='nearest').values, color=col)

# nice plotting varnames
varnames_plot = ['burned area [$km^2$]','2m temperature anomalies [$K$]','surface temperature anomalies [$K$]',
                 'diurnal temperature range sfc anomalies [$K$]','precipitation (ground) [$mm\;month^{-1}$]',
                 'precipitation (sat) [$mm\;month^{-1}$]','surface layer soil moisture [$m^3\;m^{-3}$]',
                 'terrestrial water storage [$cm$]'] 

# plot timelines
fig = plt.figure(figsize=(10,17))
ax1 = fig.add_subplot(811)
ax2 = fig.add_subplot(812)
ax3 = fig.add_subplot(813)
ax4 = fig.add_subplot(814)
ax5 = fig.add_subplot(815)
ax6 = fig.add_subplot(816)
ax7 = fig.add_subplot(817)
ax8 = fig.add_subplot(818)

eracol = 'royalblue'
origcol = 'coral'
fillcol = 'brown'

orig.burned_area.plot(ax=ax1, color=origcol)
fill.burned_area.plot(ax=ax1, color=fillcol)
#era5.burned_area.plot(ax=ax1)
plot_quantiles(quantiles.burned_area,orig.coords['time'],cols,ax1)

orig.temperature_obs.plot(ax=ax2, color=origcol)
fill.temperature_obs.plot(ax=ax2, color=fillcol)
era5.temperature_obs.plot(ax=ax2, color=eracol)
plot_quantiles(quantiles.temperature_obs,orig.coords['time'],cols,ax2,zero=10)

orig.surface_temperature.plot(ax=ax3, color=origcol)
fill.surface_temperature.plot(ax=ax3, color=fillcol)
era5.surface_temperature.plot(ax=ax3, color=eracol)
plot_quantiles(quantiles.surface_temperature,orig.coords['time'],cols,ax3,zero=-2)

orig.diurnal_temperature_range.plot(ax=ax4, color=origcol)
fill.diurnal_temperature_range.plot(ax=ax4, color=fillcol)
era5.diurnal_temperature_range.plot(ax=ax4, color=eracol)
plot_quantiles(quantiles.diurnal_temperature_range,orig.coords['time'],cols,ax4,zero=10)

orig.precipitation_obs.plot(ax=ax5, color=origcol)
fill.precipitation_obs.plot(ax=ax5, color=fillcol)
era5.precipitation_obs.plot(ax=ax5, color=eracol)
plot_quantiles(quantiles.precipitation_obs,orig.coords['time'],cols,ax5)

orig.precipitation.plot(ax=ax6, color=origcol)
fill.precipitation.plot(ax=ax6, color=fillcol)
era5.precipitation.plot(ax=ax6, color=eracol)
plot_quantiles(quantiles.precipitation,orig.coords['time'],cols,ax6)

orig.soil_moisture.plot(ax=ax7, color=origcol)
fill.soil_moisture.plot(ax=ax7, color=fillcol)
era5.soil_moisture.plot(ax=ax7, color=eracol)
plot_quantiles(quantiles.soil_moisture,orig.coords['time'],cols,ax7,zero=0.1)

orig.terrestrial_water_storage.plot(ax=ax8, color=origcol)
fill.terrestrial_water_storage.plot(ax=ax8, color=fillcol)
era5.terrestrial_water_storage.plot(ax=ax8, color=eracol)
plot_quantiles(quantiles.terrestrial_water_storage,orig.coords['time'],cols,ax8,zero=-25)

legend_elements = [Line2D([0], [0], color=eracol, lw=4, label='ERA5-Land'),
                   Line2D([0], [0], color=origcol, lw=4, label='With Gaps'),
                   Line2D([0], [0], color=fillcol, lw=4, label='Gap-filled')]
ax1.legend(handles=legend_elements, loc='upper right')

for ax in (ax1,ax2,ax3,ax4,ax5,ax6,ax7):
    ax.set_xlabel('')
    ax.set_xticklabels([])

for varname, ax in zip(varnames_plot, (ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8)):
    ax.set_title(varname)
    ax.set_ylabel('')

plt.savefig('extremeevent.png')
