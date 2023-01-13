"""
NAMESTRING
"""

import argparse
import numpy as np
import regionmask
import cartopy.crs as ccrs
from scipy.spatial.distance import jensenshannon as js
import xarray as xr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
time = slice('1995','2020')

# read data
orig = xr.open_dataset(f'{esapath}data_orig.nc').soil_moisture
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc').soil_moisture
ismn = xr.open_dataset('/net/so4/landclim/bverena/large_files/df_gaps.nc')

# select esa time period
ismn = ismn.sel(time=time)

# resample ismn to monthly
mask = np.isnan(ismn).astype(float).resample(time='MS').sum()
mask = mask <= 15
ismn = ismn.resample(time='MS').mean()
ismn = ismn.where(mask)

# take only ismn stations with more than 5 years of data
#mask = np.logical_not(np.isnan(ismn.mrso)).sum(dim='time')
#mask = mask <= 12*5
#ismn = ismn.where(~mask, drop=True)

# (optional) calculate anomalies
#orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
#fill = fill.groupby('time.month') - fill.groupby('time.month').mean()
#ismn = ismn.groupby('time.month') - ismn.groupby('time.month').mean()

# ismn to worldmap
# easier: orig and fill to ismn shape
orig_ismn = xr.full_like(ismn, np.nan)
fill_ismn = xr.full_like(ismn, np.nan)

for i, (lat,lon) in enumerate(zip(ismn.lat.values, ismn.lon.values)):
    orig_ismn.mrso[:,i] = orig.sel(lat=lat, lon=lon, method='nearest')
    fill_ismn.mrso[:,i] = fill.sel(lat=lat, lon=lon, method='nearest')
orig_ismn.to_netcdf(f'{esapath}data_orig_ismn.nc')
fill_ismn.to_netcdf(f'{esapath}data_climfilled_ismn.nc')

# normalise values for RMSE plotting
#orig_ismn = (orig_ismn - orig_ismn.mean()) / orig_ismn.std()
#fill_ismn = (fill_ismn - fill_ismn.mean()) / fill_ismn.std()
#ismn = (ismn - ismn.mean()) / ismn.std()

# get 3 stations with longest record
tmp = np.logical_not(np.isnan(ismn.mrso)).sum(dim='time')
stations = tmp.sortby(tmp, ascending=False)[:3].stations
# note: station outside US only at place 161 (Iberian Peninsula, Nr 1274)
stations = [1777,1811,1274]

#orig_stations = orig_ismn.sel(stations=stations).mrso
#fill_stations = fill_ismn.sel(stations=stations).mrso
#ismn_stations = ismn.sel(stations=stations).mrso

# plot station locations
#proj = ccrs.Robinson()
#transf = ccrs.PlateCarree()
#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111, projection=proj)
#ax.scatter(ismn.mrso.sel(stations=stations).lon, ismn.mrso.sel(stations=stations).lat, c=tmp.sel(stations=stations), transform=transf)
#ax.coastlines()
#plt.show()

# plot example station
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

col_fill = 'coral'
col_miss = 'steelblue'
col_ismn = 'olivedrab'

fill_ismn.mrso.sel(stations=stations[0]).plot(ax=ax1, color=col_fill)
orig_ismn.mrso.sel(stations=stations[0]).plot(ax=ax1, color=col_miss)
ismn.mrso.sel(stations=stations[0]).plot(ax=ax1, color=col_ismn)

fill_ismn.mrso.sel(stations=stations[1]).plot(ax=ax2, color=col_fill)
orig_ismn.mrso.sel(stations=stations[1]).plot(ax=ax2, color=col_miss)
ismn.mrso.sel(stations=stations[1]).plot(ax=ax2, color=col_ismn)

fill_ismn.mrso.sel(stations=stations[2]).plot(ax=ax3, color=col_fill)
orig_ismn.mrso.sel(stations=stations[2]).plot(ax=ax3, color=col_miss)
ismn.mrso.sel(stations=stations[2]).plot(ax=ax3, color=col_ismn)

ax1.set_title('Little River, SCAN Network, Georgia, USA')
ax2.set_title('N Piedmont Arec, SCAN Network, Virginia, USA')
ax3.set_title('Llanos de la Boveda, REMEDHUS Network, Spain')

ax1.set_xticklabels([])
ax2.set_xticklabels([])

#ax1.set_ylim([-0.15,0.15])
#ax2.set_ylim([-0.15,0.15])
#ax3.set_ylim([-0.15,0.15])

# TODO check units
ax1.set_ylabel('surface layer \nsoil moisture $m^{3}m^{-3}$')
ax2.set_ylabel('surface layer \nsoil moisture $m^{3}m^{-3}$')
ax3.set_ylabel('surface layer \nsoil moisture $m^{3}m^{-3}$')

ax1.set_xlabel('')
ax2.set_xlabel('')

plt.savefig('benchmark_ismn.pdf')