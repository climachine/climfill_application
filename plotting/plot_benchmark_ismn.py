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
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
time = slice('1995','2020')

col_fill = 'coral'
col_miss = 'darkgrey'
col_ismn = 'olivedrab'
col_intp =  'steelblue'

# control text sizes plot
SMALL_SIZE = 20
MEDIUM_SIZE = SMALL_SIZE+2
BIGGER_SIZE = SMALL_SIZE+4

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# read data
orig = xr.open_dataset(f'{esapath}data_orig.nc').soil_moisture
#intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc').soil_moisture
#fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc').soil_moisture
ismn = xr.open_dataset('/net/so4/landclim/bverena/large_files/optimal/df_gaps.nc')

# select esa time period
ismn = ismn.sel(time=time)

# resample ismn to monthly
mask = np.isnan(ismn).astype(float).resample(time='MS').sum()
mask = mask <= 15
ismn = ismn.resample(time='MS').mean()
ismn = ismn.where(mask)

# take only ismn stations with more than 2 years of data
mask = np.logical_not(np.isnan(ismn.mrso)).sum(dim='time')
mask = mask >= 12*5
ismn = ismn.where(mask, drop=True)

## ismn to worldmap
## easier: orig and fill to ismn shape
#orig_ismn = xr.full_like(ismn, np.nan)
#intp_ismn = xr.full_like(ismn, np.nan)
#fill_ismn = xr.full_like(ismn, np.nan)
#
#for i, (lat,lon) in enumerate(zip(ismn.lat.values, ismn.lon.values)):
#    orig_ismn.mrso[:,i] = orig.sel(lat=lat, lon=lon, method='nearest')
#    intp_ismn.mrso[:,i] = intp.sel(lat=lat, lon=lon, method='nearest')
#    fill_ismn.mrso[:,i] = fill.sel(lat=lat, lon=lon, method='nearest')
#orig_ismn.to_netcdf(f'{esapath}data_orig_ismn.nc')
#intp_ismn.to_netcdf(f'{esapath}data_intp_ismn.nc')
#fill_ismn.to_netcdf(f'{esapath}data_climfilled_ismn.nc')
orig_ismn = xr.open_dataarray(f'{esapath}data_orig_ismn.nc')
intp_ismn = xr.open_dataarray(f'{esapath}data_intp_ismn.nc')
fill_ismn = xr.open_dataarray(f'{esapath}data_climfilled_ismn.nc')
ismn = ismn.mrso

# get 3 stations with longest record
tmp = np.logical_not(np.isnan(ismn)).sum(dim='time')
stations = tmp.sortby(tmp, ascending=False)[:3].stations
# note: station outside US only at place 161 (Iberian Peninsula, Nr 1274)
stations = [1777,1811,1274]

# extract anomaly
#fill_ismn = fill_ismn.groupby('time.month') - fill_ismn.groupby('time.month').mean()
#intp_ismn = intp_ismn.groupby('time.month') - intp_ismn.groupby('time.month').mean()
#orig_ismn = orig_ismn.groupby('time.month') - orig_ismn.groupby('time.month').mean()
#ismn = ismn.groupby('time.month') - ismn.groupby('time.month').mean()

# get only gap-filled data for gap-filled dataset
fill_ismn_del = fill_ismn.where(np.isnan(orig_ismn))

# calculate correlation
pcorr_orig = xr.corr(ismn, orig_ismn, dim='time')
pcorr_fill = xr.corr(ismn, fill_ismn_del, dim='time')

# plot station locations
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(25,10), layout='constrained')
gs0 = fig.add_gridspec(1,2)
levels = 9

gs00 = gs0[0].subgridspec(2,1)
gs01 = gs0[1].subgridspec(3,1)

ax1 = fig.add_subplot(gs00[0], projection=proj)
ax2 = fig.add_subplot(gs00[1], projection=proj)

fs = 15
levels = np.arange(-1,1.1,0.1)

cmap = plt.get_cmap('Greys')
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)

#landmask.plot(ax=ax1, cmap=cmap, add_colorbar=False)

im= ax1.scatter(pcorr_orig.lon, pcorr_orig.lat, c=pcorr_orig, transform=transf, cmap='seismic_r',
           vmin=-1, vmax=1, alpha=1, edgecolor='white', s=40)
ax1.coastlines()
ax1.scatter(pcorr_orig.sel(stations=stations).lon, pcorr_orig.sel(stations=stations).lat, 
           marker='x', c='red', transform=transf)

#im= ax2.scatter(pcorr_fill.lon, pcorr_fill.lat, c=pcorr_fill, transform=transf, cmap='seismic_r',
#           vmin=-1, vmax=1, alpha=1, edgecolor=None)
im= ax2.scatter(pcorr_fill.lon, pcorr_fill.lat, c=pcorr_fill, transform=transf, cmap='seismic_r',
           vmin=-1, vmax=1, alpha=1, edgecolor='white', s=40)
ax2.coastlines()
ax2.scatter(pcorr_fill.sel(stations=stations).lon, pcorr_fill.sel(stations=stations).lat, 
           c='red', transform=transf, marker='x')

ax1.set_global()
ax2.set_global()

cbar_ax = fig.add_axes([0.42, 0.15, 0.02, 0.6]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Pearson correlation')
ax1.set_title('a) original ESA-CCI surface layer soil moisture')
ax2.set_title('b) gap-filled ESA-CCI surface layer soil moisture')

legend_elements = [Line2D([0],[0],marker='x', color='white', markeredgecolor='red', label='selected stations')]
ax2.legend(handles=legend_elements, loc='upper right',
          bbox_to_anchor=(1,0))

# plot distribution of correlations
#fig = plt.figure(figsize=(10,4))
#ax = fig.add_subplot(111)
##ax.bar(np.arange(len(pcorr_orig)), pcorr_orig.sortby(pcorr_orig), color=col_miss)
##ax.bar(np.arange(len(pcorr_fill)), pcorr_fill.sortby(pcorr_fill), color=col_fill)
#ax.plot(np.arange(len(pcorr_orig)), pcorr_orig.sortby(pcorr_orig), color=col_miss)
#ax.plot(np.arange(len(pcorr_fill)), pcorr_fill.sortby(pcorr_fill), color=col_fill)
#ax.grid()
#ax.set_xlabel('stations')
#ax.set_ylabel('pearson correlation')
#ax.set_title('(f) cumulative histogram of station correlations', fontsize=fs)
#legend_elements = [Line2D([0], [0], color=col_miss, lw=4, label='original ESA-CCI'),
#                   Line2D([0], [0], color=col_fill, lw=4, label='gap-filled ESA-CCI')]
#ax.legend(handles=legend_elements)
#plt.savefig('ismn_pdf.png', dpi=300)
#plt.close()

#orig_stations = orig_ismn.sel(stations=stations).mrso
#fill_stations = fill_ismn.sel(stations=stations).mrso
#ismn_stations = ismn.sel(stations=stations).mrso

# plot example station
ax1 = fig.add_subplot(gs01[0])
ax2 = fig.add_subplot(gs01[1])
ax3 = fig.add_subplot(gs01[2])

#intp_ismn.sel(stations=stations[0]).plot(ax=ax1, color=col_intp, label='Interpolated ESA CCI')
fill_ismn.sel(stations=stations[0]).plot(ax=ax1, color=col_fill, label='Gap-filled ESA CCI')
orig_ismn.sel(stations=stations[0]).plot(ax=ax1, color=col_miss, label='ESA CCI with Gaps')
ismn.sel(stations=stations[0]).plot(ax=ax1, color=col_ismn, label='ISMN station')

intp_ismn.sel(stations=stations[1]).plot(ax=ax2, color=col_intp)
fill_ismn.sel(stations=stations[1]).plot(ax=ax2, color=col_fill)
orig_ismn.sel(stations=stations[1]).plot(ax=ax2, color=col_miss)
ismn.sel(stations=stations[1]).plot(ax=ax2, color=col_ismn)

intp_ismn.sel(stations=stations[2]).plot(ax=ax3, color=col_intp)
fill_ismn.sel(stations=stations[2]).plot(ax=ax3, color=col_fill)
orig_ismn.sel(stations=stations[2]).plot(ax=ax3, color=col_miss)
ismn.sel(stations=stations[2]).plot(ax=ax3, color=col_ismn)

ax1.set_title('c) Little River, SCAN Network, Georgia, USA')
ax2.set_title('d) N Piedmont Arec, SCAN Network, Virginia, USA')
ax3.set_title('e) Llanos de la Boveda, REMEDHUS Network, Spain')

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

ax1.legend(loc='lower left')

plt.savefig('ismn.jpeg', dpi=300)
