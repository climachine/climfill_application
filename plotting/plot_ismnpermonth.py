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
ismn = xr.open_dataset('/net/so4/landclim/bverena/large_files/optimal/df_gaps.nc')
orig_ismn = xr.open_dataarray(f'{esapath}data_orig_ismn.nc')
fill_ismn = xr.open_dataarray(f'{esapath}data_climfilled_ismn.nc')

# drop unnecessary columns
fill_ismn = fill_ismn.drop(('lat_cmip','lon_cmip','network','depth_start',
                'depth_end','koeppen','latlon_cmip'))
orig_ismn = orig_ismn.drop(('lat_cmip','lon_cmip','network','depth_start',
                'depth_end','koeppen','latlon_cmip'))
ismn = ismn.drop(('lat_cmip','lon_cmip','network','depth_start',
                'depth_end','koeppen','latlon_cmip'))

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
ismn = ismn.where(mask, drop=True).mrso

# calculate correlation per month
pcorr_orig = xr.full_like(ismn.groupby('time.month').mean(), np.nan)
pcorr_fill = xr.full_like(ismn.groupby('time.month').mean(), np.nan)
for i in range(1,13):
    ismn_tmp = ismn.where(ismn['time.month'] == i, drop=True)
    orig_ismn_tmp = orig_ismn.where(ismn['time.month'] == i, drop=True)
    fill_ismn_tmp = fill_ismn.where(ismn['time.month'] == i, drop=True)
    pcorr_orig[i-1,:] = xr.corr(ismn_tmp, orig_ismn_tmp, dim='time')
    pcorr_fill[i-1,:] = xr.corr(ismn_tmp, fill_ismn_tmp, dim='time')

# calc percentage of missing values per month average over stations
bothobs = np.logical_not(np.isnan(ismn) | np.isnan(orig_ismn))
perc1 = (orig_ismn.groupby('time.month').count().mean(dim='stations') / 26 )
perc2 = (ismn.groupby('time.month').count().mean(dim='stations') / 26 )
perc3 = (bothobs.groupby('time.month').sum().mean(dim='stations') / 26 )

# plot
fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#import IPython; IPython.embed()
#perc1.to_series().plot.bar(ax=ax1, label='ESA CCI', alpha=1, color='orange',zorder=10)
#perc2.to_series().plot.bar(ax=ax1, label='ISMN', alpha=1, color='blue',zorder=5)
#perc3.to_series().plot.bar(ax=ax1, label='both', alpha=1, color='green',zorder=0)
ax1.bar(np.arange(len(perc1))-0.25, perc1, label='ESA CCI', color='orange', width=0.25)
ax1.bar(np.arange(len(perc1))+0.00, perc2, label='ISMN', color='blue', width=0.25)
ax1.bar(np.arange(len(perc1))+0.25, perc3, label='both', color='green', width=0.25)
ax1.set_ylabel('fraction of available observations')
ax1.set_xlabel('month')
ax1.set_title('a) fraction of available observations per month')
ax1.legend()

pcorr_orig.median(dim='stations').plot(ax=ax2, label='original')
pcorr_fill.median(dim='stations').plot(ax=ax2, label='gap-filled')
ax2.set_ylabel('pearson correlation')
ax2.set_xlabel('month')
ax2.set_title('b) pearson correlation per month')
ax2.legend()

plt.savefig('test.jpeg', dpi=300)
