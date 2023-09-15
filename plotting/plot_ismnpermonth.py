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
intp_ismn = xr.open_dataarray(f'{esapath}data_intp_ismn.nc')
fill_ismn = xr.open_dataarray(f'{esapath}data_climfilled_ismn.nc')

# drop unnecessary columns
fill_ismn = fill_ismn.drop(('lat_cmip','lon_cmip','network','depth_start',
                'depth_end','koeppen','latlon_cmip'))
intp_ismn = intp_ismn.drop(('lat_cmip','lon_cmip','network','depth_start',
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
pcorr_intp = xr.full_like(ismn.groupby('time.month').mean(), np.nan)
pcorr_fill = xr.full_like(ismn.groupby('time.month').mean(), np.nan)
for i in range(1,13):
    ismn_tmp = ismn.where(ismn['time.month'] == i, drop=True)
    intp_ismn_tmp = intp_ismn.where(ismn['time.month'] == i, drop=True)
    fill_ismn_tmp = fill_ismn.where(ismn['time.month'] == i, drop=True)
    pcorr_intp[i-1,:] = xr.corr(ismn_tmp, intp_ismn_tmp, dim='time')
    pcorr_fill[i-1,:] = xr.corr(ismn_tmp, fill_ismn_tmp, dim='time')

# calc percentage of missing values per month average over stations
bothobs = np.logical_not(np.isnan(ismn) | np.isnan(orig_ismn))
perc1 = orig_ismn.groupby('time.month').count().mean(dim='stations')
perc2 = ismn.groupby('time.month').count().mean(dim='stations')
perc3 = bothobs.groupby('time.month').sum().mean(dim='stations')

# plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(111)

#import IPython; IPython.embed()
#perc1.to_series().plot.bar(ax=ax1, label='ESA CCI', alpha=1, color='orange',zorder=10)
#perc2.to_series().plot.bar(ax=ax1, label='ISMN', alpha=1, color='blue',zorder=5)
perc3.to_series().plot.bar(ax=ax1, label='both', alpha=1, color='grey')
#ax1.bar(np.arange(len(perc1)), perc3, label='both', color='grey')
ax1.set_ylabel('# of months with data in 1995-2020')
ax1.set_xlabel('month')
ax1.set_title('comparison of gap-filling methods on ISMN data')
ax1.set_ylim([0,26])

ax2 = ax1.twinx() 
tmp = pcorr_intp.median(dim='stations').values
tmp2 = pcorr_fill.median(dim='stations').values
ax2.plot(tmp, label='interpolation', linewidth=2)
ax2.plot(tmp2, label='CLIMFILL', linewidth=2)
#pcorr_intp.median(dim='stations').plot(ax=ax2, label='interpolation', linewidth=2)
#pcorr_fill.median(dim='stations').plot(ax=ax2, label='CLIMFILL', linewidth=2)
ax2.set_ylabel('pearson correlation')
ax2.set_xlabel('month')
ax2.legend()

plt.savefig('test.jpeg', dpi=300)
