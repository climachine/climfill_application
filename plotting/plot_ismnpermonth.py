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
import IPython; IPython.embed()
for i in range(1,13):
    ismn_tmp = ismn.where(ismn['time.month'] == i, drop=True)
    orig_ismn_tmp = orig_ismn.where(ismn['time.month'] == i, drop=True)
    fill_ismn_tmp = fill_ismn.where(ismn['time.month'] == i, drop=True)
    pcorr_orig = xr.corr(ismn_tmp, orig_ismn_tmp, dim='time')
    pcorr_fill = xr.corr(ismn_tmp, fill_ismn_tmp, dim='time')

# aggregate per month
ismn = ismn.groupby('time.month').mean()
orig_ismn = orig_ismn.groupby('time.month').mean()
fill_ismn = fill_ismn.groupby('time.month').mean()


