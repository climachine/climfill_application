"""
NAMESTRING
"""

import argparse
import numpy as np
import regionmask
from scipy.spatial.distance import jensenshannon as js
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
verification_year = '2004'

def calc_rmse(dat1, dat2, dim):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

def area_weighted(lat):
    return np.cos(np.deg2rad(lat))

# read data
orig = xr.open_dataset(f'{esapath}data_orig.nc').soil_moisture
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc').soil_moisture
era5 = xr.open_dataset(f'{esapath}data_era5land.nc').soil_moisture

# create gappy era5
erag = era5.where(np.logical_not(np.isnan(orig)))

# select dry season
orig = orig.sel(time=orig['time.season']=='JJA')
fill = fill.sel(time=fill['time.season']=='JJA')
era5 = era5.sel(time=era5['time.season']=='JJA')
erag = erag.sel(time=erag['time.season']=='JJA')

# resample to yearly
orig = orig.resample(time='Y').mean()
fill = fill.resample(time='Y').mean()
era5 = era5.resample(time='Y').mean()
erag = erag.resample(time='Y').mean()

# calculate trends per grid point
era5_trends = era5.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)
erag_trends = erag.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)
orig_trends = orig.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)
fill_trends = fill.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)

# plot map
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(221, projection=proj)
ax2 = fig.add_subplot(222, projection=proj)
ax3 = fig.add_subplot(223, projection=proj)
ax4 = fig.add_subplot(224, projection=proj)
fs = 15

vmin= -1e-19
vmax = 1e-19
im = orig_trends.plot(ax=ax1, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='coolwarm_r', transform=transf)
erag_trends.plot(ax=ax2, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='coolwarm_r', transform=transf)
fill_trends.plot(ax=ax3, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='coolwarm_r', transform=transf)
era5_trends.plot(ax=ax4, add_colorbar=False, vmin=vmin, vmax=vmax, cmap='coolwarm_r', transform=transf)

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.6]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('RMSE on normalized values', fontsize=fs)
fig.suptitle('(a) Soil moisture trends, 1996-2020', fontsize=20)

ax1.set_title('ESA CCI original', fontsize=fs)
ax2.set_title('ERA5-Land gaps deleted', fontsize=fs)
ax3.set_title('ESA CCI gap-filled', fontsize=fs)
ax4.set_title('ERA5-Land original', fontsize=fs)
plt.savefig('trendmaps.png', dpi=300)

# northern extratropics mean
orig = orig.sel(lat=slice(23.5,90))
fill = fill.sel(lat=slice(23.5,90))
era5 = era5.sel(lat=slice(23.5,90))
erag = erag.sel(lat=slice(23.5,90))

orig = orig.weighted(area_weighted(orig.lat)).mean(dim=('lat','lon'))
fill = fill.weighted(area_weighted(fill.lat)).mean(dim=('lat','lon'))
era5 = era5.weighted(area_weighted(era5.lat)).mean(dim=('lat','lon'))
erag = erag.weighted(area_weighted(erag.lat)).mean(dim=('lat','lon'))

# 2007-2018 as baseline
#orig_mean = orig.sel(time=slice('2007','2018')).mean(dim='time')
#fill_mean = fill.sel(time=slice('2007','2018')).mean(dim='time')
#era5_mean = era5.sel(time=slice('2007','2018')).mean(dim='time')
#erag_mean = erag.sel(time=slice('2007','2018')).mean(dim='time')
orig_mean = orig.mean(dim='time')
fill_mean = fill.mean(dim='time')
era5_mean = era5.mean(dim='time')
erag_mean = erag.mean(dim='time')
##
##import IPython; IPython.embed()
orig = orig - orig_mean
fill = fill - fill_mean
era5 = era5 - era5_mean
erag = erag - erag_mean


# scale by 100 to get values less close to zero
#orig = orig*100
#fill = fill*100
#era5 = era5*100
#erag = erag*100

col_fill = 'darkgrey'
col_miss = 'indianred'
col_ismn = 'black'
col_erag = 'black'

# plot
fig = plt.figure(figsize=(11,4))
ax = fig.add_subplot(111)
im = orig.plot(ax=ax, color=col_fill, label='ESA CCI original')
fill.plot(ax=ax, color=col_miss, label='ESA CCI gap-filled')
era5.plot(ax=ax, color=col_ismn, label='ERA5-Land original')
erag.plot(ax=ax, color=col_erag, label='satellite-observable ERA5-Land', linestyle='--')
ax.set_ylabel('surface layer soil moisture, \ndeviations from 1996-2020 \naverage $[m^{3}\;m^{-3}]$', fontsize=fs)
ax.set_xlabel('time', fontsize=fs)
fig.suptitle('(b) Northern Extratropics', fontsize=fs)
ax.legend(fontsize=fs, loc='lower right')
ax.set_ylim([-0.02,0.015])
plt.savefig('sm_test3.pdf')
