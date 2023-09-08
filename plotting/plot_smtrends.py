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
ms_to_year = 365 * 24 * 3600 * 10**9
era5_trends = era5.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)*ms_to_year
erag_trends = erag.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)*ms_to_year
orig_trends = orig.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)*ms_to_year
fill_trends = fill.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients.sel(degree=1)*ms_to_year

# aggregate by region
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
#regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
#regions = regions.where(~np.isnan(landmask))
#
#era5_trends = era5_trends.groupby(regions).mean()
#erag_trends = erag_trends.groupby(regions).mean()
#orig_trends = orig_trends.groupby(regions).mean()
#fill_trends = fill_trends.groupby(regions).mean()
#
## expand back to worldmap
#def expand_to_worldmap(data,regions):
#    test = xr.full_like(regions, np.nan)
#    for region, r in zip(range(int(regions.max().item())), data):
#        test = test.where(regions != region, r) # unit stations per bio square km
#
#    return test
#
#era5_trends = expand_to_worldmap(era5_trends,regions)
#erag_trends = expand_to_worldmap(erag_trends,regions)
#orig_trends = expand_to_worldmap(orig_trends,regions)
#fill_trends = expand_to_worldmap(fill_trends,regions)

# set ocean blue, land grey
era5_trends = era5_trends.where(np.logical_not(landmask), 10) # ocean blue
erag_trends = erag_trends.where(np.logical_not(landmask), 10) # ocean blue
orig_trends = orig_trends.where(np.logical_not(landmask), 10) # ocean blue
fill_trends = fill_trends.where(np.logical_not(landmask), 10) # ocean blue

# plot map
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(13,13), layout='constrained')
gs0 = fig.add_gridspec(2,1)

gs00 = gs0[0].subgridspec(2,3, width_ratios=[1,1,0.3])
gs01 = gs0[1].subgridspec(1,1)

ax1 = fig.add_subplot(gs00[0,0], projection=proj)
ax2 = fig.add_subplot(gs00[0,1], projection=proj)
ax3 = fig.add_subplot(gs00[1,0], projection=proj)
ax4 = fig.add_subplot(gs00[1,1], projection=proj)
fs = 15
levels = 9

cmap = plt.get_cmap('seismic_r')
cmap.set_over('aliceblue')
cmap.set_bad('lightgrey')

vmin= -0.002
vmax = 0.002
im = orig_trends.plot(ax=ax1, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap, transform=transf, levels=levels)
erag_trends.plot(ax=ax2, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap, transform=transf, levels=levels)
fill_trends.plot(ax=ax3, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap, transform=transf, levels=levels)
era5_trends.plot(ax=ax4, add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap, transform=transf, levels=levels)

cbar_ax = fig.add_axes([0.88, 0.51, 0.02, 0.4]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Surface layer soil moisture trends [$m^3\;m^{-3}$ per year]')
fig.suptitle('Soil moisture trends, 1996-2020')

ax1.set_title('a) ESA CCI original')
ax3.set_title('b) ESA CCI gap-filled')
ax2.set_title('c) ERA5-Land gaps deleted')
ax4.set_title('d) ERA5-Land original')
#plt.savefig('trendmaps.png', dpi=300)

######## PLOT TIMELINE ##############

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
orig_mean = orig.mean(dim='time')
fill_mean = fill.mean(dim='time')
era5_mean = era5.mean(dim='time')
erag_mean = erag.mean(dim='time')

# subtract mean
orig = orig - orig_mean
fill = fill - fill_mean
era5 = era5 - era5_mean
erag = erag - erag_mean

col_fill = 'darkgrey'
col_miss = 'indianred'
col_ismn = 'black'
col_erag = 'black'

# plot
ax5 = fig.add_subplot(gs01[0,0])
im = orig.plot(ax=ax5, color=col_fill, label='ESA CCI original')
fill.plot(ax=ax5, color=col_miss, label='ESA CCI gap-filled')
era5.plot(ax=ax5, color=col_ismn, label='ERA5-Land original')
erag.plot(ax=ax5, color=col_erag, label='satellite-observable ERA5-Land', linestyle='--')
ax5.set_ylabel('surface layer soil moisture, \ndeviations from 1996-2020 \naverage $[m^{3}\;m^{-3}]$')
ax5.axhline(y=0, linewidth=0.5,c='black')
ax5.set_xlabel('time')
ax5.set_title('e) Northern Extratropics mean')
ax5.legend(loc='lower right')
ax5.set_ylim([-0.02,0.015])
plt.savefig('trends.jpeg',dpi=300)
