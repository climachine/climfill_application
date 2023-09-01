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
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(221, projection=proj)
ax2 = fig.add_subplot(222, projection=proj)
ax3 = fig.add_subplot(223, projection=proj)
ax4 = fig.add_subplot(224, projection=proj)
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

#regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
#                                             ax=ax1, add_label=False, projection=transf)
#regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
#                                             ax=ax2, add_label=False, projection=transf)
#regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
#                                             ax=ax3, add_label=False, projection=transf)
#regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
#                                             ax=ax4, add_label=False, projection=transf)

cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.6]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Surface layer soil moisture trends [$m^3\;m^{-3}$ per year]', fontsize=fs)
fig.suptitle('(a) Soil moisture trends, 1996-2020', fontsize=20)

#ax1.coastlines()
#ax2.coastlines()
#ax3.coastlines()
#ax4.coastlines()

ax1.set_title('ESA CCI original', fontsize=fs)
ax2.set_title('ERA5-Land gaps deleted', fontsize=fs)
ax3.set_title('ESA CCI gap-filled', fontsize=fs)
ax4.set_title('ERA5-Land original', fontsize=fs)
plt.savefig('trendmaps.png', dpi=300)

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
