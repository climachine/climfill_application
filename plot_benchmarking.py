"""
NAMESTRING
"""

import argparse
import numpy as np
import regionmask
from scipy.spatial.distance import jensenshannon as js
import xarray as xr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
verification_year = '2004'

def calc_rmse(dat1, dat2, dim):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

# NOTE:
# TWS does not have any originally missing values in 2004
# JS not possible bec needs times where all 8 vars are observed (i.e. in the
#      verification set) at the same time. ditched for now
# INIT GUESS LEFT OUT BEC HAS MISSING VALUES AND IS THEREFORE MUCH BETTER (ON 
# THE NOT MISSING POINTS OBVSLY) AND ON THE MISSING VALUES IT IS REPRESENTED
# IN INTP

# read data
orig = xr.open_dataset(f'{esapath}data_orig.nc')
intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
era5 = xr.open_dataset(f'{esapath}data_era5land.nc')

#mask_orig = xr.open_dataset(f'{esapath}mask_orig.nc')
#mask_cv = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc')

# (optional) calculate anomalies
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
era5 = era5.groupby('time.month') - era5.groupby('time.month').mean()
intp = intp.groupby('time.month') - intp.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# select verification year
#mask_orig = mask_orig.sel(time=verification_year).load()
#mask_cv = mask_cv.sel(time=verification_year).load()
#orig = orig.sel(time=verification_year).load()
#intp = intp.sel(time=verification_year).load()
#fill = fill.sel(time=verification_year).load()

# calculate mask of verification points
#mask_cv = np.logical_and(np.logical_not(mask_orig), mask_cv)

# mask everything except verification points
#orig = orig.where(mask_cv)
#intp = intp.where(mask_cv)
#fill = fill.where(mask_cv)

# sort data
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','snow_cover_fraction',
            'temperature_obs','precipitation_obs','burned_area'] #hardcoded for now
orig = orig.to_array().reindex(variable=varnames)
era5 = era5.to_array().reindex(variable=varnames)
intp = intp.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)

# aggregate to regions
# needs to be before corr bec otherwise all nans are ignored in orig
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))

orig = orig.groupby(regions).mean()
era5 = era5.groupby(regions).mean()
intp = intp.groupby(regions).mean()
fill = fill.groupby(regions).mean()

# calc metrics
corr_orig = xr.corr(orig, era5, dim=('time'))
corr_intp = xr.corr(intp, era5, dim=('time'))
corr_fill = xr.corr(fill, era5, dim=('time'))
rmse_orig = calc_rmse(orig, era5, dim=('time'))
rmse_intp = calc_rmse(intp, era5, dim=('time'))
rmse_fill = calc_rmse(fill, era5, dim=('time'))

# mean over regions
corr_orig = corr_orig.mean(dim='mask')
corr_intp = corr_intp.mean(dim='mask')
corr_fill = corr_fill.mean(dim='mask')
rmse_orig = rmse_orig.mean(dim='mask')
rmse_intp = rmse_intp.mean(dim='mask')
rmse_fill = rmse_fill.mean(dim='mask')

# plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
x_pos =np.arange(0,2*len(corr_orig),2)
wd = 0.3

ax1.bar(x_pos-wd, corr_orig, width=wd, label='orig')
ax1.bar(x_pos, corr_intp, width=wd, label='intp')
ax1.bar(x_pos+wd, corr_fill, width=wd, label='fill')

ax2.bar(x_pos-wd, rmse_orig, width=wd)
ax2.bar(x_pos, rmse_intp, width=wd)
ax2.bar(x_pos+wd, rmse_fill, width=wd)

ax1.set_xticks([])
ax1.legend()
ax2.set_xticks(x_pos+0.5*wd, varnames, rotation=90)
ax2.set_ylim([0,15]) #TODO debug remove
ax1.set_xlim([-1,16])
ax2.set_xlim([-1,16])

ax1.set_title('pearson correlation')
ax2.set_title('RSME')

plt.subplots_adjust(bottom=0.3)
plt.show()
