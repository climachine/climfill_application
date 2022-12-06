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

mask_orig = xr.open_dataset(f'{esapath}mask_orig.nc')
mask_cv = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc')

# (optional) calculate anomalies
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
intp = intp.groupby('time.month') - intp.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# normalise values for RMSE plotting
datamean = orig.mean()
datastd = orig.std()
orig = (orig - datamean) / datastd
intp = (intp - datamean) / datastd
fill = (fill - datamean) / datastd

# select verification year
mask_orig = mask_orig.sel(time=verification_year).load()
mask_cv = mask_cv.sel(time=verification_year).load()
orig = orig.sel(time=verification_year).load()
intp = intp.sel(time=verification_year).load()
fill = fill.sel(time=verification_year).load()

# calculate mask of verification points
mask_cv = np.logical_and(np.logical_not(mask_orig), mask_cv)

# mask everything except verification points
orig = orig.where(mask_cv)
intp = intp.where(mask_cv)
fill = fill.where(mask_cv)

# sort data
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','temperature_obs','precipitation_obs',
            'burned_area','diurnal_temperature_range','snow_cover_fraction'] 
varnames_plot = ['surface layer \nsoil moisture','surface temperature',
                 'precipitation (sat)','terrestrial water storage','2m temperature',
                 'precipitation (ground)', 'burned area',
                 'diurnal temperature range sfc','snow cover fraction'] 
orig = orig.to_array().reindex(variable=varnames)
intp = intp.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)

# calc metrics
#corr_intp = xr.corr(orig, intp, dim=('lat','lon','time'))
#corr_fill = xr.corr(orig, fill, dim=('lat','lon','time'))
#rmse_intp = calc_rmse(orig, intp, dim=('lat','lon','time'))
#rmse_fill = calc_rmse(orig, fill, dim=('lat','lon','time'))
corr_intp = xr.corr(orig, intp, dim=('time')).median(dim=('lat','lon'))
corr_fill = xr.corr(orig, fill, dim=('time')).median(dim=('lat','lon'))
rmse_intp = calc_rmse(orig, intp, dim=('time')).median(dim=('lat','lon'))
rmse_fill = calc_rmse(orig, fill, dim=('time')).median(dim=('lat','lon'))

corr_intp_err = xr.corr(orig, intp, dim=('time')).std(dim=('lat','lon'))
corr_fill_err = xr.corr(orig, fill, dim=('time')).std(dim=('lat','lon'))
rmse_intp_err = calc_rmse(orig, intp, dim=('time')).std(dim=('lat','lon'))
rmse_fill_err = calc_rmse(orig, fill, dim=('time')).std(dim=('lat','lon'))

# calc mean, stderr

# plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
x_pos =np.arange(0,2*len(corr_intp),2)
wd = 0.3

ax1.bar(x_pos, corr_intp, width=wd, yerr=corr_intp_err, label='Interpolation')
ax1.bar(x_pos+wd, corr_fill, width=wd, yerr=corr_fill_err, label='CLIMFILL')

ax2.bar(x_pos, rmse_intp, yerr=rmse_intp_err, width=wd)
ax2.bar(x_pos+wd, rmse_fill, yerr=rmse_fill_err, width=wd)

ax1.set_xticks([])
ax1.legend()
ax2.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90)
ax1.set_ylim([0,1]) 
ax2.set_ylim([0,2]) 
ax1.set_xlim([-1,18])
ax2.set_xlim([-1,18])

ax1.set_title('pearson correlation of verification pts')
ax2.set_title('RSME of verification pts')

plt.subplots_adjust(bottom=0.3)
#plt.show()
plt.savefig('verification.png')
