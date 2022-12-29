"""
NAMESTRING
"""

import argparse
import numpy as np
import regionmask
from scipy.spatial.distance import jensenshannon as js
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
verification_year = slice('2004','2005')

varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','temperature_obs','precipitation_obs',
            'burned_area','diurnal_temperature_range','snow_cover_fraction'] 
varnames_plot = ['surface layer \nsoil moisture','surface temperature',
                 'precipitation (sat)','terrestrial water storage','2m temperature',
                 'precipitation (ground)', 'burned area',
                 'diurnal temperature range sfc','snow cover fraction'] 

def calc_rmse(dat1, dat2, dim):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

def assemble_verification_cube(testcase, numbers=[0,1,2]):
    for n, no in enumerate(numbers):
        fill = xr.open_dataset(f'{esapath}{testcase}/verification/dataveri{no}_climfilled.nc')
        mask_cv = xr.open_dataset(f'{esapath}{testcase}/verification/maskveri{no}.nc')
        mask = np.logical_and(np.logical_not(mask_orig), mask_cv)
        fill = fill.to_array().reindex(variable=varnames)
        mask = mask.to_array().reindex(variable=varnames)

        if n == 0:
            res = xr.full_like(fill, np.nan)
        res = res.where(np.logical_not(mask), fill)
    return res
# TODO mask is the same for two subsequent months???
# use trained Regr from full timeline?

# NOTE:
# TWS does not have any originally missing values in 2004
# JS not possible bec needs times where all 8 vars are observed (i.e. in the
#      verification set) at the same time. ditched for now
# INIT GUESS LEFT OUT BEC HAS MISSING VALUES AND IS THEREFORE MUCH BETTER (ON 
# THE NOT MISSING POINTS OBVSLY) AND ON THE MISSING VALUES IT IS REPRESENTED
# IN INTP


# read data
#fill = assemble_verification_cube(testcase)
#fill = fill.to_dataset('variable')
orig = xr.open_dataset(f'{esapath}data_orig.nc')
intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc')
#intp = xr.open_dataset(f'{esapath}test5/data_interpolated.nc') # DeEUG
mask_orig = xr.open_dataset(f'{esapath}mask_orig.nc')

fill = xr.open_dataset(f'{esapath}{testcase}/verification/dataveri2_climfilled.nc')
mask_cv = xr.open_dataset(f'{esapath}{testcase}/verification/maskveri2.nc')
#fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
#mask_cv = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc')

mask = np.logical_and(np.logical_not(mask_orig), mask_cv)

# (optional) calculate anomalies
#orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
#intp = intp.groupby('time.month') - intp.groupby('time.month').mean()
#fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# normalise values for RMSE plotting
datamean = orig.mean()
datastd = orig.std()
orig = (orig - datamean) / datastd
intp = (intp - datamean) / datastd
fill = (fill - datamean) / datastd

# select verification year
#mask_orig = mask_orig.sel(time=verification_year).load()
#mask_cv = mask_cv.sel(time=verification_year).load()
orig = orig.sel(time=verification_year).load()
intp = intp.sel(time=verification_year).load()
fill = fill.sel(time=verification_year).load()
mask = mask.sel(time=verification_year).load()

# sort data
orig = orig.to_array().reindex(variable=varnames)
intp = intp.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)
mask = mask.to_array().reindex(variable=varnames)

# calculate mask of verification points
#mask_cv = np.logical_and(np.logical_not(mask_orig), mask_cv)

# mask everything except verification points
# problem: RMSE is same if masked or not
# corr is (-1,1) if less than 3 values; almost never more than 2 values, since
# 3 months missing in a year is unlikely in minicube deletion
# solution for now: not remove orig values, but arti. high corr produced
#orig = orig.where(np.logical_not(np.isnan(fill)))
orig = orig.where(mask)
intp = intp.where(mask)
fill = fill.where(mask)

# calc metrics
corr_intp = xr.corr(orig, intp, dim=('time'))
corr_fill = xr.corr(orig, fill, dim=('time'))
rmse_intp = calc_rmse(orig, intp, dim=('time'))
rmse_fill = calc_rmse(orig, fill, dim=('time'))

#corr_intp = xr.corr(orig, intp, dim=('lat','lon','time'))
#corr_fill = xr.corr(orig, fill, dim=('lat','lon','time'))
#rmse_intp = calc_rmse(orig, intp, dim=('lat','lon','time'))
#rmse_fill = calc_rmse(orig, fill, dim=('lat','lon','time'))
#corr_intp = xr.corr(orig, intp, dim=('time')).median(dim=('lat','lon'))
#corr_fill = xr.corr(orig, fill, dim=('time')).median(dim=('lat','lon'))
#rmse_intp = calc_rmse(orig, intp, dim=('time')).median(dim=('lat','lon'))
#rmse_fill = calc_rmse(orig, fill, dim=('time')).median(dim=('lat','lon'))
#
#corr_intp_err = xr.corr(orig, intp, dim=('time')).std(dim=('lat','lon'))
#corr_fill_err = xr.corr(orig, fill, dim=('time')).std(dim=('lat','lon'))
#rmse_intp_err = calc_rmse(orig, intp, dim=('time')).std(dim=('lat','lon'))
#rmse_fill_err = calc_rmse(orig, fill, dim=('time')).std(dim=('lat','lon'))

# TODO: for now: remove from corr all timepoints with less than 3 missing vals
corrmask = (np.logical_not(np.isnan(fill)).sum(dim='time') > 3).reindex(variable=varnames)
corr_intp = corr_intp.where(corrmask)
corr_fill = corr_fill.where(corrmask)

# aggregate lat lon for boxplot
corr_intp = corr_intp.stack(landpoints=('lat','lon'))
corr_fill = corr_fill.stack(landpoints=('lat','lon'))
rmse_intp = rmse_intp.stack(landpoints=('lat','lon'))
rmse_fill = rmse_fill.stack(landpoints=('lat','lon'))

# remove nan for boxplot
def filter_data(data):
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data.values, mask.values)]
    return filtered_data
corr_intp = filter_data(corr_intp)
corr_fill = filter_data(corr_fill)
rmse_intp = filter_data(rmse_intp)
rmse_fill = filter_data(rmse_fill)

# plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
x_pos =np.arange(0,2*len(corr_intp),2)
wd = 0.5

boxplot_kwargs = {'notch': False,
                  'patch_artist': True}
col_intp = 'gold'
col_fill = 'coral'

b1 = ax1.boxplot(positions=x_pos, x=corr_intp, showfliers=False, **boxplot_kwargs)
b2 = ax1.boxplot(positions=x_pos+wd, x=corr_fill, showfliers=False, **boxplot_kwargs)

b3 = ax2.boxplot(positions=x_pos, x=rmse_intp, showfliers=False, **boxplot_kwargs)
b4 = ax2.boxplot(positions=x_pos+wd, x=rmse_fill, showfliers=False, **boxplot_kwargs)

for box in b1['boxes'] + b3['boxes']:
    box.set_facecolor(col_intp)
for box in b2['boxes'] + b4['boxes']:
    box.set_facecolor(col_fill)

ax1.set_xticks([])
legend_elements = [Patch(facecolor=col_intp, edgecolor=col_intp, label='Interpolation'),
                   Patch(facecolor=col_fill, edgecolor=col_fill, label='CLIMFILL')]
ax1.legend(handles=legend_elements, loc='upper right')
ax2.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90)
ax1.set_ylim([0,1.1]) 
ax2.set_ylim([0,2]) 
ax1.set_xlim([-1,18])
ax2.set_xlim([-1,18])

ax1.set_title('Pearson correlation of verification pts')
ax2.set_title('RSME (normalised) of verification pts')

plt.subplots_adjust(bottom=0.3)
#plt.show()
plt.savefig('verification.png')

# plot
#fig = plt.figure(figsize=(10,10))
#ax1 = fig.add_subplot(211)
#ax2 = fig.add_subplot(212)
#x_pos =np.arange(0,2*len(corr_intp),2)
#wd = 0.3
#
#ax1.bar(x_pos, corr_intp, width=wd, yerr=corr_intp_err, label='Interpolation')
#ax1.bar(x_pos+wd, corr_fill, width=wd, yerr=corr_fill_err, label='CLIMFILL')
#
#ax2.bar(x_pos, rmse_intp, yerr=rmse_intp_err, width=wd)
#ax2.bar(x_pos+wd, rmse_fill, yerr=rmse_fill_err, width=wd)
#
#ax1.set_xticks([])
#ax1.legend()
#ax2.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90)
#ax1.set_ylim([0,1.2]) 
#ax2.set_ylim([0,2]) 
#ax1.set_xlim([-1,18])
#ax2.set_xlim([-1,18])
#
#ax1.set_title('pearson correlation of verification pts')
#ax2.set_title('RSME of verification pts')
#
#plt.subplots_adjust(bottom=0.3)
##plt.show()
#plt.savefig('verification.png')
