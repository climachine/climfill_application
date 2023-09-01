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

# read climfill data
orig = xr.open_dataset(f'{esapath}data_orig.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
era5 = xr.open_dataset(f'{esapath}data_era5land.nc')

# (optional) calculate anomalies
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
era5 = era5.groupby('time.month') - era5.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# sort data
varnames = ['soil_moisture','surface_temperature','precipitation',
            'temperature_obs', 'precipitation_obs','snow_cover_fraction',
            'diurnal_temperature_range'] #hardcoded for now
orig = orig.to_array().reindex(variable=varnames)
era5 = era5.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)

# normalise values for RMSE plotting
datamean = orig.mean()
datastd = orig.std()
orig = (orig - datamean) / datastd
era5 = (era5 - datamean) / datastd
fill = (fill - datamean) / datastd


# aggregate to regions
# needs to be before corr bec otherwise all nans are ignored in orig
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))

orig = orig.groupby(regions).mean()
era5 = era5.groupby(regions).mean()
fill = fill.groupby(regions).mean()

# calc metrics
corr_orig = xr.corr(orig, era5, dim=('time'))
corr_fill = xr.corr(fill, era5, dim=('time'))
rmse_orig = calc_rmse(orig, era5, dim=('time'))
rmse_fill = calc_rmse(fill, era5, dim=('time'))

# remove nan for boxplot
def filter_data(data):
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data.values, mask.values)]
    return filtered_data
corr_orig = filter_data(corr_orig)
corr_fill = filter_data(corr_fill)
rmse_orig = filter_data(rmse_orig)
rmse_fill = filter_data(rmse_fill)
corr_orig_ismn = corr_orig_ismn[~np.isnan(corr_orig_ismn)].values
corr_fill_ismn = corr_fill_ismn[~np.isnan(corr_fill_ismn)].values
rmse_orig_ismn = rmse_orig_ismn[~np.isnan(rmse_orig_ismn)].values
rmse_fill_ismn = rmse_fill_ismn[~np.isnan(rmse_fill_ismn)].values

# prepent ismn to era results
#corr_orig = [corr_orig_ismn] + corr_orig
#corr_fill = [corr_fill_ismn] + corr_fill
#rmse_orig = [rmse_orig_ismn] + rmse_orig
#rmse_fill = [rmse_fill_ismn] + rmse_fill

# plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
x_pos =np.arange(0,2*len(corr_orig),2)
wd = 0.5
fs = 15

varnames_plot = ['SM','LST','PSAT', #for order of plots
            'T2M','P2M',
            'DTR', 'SCF']
boxplot_kwargs = {'notch': False,
                  'patch_artist': True}
col_fill = 'coral'
col_miss = 'steelblue'

b1 = ax1.boxplot(positions=x_pos, x=corr_fill, showfliers=False, **boxplot_kwargs)
b2 = ax1.boxplot(positions=x_pos+wd, x=corr_orig, showfliers=False, **boxplot_kwargs)

b3 = ax2.boxplot(positions=x_pos, x=rmse_fill, showfliers=False, **boxplot_kwargs)
b4 = ax2.boxplot(positions=x_pos+wd, x=rmse_orig, showfliers=False, **boxplot_kwargs)

for box in b1['boxes'] + b3['boxes']:
    box.set_facecolor(col_fill)
for box in b2['boxes'] + b4['boxes']:
    box.set_facecolor(col_miss)
for median in b1['medians'] + b2['medians'] + b3['medians'] + b4['medians']:
    median.set_color('black')

#ax1.set_xticks([])
ax1.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90)
ax2.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90)
ax1.set_ylim([0,1]) 
ax2.set_ylim([0,1.4]) 
ax1.set_xlim([-1,14])
ax2.set_xlim([-1,14])

ax1.set_ylabel('Pearson correlation coefficient', fontsize=fs)
ax2.set_ylabel('RMSE on normalized values', fontsize=fs)
fig.suptitle('(a) Benchmarking scores', fontsize=20)

ax1.set_xticklabels(varnames_plot, fontsize=fs)
ax2.set_xticklabels(varnames_plot, fontsize=fs)

legend_elements = [Patch(facecolor=col_fill, edgecolor='black', label='CLIMFILL'),
                   Patch(facecolor=col_miss, edgecolor='black', label='With Gaps')]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=fs)
#plt.subplots_adjust(bottom=0.3)
plt.savefig('benchmarking.png')
