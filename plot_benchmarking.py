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

# read era data
era5 = xr.open_dataset(f'{esapath}data_era5land.nc')

# read ismn data
orig_ismn = xr.open_dataset(f'{esapath}data_orig_ismn.nc')
fill_ismn = xr.open_dataset(f'{esapath}data_climfilled_ismn.nc')
ismn = xr.open_dataset('/net/so4/landclim/bverena/large_files/df_gaps.nc')

# remove all ismn stations with less than 3 obs because xr.corr gives 1/-1
tmp = np.logical_not(np.isnan(ismn.mrso)).sum(dim='time')
ismn = ismn.where(tmp >= 3, drop=True)
orig_ismn = orig_ismn.where(tmp >= 3, drop=True)
fill_ismn = fill_ismn.where(tmp >= 3, drop=True)

#mask_orig = xr.open_dataset(f'{esapath}mask_orig.nc')
#mask_cv = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc')

# (optional) calculate anomalies
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
era5 = era5.groupby('time.month') - era5.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# normalise values for RMSE plotting
datamean = orig.mean()
datastd = orig.std()
orig = (orig - datamean) / datastd
era5 = (era5 - datamean) / datastd
fill = (fill - datamean) / datastd

# sort data
varnames = ['soil_moisture','surface_temperature','precipitation',
            'temperature_obs',
            'precipitation_obs','diurnal_temperature_range',
            'snow_cover_fraction'] #hardcoded for now
orig = orig.to_array().reindex(variable=varnames)
era5 = era5.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)

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
corr_orig_ismn = xr.corr(orig_ismn.mrso, ismn.mrso, dim='time')
corr_fill_ismn = xr.corr(fill_ismn.mrso, ismn.mrso, dim='time')
rmse_orig_ismn = calc_rmse(orig_ismn.mrso, ismn.mrso, dim='time')
rmse_fill_ismn = calc_rmse(fill_ismn.mrso, ismn.mrso, dim='time')

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
corr_orig = [corr_orig_ismn] + corr_orig
corr_fill = [corr_fill_ismn] + corr_fill
rmse_orig = [rmse_orig_ismn] + rmse_orig
rmse_fill = [rmse_fill_ismn] + rmse_fill

# plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
x_pos =np.arange(0,2*len(corr_orig),2)
wd = 0.5

varnames_plot = ['ismn','surface layer \nsoil moisture','surface temperature',
                 'precipitation (sat)','2m temperature',
                 'precipitation (ground)',
                 'diurnal temperature range sfc','snow cover fraction'] 
varnames_plot = ['$SM_{ISMN}$','$SM_{ERA5-Land}$','$LST_{ERA5-Land}$',
                 '$PSAT_{ERA5-Land}$','$T2M_{ERA5-Land}$',
                 '$P2M_{ERA5-Land}$',
                 '$DTR_{ERA5-Land}$','$SNOW_{ERA5-Land}$'] 
boxplot_kwargs = {'notch': False,
                  'patch_artist': True}
col_fill = 'coral'
col_miss = 'steelblue'

b1 = ax1.boxplot(positions=x_pos, x=corr_orig, showfliers=False, **boxplot_kwargs)
b2 = ax1.boxplot(positions=x_pos+wd, x=corr_fill, showfliers=False, **boxplot_kwargs)

b3 = ax2.boxplot(positions=x_pos, x=rmse_orig, showfliers=False, **boxplot_kwargs)
b4 = ax2.boxplot(positions=x_pos+wd, x=rmse_fill, showfliers=False, **boxplot_kwargs)

for box in b1['boxes'] + b3['boxes']:
    box.set_facecolor(col_miss)
for box in b2['boxes'] + b4['boxes']:
    box.set_facecolor(col_fill)
for median in b1['medians'] + b2['medians'] + b3['medians'] + b4['medians']:
    median.set_color('black')

ax1.set_xticks([])
ax2.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90)
ax1.set_ylim([0,1]) 
ax2.set_ylim([0,1.4]) 
ax1.set_xlim([-1,16])
ax2.set_xlim([-1,16])

ax1.set_title('Pearson correlation coefficient')
ax2.set_title('RSME (normalised)')

ax2.set_xticklabels(varnames_plot)

legend_elements = [Patch(facecolor=col_miss, edgecolor='black', label='With Gaps'),
                   Patch(facecolor=col_fill, edgecolor='black', label='CLIMFILL')]
ax2.legend(handles=legend_elements, loc='upper right')
plt.subplots_adjust(bottom=0.3)
plt.savefig('benchmarking.png')
