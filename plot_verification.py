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

# TODO 
# mask is the same for two subsequent months???
# use trained Regr from full timeline?

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
verification_year = slice('2004','2005')

varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','temperature_obs','precipitation_obs',
            'snow_cover_fraction','diurnal_temperature_range','burned_area'] 
#varnames_plot = ['surface layer \nsoil moisture','surface temperature',
#                 'precipitation (sat)','terrestrial water storage','2m temperature',
#                 'precipitation (ground)', 'snow cover fraction',
#                 'diurnal temperature range sfc','burned area'] 
varnames_plot = ['SM','LST','PSAT','TWS','T2M','P2M','SCF','DTR','BA']

def calc_rmse(dat1, dat2, dim):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

# read data
orig = xr.open_dataset(f'{esapath}data_orig.nc')
intp = xr.open_mfdataset(f'{esapath}{testcase}/verification/set?/data_interpolated_del.nc')
fill = xr.open_mfdataset(f'{esapath}{testcase}/verification/set?/data_climfilled_del.nc')

# average over all set
intp_set = intp.mean(dim='veriset').load()
fill_set = fill.mean(dim='veriset').load()

# normalise values for RMSE plotting
datamean = orig.mean()
datastd = orig.std()
orig = (orig - datamean) / datastd
intp = (intp - datamean) / datastd
fill = (fill - datamean) / datastd

# select verification year
orig = orig.sel(time=verification_year).load()

# sort data
orig = orig.to_array().reindex(variable=varnames)
intp = intp.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)

# (optional) calculate anomalies
orig_seas = orig.groupby('time.month').mean()
intp_seas = intp.groupby('time.month').mean()
fill_seas = fill.groupby('time.month').mean()

orig_anom = orig.groupby('time.month') - orig.groupby('time.month').mean()
intp_anom = intp.groupby('time.month') - intp.groupby('time.month').mean()
fill_anom = fill.groupby('time.month') - fill.groupby('time.month').mean()

# calc metrics
corr_intp = xr.corr(orig, intp, dim=('time'))
corr_fill = xr.corr(orig, fill, dim=('time'))
rmse_intp = calc_rmse(orig, intp, dim=('time'))
rmse_fill = calc_rmse(orig, fill, dim=('time'))

corr_intp_anom = xr.corr(orig_anom, intp_anom, dim=('time'))
corr_fill_anom = xr.corr(orig_anom, fill_anom, dim=('time'))
rmse_intp_anom = calc_rmse(orig_anom, intp_anom, dim=('time'))
rmse_fill_anom = calc_rmse(orig_anom, fill_anom, dim=('time'))

corr_intp_seas = xr.corr(orig_seas, intp_seas, dim=('month'))
corr_fill_seas = xr.corr(orig_seas, fill_seas, dim=('month'))
rmse_intp_seas = calc_rmse(orig_seas, intp_seas, dim=('month'))
rmse_fill_seas = calc_rmse(orig_seas, fill_seas, dim=('month'))

# remove from corr all timepoints with less than 3 missing vals (bec will be corr 1)
corrmask = (np.logical_not(np.isnan(fill)).sum(dim='time') > 3).reindex(variable=varnames)
corr_intp = corr_intp.where(corrmask)
corr_fill = corr_fill.where(corrmask)

corr_intp_anom = corr_intp_anom.where(corrmask)
corr_fill_anom = corr_fill_anom.where(corrmask)

corr_intp_seas = corr_intp_seas.where(corrmask)
corr_fill_seas = corr_fill_seas.where(corrmask)

# aggregate lat lon for boxplot
corr_intp = corr_intp.stack(landpoints=('lat','lon'))
corr_fill = corr_fill.stack(landpoints=('lat','lon'))
rmse_intp = rmse_intp.stack(landpoints=('lat','lon'))
rmse_fill = rmse_fill.stack(landpoints=('lat','lon'))

corr_intp_anom = corr_intp_anom.stack(landpoints=('lat','lon'))
corr_fill_anom = corr_fill_anom.stack(landpoints=('lat','lon'))
rmse_intp_anom = rmse_intp_anom.stack(landpoints=('lat','lon'))
rmse_fill_anom = rmse_fill_anom.stack(landpoints=('lat','lon'))

corr_intp_seas = corr_intp_seas.stack(landpoints=('lat','lon'))
corr_fill_seas = corr_fill_seas.stack(landpoints=('lat','lon'))
rmse_intp_seas = rmse_intp_seas.stack(landpoints=('lat','lon'))
rmse_fill_seas = rmse_fill_seas.stack(landpoints=('lat','lon'))

# remove nan for boxplot
def filter_data(data):
    mask = ~np.isnan(data)
    filtered_data = [d[m] for d, m in zip(data.values, mask.values)]
    return filtered_data
corr_intp = filter_data(corr_intp)
corr_fill = filter_data(corr_fill)
rmse_intp = filter_data(rmse_intp)
rmse_fill = filter_data(rmse_fill)

corr_intp_anom = filter_data(corr_intp_anom)
corr_fill_anom = filter_data(corr_fill_anom)
rmse_intp_anom = filter_data(rmse_intp_anom)
rmse_fill_anom = filter_data(rmse_fill_anom)

corr_intp_seas = filter_data(corr_intp_seas)
corr_fill_seas = filter_data(corr_fill_seas)
rmse_intp_seas = filter_data(rmse_intp_seas)
rmse_fill_seas = filter_data(rmse_fill_seas)

# plot
fig = plt.figure(figsize=(25,10))
ax1 = fig.add_subplot(231)
ax2 = fig.add_subplot(232)
ax3 = fig.add_subplot(233)
ax4 = fig.add_subplot(234)
ax5 = fig.add_subplot(235)
ax6 = fig.add_subplot(236)
fs = 15

x_pos =np.arange(0,2*len(corr_intp),2)
wd = 0.5

boxplot_kwargs = {'notch': False,
                  'patch_artist': True}
col_intp = 'gold'
col_fill = 'coral'

b1 = ax1.boxplot(positions=x_pos, x=corr_intp, showfliers=False, **boxplot_kwargs)
b2 = ax1.boxplot(positions=x_pos+wd, x=corr_fill, showfliers=False, **boxplot_kwargs)

b3 = ax2.boxplot(positions=x_pos, x=corr_intp_anom, showfliers=False, **boxplot_kwargs)
b4 = ax2.boxplot(positions=x_pos+wd, x=corr_fill_anom, showfliers=False, **boxplot_kwargs)
import IPython; IPython.embed()

b5 = ax3.boxplot(positions=x_pos, x=corr_intp_seas, showfliers=False, **boxplot_kwargs)
b6 = ax3.boxplot(positions=x_pos+wd, x=corr_fill_seas, showfliers=False, **boxplot_kwargs)

b7 = ax4.boxplot(positions=x_pos, x=rmse_intp, showfliers=False, **boxplot_kwargs)
b8 = ax4.boxplot(positions=x_pos+wd, x=rmse_fill, showfliers=False, **boxplot_kwargs)

b9 = ax5.boxplot(positions=x_pos, x=rmse_intp_anom, showfliers=False, **boxplot_kwargs)
b10 = ax5.boxplot(positions=x_pos+wd, x=rmse_fill_anom, showfliers=False, **boxplot_kwargs)

b11 = ax6.boxplot(positions=x_pos, x=rmse_intp_seas, showfliers=False, **boxplot_kwargs)
b12 = ax6.boxplot(positions=x_pos+wd, x=rmse_fill_seas, showfliers=False, **boxplot_kwargs)

for box in b1['boxes'] + b3['boxes'] + b5['boxes']+ b7['boxes']+ b9['boxes']+ b11['boxes']:
    box.set_facecolor(col_intp)
for box in b2['boxes'] + b4['boxes'] + b6['boxes']+ b8['boxes']+ b10['boxes']+ b12['boxes']:
    box.set_facecolor(col_fill)
for median in b1['medians'] + b2['medians'] + b3['medians'] + b4['medians'] +\
    b5['medians'] + b6['medians'] + b7['medians'] + b8['medians'] +\
    b9['medians'] + b10['medians'] + b11['medians'] + b12['medians']:
    median.set_color('black')

ax1.set_xticks([])
ax2.set_xticks([])
ax3.set_xticks([])

legend_elements = [Patch(facecolor=col_intp, edgecolor='black', label='Interpolation'),
                   Patch(facecolor=col_fill, edgecolor='black', label='CLIMFILL')]

ax6.legend(handles=legend_elements, loc='upper right', fontsize=fs)

ax4.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90, fontsize=fs)
ax5.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90, fontsize=fs)
ax6.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90, fontsize=fs)

ax1.set_ylim([-0.5,1.1]) 
ax2.set_ylim([-0.5,1.1]) 
ax3.set_ylim([-0.5,1.1]) 

ax4.set_ylim([-0.0,0.9]) 
ax5.set_ylim([-0.0,0.9]) 
ax6.set_ylim([-0.0,0.9]) 

ax1.set_xlim([-1,18])
ax2.set_xlim([-1,18])

#ax1.set_title('Pearson correlation coefficient')
#ax2.set_title('Pearson correlation coefficient')
#ax3.set_title('Pearson correlation coefficient')
#ax4.set_title('RSME (normalized)')
#ax5.set_title('RSME (normalized)')
#ax6.set_title('RSME (normalized)')
ax1.set_title('Time series', fontsize=fs)
ax2.set_title('Anomalies of time series', fontsize=fs)
ax3.set_title('Seasonal cycle', fontsize=fs)

ax1.set_ylabel('Pearson correlation coefficient', fontsize=fs)
ax4.set_ylabel('RMSE on normalized values', fontsize=fs)
fig.suptitle('(a) Verification scores', fontsize=20)

#plt.subplots_adjust(bottom=0.3)
plt.savefig('verification.png')
