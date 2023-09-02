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
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
verification_year = '2004'

def calc_rmse(dat1, dat2, dim):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

def expand_to_worldmap(data,regions):
    test = xr.full_like(regions, np.nan)
    for region, r in zip(range(int(regions.max().item())), data):
        test = test.where(regions != region, r) # unit stations per bio square km

    return test

# NOTE:
# TWS does not have any originally missing values in 2004
# JS not possible bec needs times where all 8 vars are observed (i.e. in the
#      verification set) at the same time. ditched for now

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
orig = xr.open_dataset(f'{esapath}data_orig.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
era5 = xr.open_dataset(f'{esapath}data_era5land.nc')

# (optional) calculate anomalies
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
era5 = era5.groupby('time.month') - era5.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# sort data
varnames = ['soil_moisture','surface_temperature','precipitation',
            'temperature_obs','precipitation_obs','snow_cover_fraction',
            'diurnal_temperature_range'] 
#varnames_plot = ['surface layer \nsoil moisture','surface temperature',
#                 'precipitation (sat)','2m temperature',
#                 'precipitation (ground)',
#                 'diurnal temperature range sfc','snow cover fraction'] 
orig = orig.to_array().reindex(variable=varnames)
era5 = era5.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)

# aggregate to regions
# needs to be before corr bec otherwise all nans are ignored in orig
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))
obsmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

orig = orig.groupby(regions).mean()
era5 = era5.groupby(regions).mean()
fill = fill.groupby(regions).mean()

# normalise values for RMSE plotting
import IPython; IPython.embed() #TODO: debug RMSE values
datamean = orig.mean()
datastd = orig.std()
orig = (orig - datamean) / datastd
era5 = (era5 - datamean) / datastd
fill = (fill - datamean) / datastd

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(20,8), layout='constrained')
gs0 = fig.add_gridspec(1,2)

gs00 = gs0[0].subgridspec(3,3)
gs01 = gs0[1].subgridspec(2,2, width_ratios=[0.2,1])

ax1 = fig.add_subplot(gs00[0,0], projection=proj)
ax2 = fig.add_subplot(gs00[0,1], projection=proj)
ax3 = fig.add_subplot(gs00[0,2], projection=proj)
ax4 = fig.add_subplot(gs00[1,0], projection=proj)
ax5 = fig.add_subplot(gs00[1,1], projection=proj)
ax6 = fig.add_subplot(gs00[1,2], projection=proj)
ax7 = fig.add_subplot(gs00[2,0], projection=proj)
ax8 = fig.add_subplot(gs00[2,1], projection=proj)
ax9 = fig.add_subplot(gs00[2,2], projection=proj)

ax10 = fig.add_subplot(gs01[0,1])
ax11 = fig.add_subplot(gs01[1,1])

levels = np.arange(-1,1.1,0.1)
levels = np.arange(-1.05,1.15,0.1) # colorscale go through white
fs = 15

cmap = plt.get_cmap('seismic_r')
cmap.set_over('aliceblue')
cmap.set_bad('lightgrey')

varnames_plot = ['SM','LST','PSAT','TWS','T2M','P2M','SCF','DTR','BA']
axes = [ax1,ax2,ax3,ax5,ax6,ax7,ax8]
for v, (varname, ax) in enumerate(zip(varnames, axes)):
    corrorig = xr.corr(orig.sel(variable=varname),era5.sel(variable=varname), dim='time')
    corrfill = xr.corr(fill.sel(variable=varname),era5.sel(variable=varname), dim='time')

    # mask regions with less than 10% values from landmask (greenland etc)
    count_landpoints = landmask.where(landmask!=0,1).groupby(regions).count()
    count_obspoints = obsmask.groupby(regions).sum()
    included_regions = count_obspoints/count_landpoints > 0.10

    corrfill = corrfill.where(included_regions)
    corrorig = corrorig.where(included_regions)

    # back to worldmap
    corrorig = expand_to_worldmap(corrorig,regions)
    corrfill = expand_to_worldmap(corrfill,regions)

    # ocean blue
    corrfill = corrfill.where(np.logical_not(landmask), 10) # ocean blue
    corrorig = corrorig.where(np.logical_not(landmask), 10) # ocean blue

    #landmask.plot(ax=ax, add_colorbar=False, cmap='Greys', transform=transf, vmin=-2, vmax=10)
    im = corrfill.plot(ax=ax, cmap=cmap, vmin=-1, vmax=1, transform=transf, 
                               levels=levels, add_colorbar=False)
    regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
                                                 ax=ax, add_label=False, projection=transf)

axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
letters = ['a','b','c','d','e','f','g','h','i']
for ax, letter, varname in zip(axes, letters, varnames_plot):
    ax.set_title(f'{letter}) {varname}', fontsize=fs)


cbar_ax = fig.add_axes([0.53, 0.15, 0.02, 0.6]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Pearson correlation coefficient', fontsize=fs)
fig.suptitle('(b) Pearson correlation coefficient on anomalies', fontsize=20)

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

# plot
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

b1 = ax10.boxplot(positions=x_pos, x=corr_fill, showfliers=False, **boxplot_kwargs)
b2 = ax10.boxplot(positions=x_pos+wd, x=corr_orig, showfliers=False, **boxplot_kwargs)

b3 = ax11.boxplot(positions=x_pos, x=rmse_fill, showfliers=False, **boxplot_kwargs)
b4 = ax11.boxplot(positions=x_pos+wd, x=rmse_orig, showfliers=False, **boxplot_kwargs)

for box in b1['boxes'] + b3['boxes']:
    box.set_facecolor(col_fill)
for box in b2['boxes'] + b4['boxes']:
    box.set_facecolor(col_miss)
for median in b1['medians'] + b2['medians'] + b3['medians'] + b4['medians']:
    median.set_color('black')

#ax10.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90)
ax11.set_xticks(x_pos+0.5*wd, varnames_plot, rotation=90)
ax10.set_ylim([-0.1,1]) 
ax11.set_ylim([-0.1,1.4]) 
ax10.set_xlim([-1,14])
ax11.set_xlim([-1,14])

ax10.set_ylabel('Pearson correlation coefficient', fontsize=fs)
ax11.set_ylabel('RMSE on normalized values', fontsize=fs)
fig.suptitle('Benchmarking scores', fontsize=20)

#ax10.set_xticklabels(varnames_plot, fontsize=fs)
ax11.set_xticklabels(varnames_plot, fontsize=fs)

ax10.set_title('j) median of global pearson correlation coefficient')
ax11.set_title('k) median of global RMSE')

legend_elements = [Patch(facecolor=col_fill, edgecolor='black', label='CLIMFILL'),
                   Patch(facecolor=col_miss, edgecolor='black', label='With Gaps')]
ax11.legend(handles=legend_elements, loc='upper right', fontsize=fs)
plt.savefig('benchmarking_maps.jpeg',dpi=300)
