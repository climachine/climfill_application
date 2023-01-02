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

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
verification_year = '2004'

def calc_rmse(dat1, dat2, dim):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

#def aggregate_to_regions(data, regions):
#    landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(data.lon, data.lat)
#    regions = regionmask.defined_regions.ar6.land.mask(data.lon, data.lat)
#    regions = regions.where(~np.isnan(landmask))
#
#    data = data.groupby(regions).mean()
#
#    test = xr.full_like(regions, np.nan)
#    for region, r in zip(range(int(regions.max().item())), data):
#        test = test.where(regions != region, r) # unit stations per bio square km
#
#    return test

def aggregate_to_regions(data, regions):
    test = xr.full_like(regions, np.nan)
    for region, r in zip(range(int(regions.max().item())), data):
        test = test.where(regions != region, r) # unit stations per bio square km
    return test

# NOTE:
# TWS does not have any originally missing values in 2004
# JS not possible bec needs times where all 8 vars are observed (i.e. in the
#      verification set) at the same time. ditched for now

# read data
orig = xr.open_dataset(f'{esapath}data_orig.nc')
#intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc')
#fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/verification/data_climfilled.nc')
intp = xr.open_dataset(f'{esapath}{testcase}/verification/data_interpolated.nc')

mask_orig = xr.open_dataset(f'{esapath}mask_orig.nc')

# (optional) calculate anomalies
#orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
#intp = intp.groupby('time.month') - intp.groupby('time.month').mean()
#fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# select verification year
mask_orig = mask_orig.sel(time=verification_year).load()
orig = orig.sel(time=verification_year).load()
intp = intp.sel(time=verification_year).load()
fill = fill.sel(time=verification_year).load()

# calculate mask of verification points
mask_cv = np.logical_not(mask_orig)

# mask everything except verification points
# for now not; reason: see plot_verification.py
#orig = orig.where(mask_cv)
#intp = intp.where(mask_cv)
#fill = fill.where(mask_cv)

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

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(331, projection=proj)
ax2 = fig.add_subplot(332, projection=proj)
ax3 = fig.add_subplot(333, projection=proj)
ax4 = fig.add_subplot(334, projection=proj)
ax5 = fig.add_subplot(335, projection=proj)
ax6 = fig.add_subplot(336, projection=proj)
ax7 = fig.add_subplot(337, projection=proj)
ax8 = fig.add_subplot(338, projection=proj)
ax9 = fig.add_subplot(339, projection=proj)
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
levels = np.arange(-1,1.1,0.1)
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))
obsmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

for v, (varname, ax) in enumerate(zip(varnames, axes)):
    #corrfill = xr.corr(orig.sel(variable=varname),fill.sel(variable=varname), dim='time')
    #corrintp = xr.corr(orig.sel(variable=varname),intp.sel(variable=varname), dim='time')
    rmseintp = calc_rmse(orig.sel(variable=varname), intp.sel(variable=varname), dim=('time'))
    rmsefill = calc_rmse(orig.sel(variable=varname), fill.sel(variable=varname), dim=('time'))

    # calculate skill score
    skillscore = 1 - (rmsefill/rmseintp)

    # mask regions with less than 10% verification points
    # mask regions with less than 10% points at all (greenland, deserts etc)
    count_landpoints = landmask.where(landmask!=0,1).groupby(regions).count()
    count_obspoints = obsmask.groupby(regions).sum()
    count_verification = rmsefill.notnull().groupby(regions).sum()
    valid_regions = count_verification/count_landpoints > 0.05
    included_regions = count_obspoints/count_landpoints > 0.10

    skillscore = skillscore.groupby(regions).mean()

    skillscore = skillscore.where(valid_regions)
    skillscore = skillscore.where(included_regions)

    skillscore = aggregate_to_regions(skillscore, regions)

    landmask.plot(ax=ax, add_colorbar=False, cmap='Greys', transform=transf, vmin=-2, vmax=10)
    im = skillscore.plot(ax=ax, cmap='coolwarm_r', vmin=-1, vmax=1, transform=transf, 
                  levels=levels, add_colorbar=False)
    regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
                                             ax=ax, add_label=False, projection=transf)
    ax.set_title(varnames_plot[v])

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.6]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('RMSE Skill Score')

plt.savefig('verification_maps.png')
