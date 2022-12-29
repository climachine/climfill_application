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

def expand_to_worldmap(data,regions):
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
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')
era5 = xr.open_dataset(f'{esapath}data_era5land.nc')

#mask_orig = xr.open_dataset(f'{esapath}mask_orig.nc')
#mask_cv = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc')

# (optional) calculate anomalies
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
era5 = era5.groupby('time.month') - era5.groupby('time.month').mean()
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
            'temperature_obs','precipitation_obs',
            'diurnal_temperature_range','snow_cover_fraction'] 
varnames_plot = ['surface layer \nsoil moisture','surface temperature',
                 'precipitation (sat)','2m temperature',
                 'precipitation (ground)',
                 'diurnal temperature range sfc','snow cover fraction'] 
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

# aggregate per region
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
levels = np.arange(-1,1.25,0.25)

for v, (varname, ax) in enumerate(zip(varnames, axes)):
    corrorig = xr.corr(orig.sel(variable=varname),era5.sel(variable=varname), dim='time')
    corrfill = xr.corr(fill.sel(variable=varname),era5.sel(variable=varname), dim='time')

    corrorig = expand_to_worldmap(corrorig,regions)
    corrfill = expand_to_worldmap(corrfill,regions)

    landmask.plot(ax=ax, add_colorbar=False, cmap='Greys', transform=transf, vmin=-2, vmax=10)
    im = corrfill.plot(ax=ax, cmap='coolwarm_r', vmin=-1, vmax=1, transform=transf, 
                               levels=levels, add_colorbar=False)
    regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
                                                 ax=ax, add_label=False, projection=transf)
    ax.set_title(varnames_plot[v])

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.6]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Pearson correlation')

plt.savefig('benchmarking_maps.png')
