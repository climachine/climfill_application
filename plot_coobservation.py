"""
NAMESTRING
"""

import numpy as np
import xarray as xr
import regionmask
import argparse

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase


esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
varnames = ['soil_moisture','surface_temperature','precipitation', 
            'terrestrial_water_storage','snow_cover_fraction',
            'temperature_obs','precipitation_obs','burned_area',
            'diurnal_temperature_range'] #hardcoded for now

# open data
mask = xr.open_dataset(f'{esapath}mask_orig.nc').to_array()
mask = mask.sum(dim='variable').mean(dim='time')

# masks
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(mask.lon,mask.lat)
oceanmask = ~np.isnan(oceanmask)
icedesert = np.logical_and(np.logical_not(landmask),oceanmask)

mask = mask.where(oceanmask, -10) # ocean negative
mask = mask.where(np.logical_not(icedesert), np.nan) # ice and deserts nan 

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('YlOrBr')
cmap.set_under('aliceblue')
cmap.set_bad('grey')
levels = np.arange(10)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
mask.plot(cmap=cmap, levels=levels, transform=transf)
ax.set_title('Average number of available co-occurring variables')
plt.savefig('coobs.png', dpi=300)
