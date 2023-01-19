"""
plot maps of fraction missing on daily resolution
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
data = xr.open_dataset(f'{esapath}{testcase}/data_crossval.nc')
mask_orig = xr.open_dataset(f'{esapath}mask_orig.nc')
mask_cv = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc')
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# calculate mask of verification points
mask = np.logical_and(np.logical_not(mask_orig), mask_cv)

# sel sample slice
data = data.soil_moisture.sel(time='2005-09-01')
mask = mask.soil_moisture.sel(time='2005-09-01')

# calculate ocean mask
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(data.lon,data.lat)
oceanmask = ~np.isnan(oceanmask)

# calculate mask of excluded areas 
icedesert = np.logical_and(np.logical_not(landmask),oceanmask)

data = data.where(oceanmask, -10) # ocean negative
data = data.where(np.logical_not(mask), 10) # verification points set to 10
#data = data.where(icedesert, np.nan) # ice and deserts nan 

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('seismic_r')
cmap.set_under('aliceblue')
cmap.set_over('grey')
levels = np.arange(0,0.45,0.05)
cbar_kwargs = {'label': '$m^{3}m^{-3}$'}

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
data.plot(ax=ax, cmap=cmap, transform=transf, vmin=0, vmax=0.4, levels=levels,
          cbar_kwargs=cbar_kwargs)
#icedesert.plot(ax=ax, hatch='\\', add_colorbar=False) #https://github.com/matplotlib/matplotlib/issues/3058
ax.set_facecolor('lightgrey')
ax.set_title('ESA CCI surface layer soil moisture, Sept 2005')
legend_elements = [Patch(facecolor='grey', edgecolor='grey', 
                   label='validation points'),
                   Patch(facecolor='lightgrey', edgecolor='lightgrey',
                   label='unobserved points')]
ax.legend(handles=legend_elements, loc='upper right',
          bbox_to_anchor=(1,0))
plt.savefig('minicubes.png', dpi=300)
