"""
plot maps of fraction missing on daily resolution
"""

import numpy as np
import xarray as xr
import regionmask
import argparse

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

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

# sel sample slice
data = data.soil_moisture.sel(time='2005-09-01')

# set ocean to negative
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(data.lon,data.lat)
oceanmask = ~np.isnan(oceanmask)
data = data.where(oceanmask, -10)

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('coolwarm_r')
cmap.set_under('aliceblue')
levels = np.arange(0,0.45,0.05)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
data.plot(ax=ax, cmap=cmap, transform=transf, vmin=0, vmax=0.4, levels=levels)
ax.set_facecolor('lightgrey')
ax.set_title('ESA CCI surface layer soil moisture, Sept 2005')
plt.savefig('minicubes.png', dpi=300)
