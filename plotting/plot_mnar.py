"""
NAMESTRING
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import regionmask

largefilepath = '/net/so4/landclim/bverena/large_files/'

# open dataxr, dataxr_lost
varnames = ['tp', 'swvl1', 'skt','tws']
real = xr.open_dataarray(largefilepath + f'real_small/datacube_lost_noip_idebug_True.nc')
random = xr.open_dataarray(largefilepath + f'random_20/datacube_lost_noip_idebug_True.nc')
swaths = xr.open_dataarray(largefilepath + f'swaths_20/datacube_lost_noip_idebug_True.nc')

# select one month
real = real.loc['skt'].sel(time='2003-08-15')
random = random.loc['skt'].sel(time='2003-08-15')
swaths = swaths.loc['skt'].sel(time='2003-08-15')

# select south america
real =   real.sel(latitude=slice(-56,14), longitude=slice  (-99,-23))
random = random.sel(latitude=slice(-56,14), longitude=slice(-99,-23))
swaths = swaths.sel(latitude=slice(-56,14), longitude=slice(-99,-23))

# color ocean blue
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(real.longitude,real.latitude)
oceanmask = ~np.isnan(oceanmask)

real = real.where(oceanmask, -100) # ocean negative
random = random.where(oceanmask, -100) # ocean negative
swaths = swaths.where(oceanmask, -100) # ocean negative

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('twilight_shifted_r')
cmap.set_under('aliceblue')
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
real.plot(transform=transf, cmap=cmap, vmin= 250, vmax=320,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('real_missingness.pdf')
plt.close()

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('twilight_shifted_r')
cmap.set_under('aliceblue')
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
random.plot(transform=transf, cmap=cmap, vmin= 250, vmax=320,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('random_missingness.pdf')
plt.close()

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('twilight_shifted_r')
cmap.set_under('aliceblue')
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
swaths.plot(transform=transf, cmap=cmap, vmin= 250, vmax=320,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('swaths_missingness.pdf')
plt.close()
