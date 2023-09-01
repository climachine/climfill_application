"""
NAMESTRING
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import regionmask


# open dataxr, dataxr_lost
largefilepath = '/net/so4/landclim/bverena/large_files/'
#savepath = '/net/so4/landclim/bverena/large_files/2003_2020/'
savepath = '/net/so4/landclim/bverena/large_files/real_small/'
orig = xr.open_dataarray(savepath + f'datacube_noip_idebug_True.nc')
lost = xr.open_dataarray(savepath + f'datacube_lost_noip_idebug_True.nc')
intp = xr.open_dataarray(savepath + f'datacube_interp_noip_idebug_True.nc')
fill = xr.open_dataarray(savepath + f'data_climfill_150_idebug_True.nc')
#orig = xr.open_dataarray(savepath + f'datacube_original_2003_2009.nc')
#lost = xr.open_dataarray(savepath + f'datacube_lost_2003_2009.nc')
#intp = xr.open_dataarray(savepath + f'datacube_interp_2003_2009.nc')
#fill = xr.open_dataarray(savepath + f'data_climfilled_2003.nc')
#random = xr.open_dataarray(largefilepath + f'random_20/datacube_lost_noip_idebug_True.nc')
#random = xr.open_dataarray(largefilepath + f'random_20/datacube_interp_noip_idebug_True.nc')
random = xr.open_dataarray(largefilepath + f'random_20/data_climfill_150_idebug_True.nc')
#swaths = xr.open_dataarray(largefilepath + f'swaths_20/datacube_lost_noip_idebug_True.nc')
#swaths = xr.open_dataarray(largefilepath + f'swaths_20/datacube_interp_noip_idebug_True.nc')
swaths = xr.open_dataarray(largefilepath + f'swaths_20/data_climfill_150_idebug_True.nc')

# select one month
orig = orig.loc['skt'].sel(time='2003-08-15')
lost = lost.loc['skt'].sel(time='2003-08-15')
intp = intp.loc['skt'].sel(time='2003-08-15')
fill = fill.loc['skt'].sel(time='2003-08-15')
random = random.loc['skt'].sel(time='2003-08-15')
swaths = swaths.loc['skt'].sel(time='2003-08-15')

# select south africa
orig = orig.sel(latitude=slice(-35,0), longitude=slice(8,52))
lost = lost.sel(latitude=slice(-35,0), longitude=slice(8,52))
intp = intp.sel(latitude=slice(-35,0), longitude=slice(8,52))
fill = fill.sel(latitude=slice(-35,0), longitude=slice(8,52))
random = random.sel(latitude=slice(-35,0), longitude=slice(8,52))
swaths = swaths.sel(latitude=slice(-35,0), longitude=slice(8,52))

# color ocean blue
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.longitude,orig.latitude)
oceanmask = ~np.isnan(oceanmask)

orig = orig.where(oceanmask, -100) # ocean negative
lost = lost.where(oceanmask, -100) # ocean negative
intp = intp.where(oceanmask, -100) # ocean negative
fill = fill.where(oceanmask, -100) # ocean negative
random = random.where(oceanmask, -100) # ocean negative
swaths = swaths.where(oceanmask, -100) # ocean negative

intp = intp.where(np.logical_not(np.isnan(fill))) # remove lakes

# plot
vmin = 275
vmax = 300
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('twilight_shifted_r')
cmap.set_under('aliceblue')
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
orig.plot(transform=transf, cmap=cmap, vmin= vmin, vmax=vmax,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('orig.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
lost.plot(transform=transf, cmap=cmap, vmin= vmin, vmax=vmax,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('lost.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
intp.plot(transform=transf, cmap=cmap, vmin= vmin, vmax=vmax,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('intp.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
fill.plot(transform=transf, cmap=cmap, vmin= vmin, vmax=vmax,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('fill.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
random.plot(transform=transf, cmap=cmap, vmin= vmin, vmax=vmax,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('random.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
swaths.plot(transform=transf, cmap=cmap, vmin= vmin, vmax=vmax,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('swaths.pdf')
plt.close()

# climfilled paper3
savepath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
lost = xr.open_dataset(savepath + f'data_orig.nc').to_array()
intp = xr.open_dataset(savepath + f'test8/data_initguess.nc').to_array()
fill = xr.open_dataset(savepath + f'test8/data_climfilled.nc').to_array()
era5 = xr.open_dataset(savepath + f'data_era5land.nc').to_array()

# select one month
lost = lost.sel(variable='surface_temperature').sel(time='2017-11')
intp = intp.sel(variable='surface_temperature').sel(time='2017-11')
fill = fill.sel(variable='surface_temperature').sel(time='2017-11')
era5 = era5.sel(variable='surface_temperature').sel(time='2017-11')

# select south africa
lost = lost.sel(lat=slice(-35,0), lon=slice(8,52))
intp = intp.sel(lat=slice(-35,0), lon=slice(8,52))
fill = fill.sel(lat=slice(-35,0), lon=slice(8,52))
era5 = era5.sel(lat=slice(-35,0), lon=slice(8,52))

# color ocean blue
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(lost.lon,lost.lat)
oceanmask = ~np.isnan(oceanmask)

lost = lost.where(oceanmask, -100) # ocean negative
intp = intp.where(oceanmask, -100) # ocean negative
fill = fill.where(oceanmask, -100) # ocean negative
era5 = era5.where(oceanmask, -100) # ocean negative

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
lost.plot(transform=transf, cmap=cmap, vmin= 15, vmax=55,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('lost_realworld.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
intp.plot(transform=transf, cmap=cmap, vmin= 15, vmax=55,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('intp_realworld.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
fill.plot(transform=transf, cmap=cmap, vmin= 15, vmax=55,
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('fill_realworld.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
era5.plot(transform=transf, cmap=cmap, vmin=10, vmax=35,
                               add_colorbar=True, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('era5_realworld.pdf')
plt.close()
