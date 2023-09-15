"""
NAMESTRING
"""

import numpy as np
import xarray as xr
import regionmask
import xesmf as xe

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
varnames = ['soil_moisture','surface_temperature','precipitation', 
            'terrestrial_water_storage','snow_cover_fraction',
            'temperature_obs','precipitation_obs','burned_area',
            'diurnal_temperature_range'] #hardcoded for now

landclimstoragepath = '/net/exo/landclim/data/dataset/'
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.75,90, 0.5)), # same as cdo_weights.nc
                     'lon': (['lon'], np.arange(-180, 180,0.5))})

# open data
data = xr.open_dataset(f'{esapath}data_orig.nc').to_array()
#sm = xr.open_dataset(landclimstoragepath + \
#    'ESA-CCI-SM_combined/v07.1/0.25deg_lat-lon_1d/processed/netcdf/' + \
#    'ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-2020.nc')['sm']
#lst = xr.open_dataset('modis_surface_temperature_20200816.nc').load().to_array()
##lst = xr.open_dataset(landclimstoragepath + \
##    'MODIS_Land-Surface-Temperature-MYD11C1/v006/0.05deg_lat-lon_1d/original/' + \
##    'MYD11C1.A2020229.006.2020336150731.hdf')['LST_Day_CMG']
##ydim = lst.dims[0]
##xdim = lst.dims[1]
##lst = lst.assign_coords(**{ydim: np.arange(90,-90,-0.05)})
##lst = lst.assign_coords(**{xdim: np.arange(-180,180,0.05)})
##lst = lst.rename({ydim:'lat',xdim:'lon'})
##lst = lst.load()
##lst = lst - 273.15
#    precip = xr.open_dataset(landclimstoragepath + \
#    'GPM_IMERG_L3/v06/0.1deg_lat-lon_1m/original/'
#    data = xr.open_mfdataset(f'{filepath}*.nc4')['precipitation']
#
#    # convert DatetimeJulian to datetimeindex
#    data['time'] = data.indexes['time'].to_datetimeindex()
#
#    data = data.sel(time=timeslice)
#
#    # convert from [mm/hour] to [mm/month]
#    data = data*24*30.5
#
#    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
#    data = regridder(data)

# regrid
#regridder = xe.Regridder(sm, ds_out, 'bilinear', reuse_weights=False) 
#sm = regridder(sm)
#regridder = xe.Regridder(lst, ds_out, 'bilinear', reuse_weights=False) 
#lst = regridder(lst)

# masks
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(data.lon,data.lat)
oceanmask = ~np.isnan(oceanmask)
#icedesert = np.logical_and(np.logical_not(landmask),oceanmask)

data = data.where(oceanmask, -100) # ocean negative
#lst = lst.where(oceanmask, -100) # ocean negative
#sm = sm.where(oceanmask, -100) # ocean negative
#lst = lst.where(oceanmask, -100) # ocean negative
#mask = mask.where(np.logical_not(icedesert), np.nan) # ice and deserts nan 

data = data.sel(time='2018-07-01')

lst = data.sel(variable='surface_temperature')
sm = data.sel(variable='soil_moisture')
pre = data.sel(variable='precipitation')
tws = data.sel(variable='terrestrial_water_storage')
dtr = data.sel(variable='diurnal_temperature_range')
ba = data.sel(variable='burned_area')
scf = data.sel(variable='snow_cover_fraction')

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
cmap = plt.get_cmap('twilight_shifted_r')
cmap.set_under('aliceblue')
#cmap.set_bad('lightgrey')
levels = np.arange(10)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax)
ax.coastlines()
plt.savefig('sm.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
sm = data.sel(variable='soil_moisture')
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax, alpha=0.2)
sm = sm.where(np.logical_not(np.isnan(lst)))
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('sm_lst.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
sm = data.sel(variable='soil_moisture')
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax, alpha=0.2)
sm = sm.where(np.logical_not(np.isnan(lst)))
sm = sm.where(np.logical_not(np.isnan(pre)))
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('sm_lst_pre.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
sm = data.sel(variable='soil_moisture')
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax, alpha=0.2)
sm = sm.where(np.logical_not(np.isnan(lst)))
sm = sm.where(np.logical_not(np.isnan(pre)))
sm = sm.where(np.logical_not(np.isnan(tws)))
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('sm_lst_pre_tws.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
sm = data.sel(variable='soil_moisture')
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax, alpha=0.2)
sm = sm.where(np.logical_not(np.isnan(lst)))
sm = sm.where(np.logical_not(np.isnan(pre)))
sm = sm.where(np.logical_not(np.isnan(tws)))
sm = sm.where(np.logical_not(np.isnan(dtr)))
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('sm_lst_pre_tws_dtr.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
sm = data.sel(variable='soil_moisture')
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax, alpha=0.2)
sm = sm.where(np.logical_not(np.isnan(lst)))
sm = sm.where(np.logical_not(np.isnan(pre)))
sm = sm.where(np.logical_not(np.isnan(tws)))
sm = sm.where(np.logical_not(np.isnan(dtr)))
sm = sm.where(np.logical_not(np.isnan(ba)))
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('sm_lst_pre_tws_dtr_ba.pdf')
plt.close()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
sm = data.sel(variable='soil_moisture')
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax, alpha=0.2)
sm = sm.where(np.logical_not(np.isnan(lst)))
sm = sm.where(np.logical_not(np.isnan(pre)))
sm = sm.where(np.logical_not(np.isnan(tws)))
sm = sm.where(np.logical_not(np.isnan(dtr)))
sm = sm.where(np.logical_not(np.isnan(ba)))
sm = sm.where(np.logical_not(np.isnan(scf)))
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
plt.savefig('sm_lst_pre_tws_dtr_ba_scf.pdf')
plt.close()




#lst.plot(transform=transf, cmap=cmap, vmin=-50, vmax=50, 
#                               add_colorbar=False, ax=ax, alpha=0.2)
#lst = lst.where(np.logical_not(np.isnan(sm)))
#lst.plot(transform=transf, cmap=cmap, vmin=-50, vmax=50, 
#                               add_colorbar=False, ax=ax)
#pre.plot(transform=transf, cmap=cmap, vmin=-50, vmax=50, 
#                               add_colorbar=False, ax=ax, alpha=0.2)
#pre = pre.where(np.logical_not(np.isnan(sm)))
#pre = pre.where(np.logical_not(np.isnan(lst)))
#pre.plot(transform=transf, cmap=cmap, vmin=0, vmax=500, 
#                               add_colorbar=False, ax=ax)

data = xr.open_dataset(f'{esapath}data_orig.nc').to_array()
fill = xr.open_dataset(f'{esapath}test8/data_climfilled.nc').to_array()
fill = fill.sel(time='2020-03-01')
data = data.sel(time='2020-03-01')
data = data.where(oceanmask, -100) # ocean negative
fill = fill.where(oceanmask, -100) # ocean negative

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
sm = data.sel(variable='soil_moisture')
sm_fill = fill.sel(variable='soil_moisture')
sm_fill.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax, alpha=0.3)
sm.plot(transform=transf, cmap=cmap, vmin=0, vmax=0.5, 
                               add_colorbar=False, ax=ax)
ax.set_title('')
ax.coastlines()
#plt.savefig('title_image.png', dpi=1000)
plt.savefig('title_image.pdf')
plt.close()
