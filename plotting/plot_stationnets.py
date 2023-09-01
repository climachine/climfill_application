"""
NAMESTRING
"""

import numpy as np
import pandas as pd
import xarray as xr
import regionmask

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
proj = ccrs.PlateCarree()
transf = ccrs.PlateCarree()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask
oceanmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(landmask.lon,landmask.lat)
oceanmask = ~np.isnan(oceanmask)
cmap = plt.get_cmap('twilight_shifted')
cmap.set_under('aliceblue')
cmap.set_over('lightgrey')
s=1
xy=(0.88,0.10)

# ISMN
largefilepath = '/net/so4/landclim/bverena/large_files/'
ismn = xr.open_dataset(f'{largefilepath}df_gaps.nc')['mrso']

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
oceanmask.plot(cmap=cmap, transform=transf, vmin=0.4, vmax=0.5, add_colorbar=False)
ax.scatter(ismn.lon, ismn.lat, transform=transf, c='#B7352D', marker='v', s=s)
ax.annotate('soil moisture, ISMN',xy=xy, xycoords='axes fraction', horizontalalignment='right', backgroundcolor='white')
ax.set_global()
plt.savefig('ismn_map.png', dpi=600)

# GSIM
gsim = pd.read_csv('/net/exo/landclim/data/dataset/GSIM_Station-catalog-and-Catchment-boundary/20180327/basin-scale_none_time-invariant/original/unzipped/GSIM_metadata/GSIM_catalog/GSIM_metadata.csv')

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
oceanmask.plot(cmap=cmap, transform=transf, vmin=0.4, vmax=0.5, add_colorbar=False)
ax.scatter(gsim.longitude, gsim.latitude, transform=transf, c='#B7352D', marker='v', s=s)
ax.annotate('runoff, GSIM',xy=xy, xycoords='axes fraction', horizontalalignment='right', backgroundcolor='white')
ax.set_global()
plt.savefig('gsim_map.png', dpi=600)

# FLUXNET
fluxnet = pd.read_csv('/home/bverena/climfill_esa_cci/fluxnet_sites.csv')

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
oceanmask.plot(cmap=cmap, transform=transf, vmin=0.4, vmax=0.5, add_colorbar=False)
ax.scatter(fluxnet.LOCATION_LONG, fluxnet.LOCATION_LAT, transform=transf, c='#B7352D', marker='v', s=s)
ax.annotate('energy and water fluxes, FLUXNET',xy=xy, xycoords='axes fraction', horizontalalignment='right', backgroundcolor='white')
ax.set_global()
plt.savefig('fluxnet_map.png', dpi=600)

# GHCN
ghcn = pd.read_csv('/net/exo/landclim/data/dataset/GHCN-DAILY/v3.20/point-scale_none_1d/original/ghcnd-inventory.txt', sep='\s+',
                   header=None,
                   names=['name','lat','lon','var','yearstart','yearend'])
ghcn = ghcn.groupby('name').mean()

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111, projection=proj)
oceanmask.plot(cmap=cmap, transform=transf, vmin=0.4, vmax=0.5, add_colorbar=False)
ax.scatter(ghcn.lon, ghcn.lat, transform=transf, c='#B7352D', marker='v', s=s)
ax.annotate('temperature and precipitation, GHCN',xy=xy, xycoords='axes fraction', horizontalalignment='right', backgroundcolor='white')
ax.set_global()
plt.savefig('ghcn_map.png', dpi=600)
