"""
plot ismn station locations and cmip6 observed and unobserved grid points

"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

largefilepath = '/net/so4/landclim/bverena/large_files/'

obsmask = xr.open_dataarray(f'{largefilepath}opscaling/obsmask.nc').squeeze()
smmask = xr.open_dataarray(f'{largefilepath}opscaling/landmask.nc').squeeze()
mask = np.logical_and(smmask, np.logical_not(obsmask))
stations = xr.open_dataset(f'{largefilepath}df_gaps.nc')['mrso']

inactive_networks = ['HOBE','PBO_H20','IMA_CAN1','SWEX_POLAND','CAMPANIA',
                     'HiWATER_EHWSN', 'METEROBS', 'UDC_SMOS', 'KHOREZM',
                     'ICN','AACES','HSC_CEOLMACHEON','MONGOLIA','RUSWET-AGRO',
                     'CHINA','IOWA','RUSWET-VALDAI','RUSWET-GRASS']
stations = stations.resample(time='Y').mean()
stations = stations.where(~stations.network.isin(inactive_networks), drop=True) # takes too long

# make nan
obsmask = obsmask.astype(int)
obsmask = obsmask.where(obsmask != 0, np.nan)

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection=proj)
obsmask.plot(ax=ax, cmap='Purples', vmin=0, vmax=2, transform=transf, add_colorbar=False)
ax.scatter(stations.lon, stations.lat, transform=transf, c='black', marker='v', s=2) 
ax.coastlines()
plt.savefig('obsmask.png', dpi=300, bbox_inches='tight')
plt.close()

# make nan
mask = mask.astype(int)
mask = mask.where(mask != 0, np.nan)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection=proj)
mask.plot(ax=ax, cmap='Reds', vmin=0, vmax=2, transform=transf, add_colorbar=False)
obsmask.plot(ax=ax, cmap='Purples', vmin=0, vmax=2, transform=transf, add_colorbar=False)
ax.scatter(stations.lon, stations.lat, transform=transf, c='black', marker='v', s=2) 
ax.coastlines()
plt.savefig('smcoupmask.png', dpi=300, bbox_inches='tight')
plt.close()

# random strategy
random = xr.open_dataarray('/home/bverena/optimal_station_network/niter_random_UKESM1-0-LL_corr_smmask2.nc')
geogr = xr.open_dataarray('/home/bverena/optimal_station_network/niter_interp_UKESM1-0-LL_corr_smmask2.nc')
systematic = xr.open_dataarray('/home/bverena/optimal_station_network/niter_systematic_UKESM1-0-LL_corr_smmask2.nc')

for i in range(10):
    obsmask = obsmask.where(systematic != i, 1)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection=proj)
    mask.plot(ax=ax, cmap='Reds', vmin=0, vmax=2, transform=transf, add_colorbar=False)
    obsmask.plot(ax=ax, cmap='Purples', vmin=0, vmax=2, transform=transf, add_colorbar=False)
    #(random == i).plot(ax=ax, cmap='Greys', vmin=0, vmax=2, transform=transf, add_colorbar=False)
    ax.scatter(stations.lon, stations.lat, transform=transf, c='black', marker='v', s=2) 
    ax.coastlines()
    ax.set_title('')
    plt.savefig(f'obsmask_sys_{i}.png', dpi=300, bbox_inches='tight')
    plt.close()
