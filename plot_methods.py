"""
NAMESTRING
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

largefilepath = '/net/so4/landclim/bverena/large_files/'
stations = xr.open_dataset(f'{largefilepath}df_gaps.nc')['mrso']
landmask = xr.open_dataarray(f'{largefilepath}climfill_esa/landmask.nc').squeeze()
interp = xr.full_like(landmask, 0).astype(float)


maxlat = 55
minlat = 35
maxlon = 27
minlon = -10
time = '2017-08-01'
interp = interp.sel(lat=slice(minlat,maxlat), lon=slice(minlon,maxlon))
stations = stations.sel(time=time)
stations = stations.where((stations.lat > minlat) & (stations.lat < maxlat) & (stations.lon > minlon) & (stations.lon < maxlon), drop=True)

# fill initial values into grid points
for n, (lat, lon) in enumerate(zip(stations.lat, stations.lon)):
    tmp = interp.sel(lat=lat, lon=lon, method='nearest')
    interp.loc[tmp.lat, tmp.lon] = stations[n].item()
    import IPython; IPython.embed()

proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection=proj)
interp.plot(ax=ax, cmap='Greys', vmin=0, vmax=2, transform=transf, add_colorbar=False)
ax.scatter(stations.lon, stations.lat, transform=transf, c='black', marker='v', s=2) 
ax.coastlines()
plt.show()
#plt.savefig(f'obsmask.pdf', bbox_inches='tight')
