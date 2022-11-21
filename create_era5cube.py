"""
NAMESTRING
"""

import numpy as np
import xarray as xr
import xesmf as xe

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
era5landpath = '/net/exo/landclim/data/dataset/ERA5-Land/recent/0.25deg_lat-lon_1m/processed/regrid/'
era5landpathsum = '/net/exo/landclim/data/dataset/ERA5-Land/recent/0.25deg_lat-lon_1m/processed/regrid_tsum1m/'
era5pathconstant = '/net/exo/landclim/data/dataset/ERA5_deterministic/recent/0.25deg_lat-lon_time-invariant/processed/regrid/'

# common temporal extent and spatial resolution 
timeslice = slice('1995','2020')
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.75,90, 0.5)), # same as cdo_weights.nc
                     'lon': (['lon'], np.arange(-180, 180,0.5))})

# read data
swvl1 = xr.open_mfdataset(f'{era5landpath}era5-land_recent.swvl1.*.nc')
swvl2 = xr.open_mfdataset(f'{era5landpath}era5-land_recent.swvl2.*.nc')
swvl3 = xr.open_mfdataset(f'{era5landpath}era5-land_recent.swvl3.*.nc')
swvl4 = xr.open_mfdataset(f'{era5landpath}era5-land_recent.swvl4.*.nc')
skt = xr.open_mfdataset(f'{era5landpath}era5-land_recent.skt.*.nc')
tp = xr.open_mfdataset(f'{era5landpathsum}era5-land_recent.tp.*.nc')
sd = xr.open_mfdataset(f'{era5landpath}era5-land_recent.sd.*.nc')
scf = xr.open_mfdataset(f'{era5landpath}era5-land_recent.snowc.*.nc')
t2m = xr.open_mfdataset(f'{era5landpath}era5-land_recent.t2m.*.nc')
cl = xr.open_dataset(f'{era5pathconstant}era5_deterministic_recent.cl.025deg.time-invariant.nc')
dl = xr.open_dataset(f'{era5pathconstant}era5_deterministic_recent.dl.025deg.time-invariant.nc')

# calculate tws 
# convert ERA unit [m**3/m**3] to GRACE unit [water equivalent thickness, cm]
# ERA5 soil layer thicknesses (from above): 0-7, 7-28, 28-100,100-289cm
tws = xr.merge([swvl1,swvl2,swvl3,swvl4]).to_array()
layer_thickness = np.array([7, 28 - 7, 100 - 28, 289 - 100]) / 100
tws = layer_thickness[:, np.newaxis, np.newaxis, np.newaxis] * tws
tws = tws.sum(dim='variable')
tws = tws + (cl.cl * dl.dl * 100).values # lake depth in cm added
tws = tws + sd * 100  # add snow depth
twsmean = tws.sel(time=slice('2004','2009')).mean(dim='time')
tws = tws - twsmean  # substracting mean 2004-2009 as in GRACE baseline
tws = tws.rename(sd='tws')

# convert snow unit from [m] to [mm]: x1000
sd = sd * 1000

# convert tp unit from [m/month] to [mm/month]:
tp = tp * 1000 # m to mm

# convert Kelvin to Celcius
t2m = t2m - 273.15
skt = skt - 273.15

# merge to one dat
data = xr.concat([swvl1.to_array(),skt.to_array(),tp.to_array(),
                  tws.to_array(),scf.to_array(),t2m.to_array()], dim='variable')

# sel timeslice
data = data.sel(time=timeslice)

# regridding
regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
data = regridder(data)

# align with satellite dataset
data = data.to_dataset('variable')
data = data.rename(swvl1='soil_moisture',skt='surface_temperature',
                   tp='precipitation',tws='terrestrial_water_storage',
                   t2m='temperature_obs', snowc='snow_cover_fraction')
tp_obs = data.precipitation.copy()
data['precipitation_obs'] = tp_obs

# save
data.to_netcdf(f'{esapath}/data_era5land.nc')
