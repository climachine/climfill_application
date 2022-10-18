"""
feature engineering
"""

import xarray as xr
from climfill.feature_engineering import (
    create_embedded_feature,
    create_lat_lon_features,
    create_time_feature,
    stack_constant_maps,
)

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
# TODO continue running and testing here
data = xr.open_dataset(f'{esapath}interpolated/data_interpolated.nc').to_array().load()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask
import IPython; IPython.embed()

# constant maps include:
# landcover, aboveground biomass
# TODO include rest
landcover = xr.open_dataset(f'{esapath}landcover_yearly/landcover_2003.nc')

# step 2.1:  add longitude and latitude as predictors
latitude_arr, longitude_arr = create_lat_lon_features(constant_maps)
constant_maps['latdata'] = latitude_arr
constant_maps['londata'] = longitude_arr
constant_maps = constant_maps.to_array()

# step 2.2: create mask of missing values
mask = np.isnan(data)

# step 2.3 (optional): remove ocean points for reducing file size
landlat, landlon = np.where(landmask)
data = data.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints'))
mask = mask.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints'))
constant_maps = constant_maps.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                                   lat=xr.DataArray(landlat, dims='landpoints'))

# step 2.4: add time as predictor
time_arr = create_time_feature(data)
data = data.to_dataset(dim='variable')
data['timedat'] = time_arr
data = data.to_array()

# step 2.5: add time lags as predictors
lag_007b = create_embedded_feature(data, start=-7,   end=0, name='lag_7b')
lag_030b = create_embedded_feature(data, start=-30,  end=-7, name='lag_30b')
lag_180b = create_embedded_feature(data, start=-180, end=-30, name='lag_180b')
lag_007f = create_embedded_feature(data, start=0,    end=7, name='lag_7f')
lag_030f = create_embedded_feature(data, start=7,    end=30, name='lag_30f')
lag_180f = create_embedded_feature(data, start=30,   end=180, name='lag_180f')
data = xr.concat(
    [data, lag_007b, lag_030b, lag_180b, lag_007f, lag_030f, lag_180f], 
        dim='variable', join='left', fill_value=0)

# fill still missing values at beginning of time series
varmeans = data.mean(dim=('time'))
data = data.fillna(varmeans)

# step 2.6: concatenate constant maps and variables and features
constant_maps = stack_constant_maps(data, constant_maps)
data = xr.concat([data, constant_maps], dim='variable')

# step 2.7: normalise data
datamean = data.mean(dim=('time', 'landpoints'))
datastd = data.std(dim=('time', 'landpoints'))
data = (data - datamean) / datastd

# step 2.8: stack into tabular data
data = data.stack(datapoints=('time', 'landpoints')).reset_index('datapoints').T
mask = mask.stack(datapoints=('time', 'landpoints')).reset_index('datapoints').T

# save
data.to_netcdf(f'{esapath}datatable.nc')
mask.to_netcdf(f'{esapath}masktable.nc')
