"""
feature engineering
"""

from datetime import datetime
import numpy as np
import xarray as xr
import argparse
from climfill.feature_engineering import (
    create_embedded_feature,
    create_time_feature,
    stack_constant_maps,
)

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
print(f'{datetime.now()} read data...')
data = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc').to_array().load()
mask = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc').to_array().load() #needs to be crossval such taht verification points are updated
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# constant maps include:
# topography, aboveground biomass
constant_maps = xr.open_dataset(f'{esapath}topography.nc')

# step 2.1:  add longitude and latitude as predictors
print(f'{datetime.now()} add lat lon...')
londata, latdata = np.meshgrid(constant_maps.lon, constant_maps.lat)
constant_maps['latdata'] = (("lat", "lon"), latdata)
constant_maps['londata'] = (("lat", "lon"), londata)
constant_maps = constant_maps.to_array()

# step 2.2 (optional): remove ocean points for reducing file size
print(f'{datetime.now()} remove ocean points...')
landlat, landlon = np.where(landmask)
data = data.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints'))
mask = mask.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints'))
constant_maps = constant_maps.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                                   lat=xr.DataArray(landlat, dims='landpoints'))

# step 2.3: add time as predictor
print(f'{datetime.now()} add time...')
time_arr = create_time_feature(data)
data = data.to_dataset(dim='variable')
data['timedat'] = time_arr
data = data.to_array()

# step 2.4: add time lags as predictors
print(f'{datetime.now()} add time lags...')
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
print(f'{datetime.now()} gapfill start of timeseries...')
varmeans = data.mean(dim=('time'))
data = data.fillna(varmeans)

# step 2.5: concatenate constant maps and variables and features
print(f'{datetime.now()} concatenate data and constant maps...')
constant_maps = stack_constant_maps(data, constant_maps) 
data = xr.concat([data, constant_maps], dim='variable')

# assert that no missing values are still NaN
assert np.isnan(data).sum().item() == 0

# step 2.6: normalise data
# tree-based methods do not need standardisation
# exchange.com/questions/5277/do-you-have-to-normalize-data-when-building-decis
#ion-trees-using-r
#datamean = data.mean(dim=('time', 'landpoints'))
#datastd = data.std(dim=('time', 'landpoints'))
#data = (data - datamean) / datastd

# step 2.7: stack into tabular data
print(f'{datetime.now()} stack...')
data = data.stack(datapoints=('time', 'landpoints')).reset_index('datapoints').T
mask = mask.stack(datapoints=('time', 'landpoints')).reset_index('datapoints').T

# save
print(f'{datetime.now()} save...')
data = data.to_dataset('variable')
data.to_netcdf(f'{esapath}{testcase}/datatable.nc')
mask = mask.to_dataset('variable')
mask.to_netcdf(f'{esapath}{testcase}/masktable.nc')
