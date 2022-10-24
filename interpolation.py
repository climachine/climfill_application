"""
interpolate initial gapfill
"""

import numpy as np
import xarray as xr
import argparse
from datetime import datetime
from climfill.interpolation import gapfill_thin_plate_spline, gapfill_kriging

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
print(f'{datetime.now()} read data...')
data = xr.open_dataset(f'{esapath}{testcase}/data_crossval.nc')
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# xarray/dask issue https://github.com/pydata/xarray/issues/3813
# value assignment only works if non-dask array
data = data.to_array().load()

# divide into monthly climatology and anomalies
print(f'{datetime.now()} divide clim and anomalies...')
data_monthly = data.groupby('time.month').mean()
data_anom = data.groupby('time.month') - data_monthly 

# gapfill monthly data with thin-plate-spline interpolation
print(f'{datetime.now()} gapfill seasonal...')
rbf_kwargs = {'burned_area':              {'neighbors': 100, 
                                           'smoothing': 0.1, 
                                           'degree': 1},
              'soil_moisture':            {'neighbors': 100,
                                           'smoothing': 10, 
                                           'degree': 2}, 
              'precipitation_obs':        {'neighbors': 100, 
                                           'smoothing': 0.1, 
                                           'degree': 1},
              'precipitation':            {'neighbors': 100, 
                                           'smoothing': 0.1, 
                                           'degree': 1},
              'snow_water_equivalent':    {'neighbors': 100, 
                                           'smoothing': 0.1, 
                                           'degree': 1},
              'surface_temperature':      {'neighbors': 100, 
                                           'smoothing': 0.1, 
                                           'degree': 1},
              'temperature_obs':          {'neighbors': 100, 
                                           'smoothing': 0.1, 
                                           'degree': 1},
              'terrestrial_water_storage':{'neighbors': 100, 
                                           'smoothing': 0.1, 
                                           'degree': 1}}
data_monthly = gapfill_thin_plate_spline(data_monthly, landmask, rbf_kwargs)

# gapfill anomalies with kriging
print(f'{datetime.now()} gapfill anomalies...')
kriging_kwargs = {'burned_area':              {'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5},
                  'soil_moisture':            {'constant_value': 100,
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5}, 
                  'precipitation_obs':        {'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5},
                  'precipitation':            {'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5},
                  'snow_water_equivalent':    {'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5},
                  'surface_temperature':      {'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5},
                  'temperature_obs':          {'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5},
                  'terrestrial_water_storage':{'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5}}
import warnings # DEBUG
warnings.simplefilter('ignore')
data_anom = gapfill_kriging(data_anom, landmask, kriging_kwargs)

# step 1.4: add monthly climatology and anomalies back together
print(f'{datetime.now()} add together...')
data = data_anom.groupby('time.month') + data_monthly
data = data.drop('month') # month not needed anymore

# necessary if full days are missing: fill all remaining gaps with variable mean
# TODO interpolate in time, because mostly full slices are missing
print(f'{datetime.now()} last missing values...')
if np.isnan(data).sum() != 0: # if still missing values present
    print('still missing values treatment')
    data = data.interpolate_na(dim='time', method='linear')
    variable_mean = data.mean(dim=('time', 'lat', 'lon'))
    data = data.fillna(variable_mean)

# test if all missing values are caught and infilled
assert np.isnan(data).sum().item() == 0

# save
print(f'{datetime.now()} save...')
data = data.to_dataset('variable')
data.to_netcdf(f'{esapath}{testcase}/data_interpolated.nc')
