"""
interpolate initial gapfill
"""

import numpy as np
import xarray as xr
import argparse
from datetime import datetime, timedelta
from climfill.interpolation import gapfill_thin_plate_spline, gapfill_kriging

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data, NEW: without crossval holes
print(f'{datetime.now()} read data...')
data = xr.open_dataset(f'{esapath}{testcase}/data_crossval.nc')
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# xarray/dask issue https://github.com/pydata/xarray/issues/3813
# value assignment only works if non-dask array
data = data.to_array().load()

# divide into climatology and anomalies
data_monthly = data.groupby('time.month').mean()
data_anom = data.groupby('time.month') - data_monthly 

# gapfill monthly data with thin-plate-spline interpolation
print(f'{datetime.now()} gapfill seasonal...')
rbf_kwargs = {'burned_area':              {'neighbors': 200, # CVed now!
                                           'smoothing': 1000, 
                                           'degree': 1},
              'soil_moisture':            {'neighbors': 200,
                                           'smoothing': 100, 
                                           'degree': 1}, 
              'precipitation_obs':        {'neighbors': 200, 
                                           'smoothing': 0, 
                                           'degree': 1},
              'precipitation':            {'neighbors': 200, 
                                           'smoothing': 100, 
                                           'degree': 1},
              'snow_cover_fraction':      {'neighbors': 200, 
                                           'smoothing': 100, 
                                           'degree': 1},
              'landcover':                {'neighbors': 100, 
                                           'smoothing': 0.1, 
                                           'degree': 1},
              'surface_temperature':      {'neighbors': 200, 
                                           'smoothing': 1000, 
                                           'degree': 1},
              'temperature_obs':          {'neighbors': 200, 
                                           'smoothing': 0.1, 
                                           'degree': 1},
              'terrestrial_water_storage':{'neighbors': 200, 
                                           'smoothing': 1, 
                                           'degree': 1},
              'diurnal_temperature_range':{'neighbors': 100, 
                                           'smoothing': 1000, 
                                           'degree': 1}}
data_monthly = gapfill_thin_plate_spline(data_monthly.copy(deep=True), landmask, rbf_kwargs)

# gapfill anomalies with kriging
print(f'{datetime.now()} gapfill anomalies...')
kriging_kwargs = {'burned_area':              {'constant_value': 1, # CVed!
                                               'length_scale': 20, 
                                               'npoints': 10, 
                                               'repeats': 10},
                  'soil_moisture':            {'constant_value': 10,
                                               'length_scale': 1, 
                                               'npoints': 1000, 
                                               'repeats': 2}, 
                  'precipitation_obs':        {'constant_value': 10, 
                                               'length_scale': 0.1, 
                                               'npoints': 1000, 
                                               'repeats': 1},
                  'precipitation':            {'constant_value': 10, 
                                               'length_scale': 0.1, 
                                               'npoints': 1000, 
                                               'repeats': 2},
                  'snow_cover_fraction':      {'constant_value': 100, 
                                               'length_scale': 1, 
                                               'npoints': 1000, 
                                               'repeats': 2},
                  'landcover':                {'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 5},
                  'surface_temperature':      {'constant_value': 10, 
                                               'length_scale': 30, 
                                               'npoints': 10, 
                                               'repeats': 1},
                  'temperature_obs':          {'constant_value': 100, 
                                               'length_scale': 10, 
                                               'npoints': 100, 
                                               'repeats': 10},
                  'terrestrial_water_storage':{'constant_value': 100, 
                                               'length_scale': 1, 
                                               'npoints': 100, 
                                               'repeats': 10},
                  'diurnal_temperature_range':{'constant_value': 10, 
                                               'length_scale': 50, 
                                               'npoints': 10, 
                                               'repeats': 10}}
import warnings # DEBUG
warnings.simplefilter('ignore')
data_anom = gapfill_kriging(data_anom.copy(deep=True), landmask, kriging_kwargs)

# step 1.4: add monthly climatology and anomalies back together
print(f'{datetime.now()} add together...')
data = data_anom.groupby('time.month') + data_monthly
data = data.drop('month') # month not needed anymore

# necessary if full months are missing: interpolate in time
#print(f'{datetime.now()} fill individual months with time interp...')
#import IPython; IPython.embed()
#if np.isnan(data).sum() != 0: # if still missing values present
#    pass
#    #data = data.interpolate_na(dim='time', method='linear', max_gap=timedelta(days=32)) # Module not found bottleneck

# elongate time series: gapfill with existing climatology
if np.isnan(data).sum() != 0: # if still missing values present
    years = np.unique(data['time.year'])
    for year in years:
        tslice=slice(f'{year}-01-01',f'{year}-12-31')
        data.loc[:,tslice,:,:] = data.loc[:,tslice,:,:].fillna(data_monthly.values)

if np.isnan(data).sum() != 0: # if still missing values present
    print(f'{datetime.now()} last gaps fill with variable mean ...')
    variable_mean = data.mean(dim=('time', 'lat', 'lon'))
    data = data.fillna(variable_mean)

# test if all missing values are caught and infilled
assert np.isnan(data).sum().item() == 0

# save
print(f'{datetime.now()} save...')
data = data.to_dataset('variable')
data.to_netcdf(f'{esapath}{testcase}/data_interpolated.nc')
