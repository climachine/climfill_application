"""
interpolate initial gapfill
"""

import xarray as xr
from climfill.interpolation import gapfill_thin_plate_spline, gapfill_kriging

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
data = xr.open_mfdataset(f'{esapath}*.nc')

# extract landmask
landmask = data.landmask
data = data.drop('landmask') 

# divide into monthly climatology and anomalies
data_monthly = data.groupby('time.month').mean()
data_anom = data.groupby('time.month') - data_monthly

# gapfill monthly data with thin-plate-spline interpolation
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
# xarray/dask issue https://github.com/pydata/xarray/issues/3813
# value assignment only works if non-dask array
#data_monthly = gapfill_thin_plate_spline(data_monthly.to_array().load(), landmask, rbf_kwargs)

# gapfill anomalies with kriging
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
data_anom = gapfill_kriging(data_anom.to_array().load(), landmask, kriging_kwargs)

# step 1.4: add monthly climatology and anomalies back together
data = data_anom.groupby('time.month') + data_monthly
data = data.drop('month') # month not needed anymore

# necessary if full days are missing: fill all remaining gaps with variable mean
if np.isnan(data).sum() != 0: # if still missing values present
    print('still missing values treatment')
    variable_mean = data.mean(dim=('time', 'lat', 'lon'))
    data = data.fillna(variable_mean)

# test if all missing values are caught and infilled
assert np.isnan(data).sum().item() == 0

# save
data = data.to_dataset('variable')
data.to_netcdf(f'{esapath}data_interpolated.nc')
