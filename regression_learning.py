"""
NAMESTRING
"""

from datetime import datetime
import argparse
import xarray as xr
from sklearn.ensemble import RandomForestRegressor

from climfill.regression_learning import Imputation

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
parser.add_argument('--cluster', '-c', dest='cluster', type=int)
args = parser.parse_args()
testcase = args.testcase
c = args.cluster

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
#esapath = '/cluster/work/climate/bverena/climfill_esa_cci/' # euler

varnames = ['soil_moisture','surface_temperature','precipitation', 
            'terrestrial_water_storage','snow_cover_fraction',
            'temperature_obs','precipitation_obs','burned_area',
            'diurnal_temperature_range'] #hardcoded for now

# read data
# mask needs explicit bool otherwise lostmask is saved as int (0,1) and 
# numpy selects all datapoints as missing in imputethis since 0 and 1 
# are treated as true and no false are found
print(f'{datetime.now()} read data...')
data = xr.open_dataset(f'{esapath}{testcase}/clusters/datacluster_init_c{c:02d}.nc')['data']
mask = xr.open_dataset(f'{esapath}{testcase}/clusters/maskcluster_init_c{c:02d}.nc')['data'].astype(bool)

# gapfilling
print(f'{datetime.now()} gapfilling...')
rf_kwargs = {'burned_area':              {'n_estimators': 300, # CVed now!
                                          'min_samples_leaf': 1, 
                                          'max_samples': 1,
                                          'max_features': 0.5},
             'soil_moisture':            {'n_estimators': 300,
                                          'min_samples_leaf': 10, 
                                          'max_samples': 1.0,
                                          'max_features': 0.3},
             'precipitation_obs':        {'n_estimators': 500, 
                                          'min_samples_leaf': 10, 
                                          'max_samples': 0.5,
                                          'max_features': 0.9},
             'precipitation':            {'n_estimators': 500, 
                                          'min_samples_leaf': 10, 
                                          'max_samples': 0.5,
                                          'max_features': 0.5},
             'snow_cover_fraction':      {'n_estimators': 500, 
                                          'min_samples_leaf': 10, 
                                          'max_samples': 0.1,
                                          'max_features': 0.5},
             'landcover':                {'n_estimators': 300, 
                                          'min_samples_leaf': 2, 
                                          'max_samples': 0.5,
                                          'max_features': 0.5},
             'surface_temperature':      {'n_estimators': 300, 
                                          'min_samples_leaf': 10, 
                                          'max_samples': 0.5, #0.3 esa
                                          'max_features': 0.5},
             'temperature_obs':          {'n_estimators': 300, 
                                          'min_samples_leaf': 10, 
                                          'max_samples': 1.0,
                                          'max_features': 0.5},
             'terrestrial_water_storage':{'n_estimators': 300, 
                                          'min_samples_leaf': 0.05, 
                                          'max_samples': 0.5,
                                          'max_features': 0.3},
             'diurnal_temperature_range':{'n_estimators': 500, 
                                          'min_samples_leaf': 10, 
                                          'max_samples': 0.5, #0.1 esa
                                          'max_features': 0.5}} #0.3 esa
#rf_settings = {'n_estimators': 300,  # CV and consult table
#               'min_samples_leaf': 2,
#               'max_features': 0.5, 
#               'max_samples': 0.5, 
#               'bootstrap': True,
#               'warm_start': False,
#               'n_jobs': 30} # depends on your number of cpus 
for varname in varnames:
    rf_kwargs[varname]['bootstrap'] = True
    rf_kwargs[varname]['warm_start'] = False
    rf_kwargs[varname]['n_jobs'] = 30
regr_dict = {varname: RandomForestRegressor(**rf_kwargs[varname]) for varname in varnames}
verbose = 1
maxiter = 20

impute = Imputation(maxiter=maxiter)
data_gapfilled, test = impute.impute(
    data=data, mask=mask, regr_dict=regr_dict, verbose=verbose
)

data_gapfilled = data_gapfilled.sel(variable=varnames)

# save
print(f'{datetime.now()} save...')
data_gapfilled = data_gapfilled.to_dataset('variable')
data_gapfilled.to_netcdf(f'{esapath}{testcase}/clusters/datacluster_iter_c{c:02d}.nc')
