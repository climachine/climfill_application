"""
NAMESTRING
"""

# running time
# from 19-11 10:30 to
# update MODIS: with debug params, change in sm, without not; indep of varnames list

from datetime import datetime
import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.ensemble import RandomForestRegressor
from climfill.regression_learning import Imputation

def calc_rmse(dat1, dat2, dim=None):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
crossvalidation_year = '2005'

# load orig data
orig = xr.open_dataset(f'{esapath}data_orig.nc').to_array()
data = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc').to_array()
mask = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc').to_array().astype(bool)
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# select crossvalidation year
orig = orig.sel(time=crossvalidation_year)
data = data.sel(time=crossvalidation_year)
mask = mask.sel(time=crossvalidation_year)

# get list of variables
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','snow_cover_fraction',
            'temperature_obs','precipitation_obs','burned_area',
            'diurnal_temperature_range'] #hardcoded for now

# DEBUG: only modis CV 
#varnames = ['surface_temperature','diurnal_temperature_range']

# select ranges of crossval parameters RBFInterp
n_estimators = [500,300]
min_samples_leaf = [1,10,0.05,0.1]
max_samples = [1.0,0.5,0.3,0.1]
max_features = [0.9, 0.5, 0.3]
#maxiter = [20,10] # maxiter cannot be different between variables, cv does not make sense
parameters = [n_estimators,min_samples_leaf, max_samples, max_features]
paramnames = ['n_estimators','min_samples_leaf','max_samples','max_features']
#parameters = [[1,10,100],[0.05],[0.5],[0.5],[10]] #DEBUG

# cross-validate on parameters and folds (cubes)
params_combinations = list(itertools.product(*parameters))
res = []

print(f'{datetime.now()} {len(params_combinations)} parameter combination to test ...')

#  convert to datatable format
# TODO change in error before and after conversion
# TODO check on landmap whether correct error is picked up
# TODO check difference between tmp and data
landlat, landlon = np.where(landmask)
data = data.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints'))
data = data.stack(datapoints=('time', 'landpoints')).reset_index('datapoints').T

orig = orig.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints'))
orig = orig.stack(datapoints=('time', 'landpoints')).reset_index('datapoints').T

mask = mask.isel(lon=xr.DataArray(landlon, dims='landpoints'),
                 lat=xr.DataArray(landlat, dims='landpoints'))
mask = mask.stack(datapoints=('time', 'landpoints')).reset_index('datapoints').T

for params in params_combinations:

    print(f'{datetime.now()} parameter combination {params} ...')

    # gapfill
    rbf_kwargs = dict(zip(paramnames[:-1], params[:-1]))
    rbf_kwargs['n_jobs'] = 30
    regr_dict = {varname: RandomForestRegressor(**rbf_kwargs) for varname in varnames}

    impute = Imputation(maxiter=params[-1])

    tmp, _ = impute.impute(data.copy(deep=True), mask.copy(deep=True),
                           regr_dict, verbose=0)
    
    tmp = tmp.sel(variable=varnames)
    # postprocessing both for DEBUGGING 
    #from climfill.postprocessing import to_latlon
    #tmp = tmp.set_index(datapoints=('time','landpoints')).unstack('datapoints')
    #tmp = to_latlon(tmp, landmask)

    #orig = orig.set_index(datapoints=('time','landpoints')).unstack('datapoints')
    #orig = to_latlon(orig, landmask)

    #import IPython; IPython.embed()
    # calc rmse
    rmse = calc_rmse(tmp, orig, dim='datapoints')
    print(f'{datetime.now()} rmse: {rmse.sel(variable="surface_temperature").item()} ...')
    import IPython; IPython.embed()
    res.append([*params, *rmse.values])

import IPython; IPython.embed()
df = pd.DataFrame(res)
df.columns = paramnames + varnames
df = df.replace(0, np.nan) # zero rmse is unrealistic and extremely likely due to Singularity Error TODO implement somehow differently? # problem: if only occurs in individual months, overall rmse is underestimated
print(df.set_index(paramnames).idxmin())
import IPython; IPython.embed()
