"""
NAMESTRING
"""

# TODO logical problem: CV over clusters, but clusters may not only include
# CV year. solve

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
orig = xr.open_dataset(f'{esapath}{testcase}/data_fromcluster.nc').to_array()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# get orig in shape for comparison
orig = orig.sel(time=crossvalidation_year)
landlat, landlon = np.where(landmask)
orig = orig.isel(lon=xr.DataArray(landlon, dims="landpoints"), 
                 lat=xr.DataArray(landlat, dims="landpoints"))

# get list of variables
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','snow_cover_fraction',
            'temperature_obs','precipitation_obs','burned_area',
            'diurnal_temperature_range'] #hardcoded for now

# select ranges of crossval parameters RBFInterp
n_estimators = [300,500]
min_samples_leaf = [0.05,0.1, 0.2, 0.3, 0.4, 0.5]
max_samples = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
max_features = [0.9, 0.5, 0.3]
maxiter = [10,20]
parameters = [n_estimators,min_samples_leaf, max_samples, max_features, maxiter]
paramnames = ['n_estimators','min_samples_leaf','max_samples','max_features','maxiter']
parameters = [[1,10,300],[0.05],[0.05],[0.5],[10]] #DEBUG

# cross-validate on parameters and folds (cubes)
params_combinations = list(itertools.product(*parameters))
res = []

print(f'{datetime.now()} {len(params_combinations)} parameter combination to test ...')

# rf speciality: take 3 largest clusters
clusters = ['c00','c28','c01'] #c03 left out bec too large (8.1G compared to ~700MB)
clusters = ['c00'] #DEBUG TODO new since clustering new

#def to_latlon_1year(data, lat, lon, landmask):
#    shape = landmask.shape
#    tmp = xr.DataArray(np.full((data.coords['variable'].size,
#                                data.coords['time'].size,
#                                shape[0],shape[1]),np.nan), 
#                       coords=[data.coords['variable'], data.coords['time'], 
#                              landmask.coords['lat'], landmask.coords['lon']], 
#                       dims=['variable','time','lat','lon'])
#    tmp.values[:,:,lat,lon] = data
#    return tmp

for cluster in clusters:

    # DEBUG log
    # error1: without impute, rmse between cluster and interp data should be zero
    # data_interpolated and data_fromcluster are the same checked at 4D
    # data_fromcluster and cluster00 are NOT the same at 3D
    # cluster00 cannot be transformed to 4D because lat&lon is all nan
    # error2: with impute, rmse should differ accross diff params combinations

    # read data
    data = xr.open_dataset(f'{esapath}{testcase}/clusters/datacluster_init_{cluster}.nc')['data']
    mask = xr.open_dataset(f'{esapath}{testcase}/clusters/maskcluster_init_{cluster}.nc')['data'].astype(bool)
    import IPython; IPython.embed()

    # postprocess
    data = data.sel(variable=varnames)
    data = data.set_index(datapoints=('time', 'landpoints')).unstack('datapoints')
    data = data.sel(time=data['time.year'] == int(crossvalidation_year))

    # prepare orig
    #test = orig.copy(deep=True) #DEBUG
    orig = orig.sel(landpoints=data.landpoints)#.drop_sel(variable='landcover')
    #orig = orig.sel(time=data['time.year'] == int(crossvalidation_year))
    import IPython; IPython.embed()

    for params in params_combinations:

        print(f'{datetime.now()} cluster {cluster} parameter combination {params} ...')

        # gapfill
        rbf_kwargs = dict(zip(paramnames[:-1], params[:-1]))
        regr_dict = {varname: RandomForestRegressor(**rbf_kwargs) for varname in varnames}

        impute = Imputation(maxiter=params[-1])

        tmp, _ = impute.impute(data.copy(deep=True), mask.copy(deep=True),
                               regr_dict, verbose=0)
        import IPython; IPython.embed()

        # postprocess
        tmp = tmp.sel(variable=varnames)
        tmp = tmp.set_index(datapoints=('time', 'landpoints')).unstack('datapoints')
        tmp = tmp.sel(time=tmp['time.year'] == int(crossvalidation_year))

        # prepare orig
        #test = orig.copy(deep=True) #DEBUG
        import IPython; IPython.embed()
        tmp2 = orig.sel(landpoints=data.landpoints).drop_sel(variable='landcover')
        tmp2 = tmp2.sel(time=data['time.year'] == int(crossvalidation_year))

        # calc rmse
        rmse = calc_rmse(tmp, tmp2, dim=('landpoints','time'))
        import IPython; IPython.embed()
        res.append([cluster, *params, *rmse.values])
        print(calc_rmse(tmp.sel(variable='soil_moisture'),tmp2.sel(variable='soil_moisture')).item())
        del(tmp,tmp2)

df = pd.DataFrame(res)
df.columns = ['cluster'] + paramnames + varnames
df = df.replace(0, np.nan) # zero rmse is unrealistic and extremely likely due to Singularity Error TODO implement somehow differently? # problem: if only occurs in individual months, overall rmse is underestimated
df = df.groupby('cluster').mean()
print(df.set_index(paramnames).idxmin())
import IPython; IPython.embed()
