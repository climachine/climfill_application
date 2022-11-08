"""
NAMESTRING
"""

# TODO logical problem: CV over clusters, but clusters may not only include
# CV year. solve

import argparse
import itertools

import numpy as np
import pandas as pd
import xarray as xr

from climfill.interpolation import gapfill_kriging

def calc_rmse(dat1, dat2, dim=None):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
crossvalidation_year = '2005'

# load data
orig = xr.open_dataset(f'{esapath}data_orig.nc').to_array()
data = xr.open_dataset(f'{esapath}{testcase}/data_crossval.nc').to_array()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# get list of variables
varnames = data.coords['variable'].values

# monthly mean from whole timeseries, such that anomalies exist
data_monthly = data.groupby('time.month').mean()
orig_monthly = orig.groupby('time.month').mean()

# timeslice
data = data.sel(time=crossvalidation_year)
orig = orig.sel(time=crossvalidation_year)

# divide into seasonal and anomal
data_anom = data.groupby('time.month') - data_monthly
orig_anom = orig.groupby('time.month') - orig_monthly

# select ranges of crossval parameters GP
# TODO maybe gp approx not necessary anymore?
constant_value = [0.01, 0.1, 1, 10, 100]
length_scale = [0.1,1,10,20,30,50,100]
repeats = [1,2,5,10]
npoints = [10,100,1000,2000] 
iarameters = [constant_value, length_scale, repeats, npoints]
#parameters = [[10],[1],[1],[10,20]] # DEBUG
paramnames = ['constant_value','length_scale','repeats','npoints']

# cross-validate on parameters and folds (cubes)
params_combinations = list(itertools.product(*parameters))
res = []

for params in params_combinations:
    gp_kwargs = {varname: dict(zip(paramnames, params)) for varname in varnames}

    tmp = gapfill_kriging(data_anom.copy(deep=True), 
                          landmask, gp_kwargs) 
          # deep copy necessary otherwise data_monthly gets gapfilled (ie. 
          # is identical to) tmp after 1 loop

    rmse = calc_rmse(tmp, orig_anom, dim=('lat','lon','time'))
    res.append([*params, *rmse.values])

df = pd.DataFrame(res)
df.columns = paramnames + varnames.tolist()
df = df.replace(0, np.nan) # zero rmse is unrealistic and extremely likely due to Singularity Error TODO implement somehow differently?
print(df.set_index(paramnames).idxmin())
import IPython; IPython.embed()
