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

from climfill.interpolation import gapfill_thin_plate_spline

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
varnames = list(data.coords['variable'].values)

# DEBUG: only modis CV
varnames = ['surface_temperature','diurnal_temperature_range']
data = data.sel(variable=varnames)
orig = orig.sel(variable=varnames)

# timeslice
data = data.sel(time=crossvalidation_year)
orig = orig.sel(time=crossvalidation_year)

# divide into seasonal and anomal
data_monthly = data.groupby('time.month').mean()
orig_monthly = orig.groupby('time.month').mean()

# select ranges of crossval parameters RBFInterp
neighbors = [50,100,200]
smoothing = [0, 0.1, 1, 10, 100, 1000]
degree = [1,2] # degree 3 removed bec lots of errors and never best param
parameters = [neighbors, smoothing, degree]
#parameters = [[20,50],[10],[1,2]] # DEBUG 
paramnames = ['neighbors','smoothing','degree']

# cross-validate on parameters and folds (cubes)
params_combinations = list(itertools.product(*parameters))
res = []

print(f'{datetime.now()} {len(params_combinations)} parameter combination to test ...')

for params in params_combinations:
    print(f'{datetime.now()} parameter combination {params} ...')
    tps_kwargs = {varname: dict(zip(paramnames, params)) for varname in varnames}

    tmp = gapfill_thin_plate_spline(data_monthly.copy(deep=True), 
                                    landmask, tps_kwargs) 
          # deep copy necessary otherwise data_monthly gets gapfilled (ie. 
          # is identical to) tmp after 1 loop

    rmse = calc_rmse(tmp, orig_monthly, dim=('lat','lon','month'))
    res.append([*params, *rmse.values])

df = pd.DataFrame(res)
df.columns = paramnames + varnames
df = df.replace(0, np.nan) # zero rmse is unrealistic and extremely likely due to Singularity Error TODO implement somehow differently? # problem: if only occurs in individual months, overall rmse is underestimated
print(df.set_index(paramnames).idxmin())
import IPython; IPython.embed()
