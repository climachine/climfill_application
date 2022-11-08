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

from climfill.interpolation import gapfill_thin_plate_spline, gapfill_kriging

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

# timeslice
data = data.sel(time=crossvalidation_year)
orig = orig.sel(time=crossvalidation_year)

# divide into seasonal and anomal
data_monthly = data.groupby('time.month').mean()
data_anom = data.groupby('time.month') - data_monthly 
orig_monthly = orig.groupby('time.month').mean()
orig_anom = orig.groupby('time.month') - orig_monthly 

# select ranges of crossval parameters RBFInterp
neighbors = [20,50,100]
smoothing = [0, 0.1, 1, 10, 100, 1000]
degree = [1,2,3]
parameters = [neighbors, smoothing, degree]
#parameters = [[20,50],[10],[1,2]] # DEBUG 
paramnames = ['neighbors','smoothing','degree']

# cross-validate on parameters and folds (cubes)
params_combinations = list(itertools.product(*parameters))
res = []

for params in params_combinations:
    rbf_kwargs = {varname: dict(zip(paramnames, params)) for varname in varnames}

    tmp = gapfill_thin_plate_spline(data_monthly.copy(deep=True), landmask, rbf_kwargs) #copy necessary otherwise data_monthly gets gapfilled (ie. is identical to) tmp after 1 loop

    rmse = calc_rmse(tmp, orig_monthly, dim=('lat','lon','month'))
    # rmse zero for some variables
    # is possible for individual param combinations due to 
    # Singularity Error in thin-plate-spline
    # tested: not zero rmse if neighbors large enough
    # but rmse between data_monthly and orig_monthly should be zero, is not
    # is possible because different monthly means, since more values are 
    # missing in crossval; original ncs and anomalies have rmse zero checked
    # also: deep copy necessary as input for gapfill_thin_plate_spline
    res.append([*params, *rmse.values])

df = pd.DataFrame(res)
df.columns = paramnames + varnames.tolist()
df = df.replace(0, np.nan) # zero rmse is unrealistic and extremely likely due to Singularity Error TODO implement somehow differently?
print(df.set_index(paramnames).idxmin())
import IPython; IPython.embed()
