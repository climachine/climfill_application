"""
NAMESTRING
"""

import argparse
from datetime import datetime
import numpy as np
import xarray as xr

from climfill.postprocessing import to_latlon

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
parser.add_argument('--set', '-s', dest='veriset', type=str)
parser.set_defaults(veriset=None)
args = parser.parse_args()
testcase = args.testcase
veriset = args.veriset

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
if veriset is None:
    filepath = f'{esapath}{testcase}/clusters/'
    savepath = f'{esapath}{testcase}/'
else:
    filepath = f'{esapath}{testcase}/verification/set{veriset}/clusters/'
    savepath = f'{esapath}{testcase}/verification/set{veriset}/'

# read data
print(f'{datetime.now()} read data...')
data = xr.open_mfdataset(f'{filepath}datacluster_iter_c*.nc', 
                         combine='nested', concat_dim='datapoints').to_array().load()
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

# unstack
print(f'{datetime.now()} unstack...')
data = data.set_index(datapoints=('time','landpoints')).unstack('datapoints')
data = to_latlon(data, landmask)

# save
print(f'{datetime.now()} save...')
data = data.to_dataset('variable')
data.to_netcdf(f'{savepath}/data_climfilled.nc')
