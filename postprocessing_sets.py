"""
NAMESTRING
"""

import argparse
import numpy as np
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
parser.add_argument('--set', '-s', dest='veriset', type=str)
parser.set_defaults(veriset=None)
args = parser.parse_args()
testcase = args.testcase
veriset = args.veriset

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','temperature_obs','precipitation_obs',
            'snow_cover_fraction','diurnal_temperature_range','burned_area'] 
label = 'interpolated'
mask_orig = xr.open_dataset(f'{esapath}mask_orig.nc')

fill = xr.open_dataset(f'{esapath}{testcase}/verification/set{veriset}/data_{label}.nc')
mask_cv = xr.open_dataset(f'{esapath}{testcase}/verification/set{veriset}/mask_crossval.nc')

mask = np.logical_and(np.logical_not(mask_orig), mask_cv)
fill = fill.to_array().reindex(variable=varnames)
mask = mask.to_array().reindex(variable=varnames)

fill = fill.where(mask)

verification_year = slice('2004','2005')
fill = fill.sel(time=verification_year)

fill = fill.expand_dims(veriset=[veriset])
fill.to_dataset('variable').to_netcdf(f'{esapath}{testcase}/verification/set{veriset}/data_{label}_del.nc')
