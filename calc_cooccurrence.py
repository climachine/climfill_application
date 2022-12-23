"""
NAMESTRING
"""

import itertools
import numpy as np
import pandas as pd
import xarray as xr

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','temperature_obs','precipitation_obs',
            'snow_cover_fraction','diurnal_temperature_range','burned_area'] 

# read data
mask = xr.open_dataset(f'{esapath}mask_orig.nc') 
landmask = xr.open_dataset(f'{esapath}landmask.nc').landmask
variable_combinations = list(itertools.product(varnames, varnames))

# calculate fraction of missing values overall
n_relevant = landmask.sum() * len(mask.time)
frac_mis = mask.sum() / n_relevant

tmp = np.empty((len(varnames),len(varnames)))
df = pd.DataFrame(tmp)
df.columns = varnames 
df.index = varnames 
for (var1, var2) in variable_combinations:
    both_obs = (np.logical_not(mask[var1]) & np.logical_not(mask[var2])).sum().item()
    frac_obs = np.around((both_obs / n_relevant.item())*100, decimals=2)
    df.loc[var1,var2] = frac_obs
df.to_csv('frac_obs_twovars.csv') 
