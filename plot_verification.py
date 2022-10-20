"""
NAMESTRING
"""

import argparse
import numpy as np
from scipy.spatial.distance import jensenshannon as js
import xarray as xr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
verification_year = '2004'

def calc_rmse(dat1, dat2, dim):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

def js_one_point(lndpt1, lndpt2, bins):
    hist1, _ = np.histogramdd(lndpt1, bins=bins)
    hist2, _ = np.histogramdd(lndpt2, bins=bins)
    return js(hist1.flatten(),hist2.flatten())

# NOTE:
# TWS does not have any originally missing values in 2004
# soil moisture does not have any CV values, prob because hard to catch them
#      maybe make sure it is 10% missing per variable and not overall? # DONE in test2
# naturally gap-free variables have gaps. # change! # update: no since time
#      elongation is also a gap!
# JS not possible bec needs times where all 8 vars are observed (i.e. in the
#      verification set) at the same time. ditched for now

# read data
orig = xr.open_dataset(f'{esapath}{testcase}/data_orig.nc')
intp = xr.open_dataset(f'{esapath}{testcase}/data_interpolated.nc')
fill = xr.open_dataset(f'{esapath}{testcase}/data_climfilled.nc')

mask_orig = xr.open_dataset(f'{esapath}{testcase}/mask_orig.nc')
mask_cv = xr.open_dataset(f'{esapath}{testcase}/mask_crossval.nc')

# select verification year
mask_orig = mask_orig.sel(time=verification_year).load()
mask_cv = mask_cv.sel(time=verification_year).load()
orig = orig.sel(time=verification_year).load()
intp = intp.sel(time=verification_year).load()
fill = fill.sel(time=verification_year).load()

# calculate mask of verification points
mask_cv = np.logical_and(np.logical_not(mask_orig), mask_cv)

# mask everything except verification points
orig = orig.where(mask_cv)
intp = intp.where(mask_cv)
fill = fill.where(mask_cv)

# sort data
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','snow_water_equivalent',
            'temperature_obs','precipitation_obs','burned_area'] #hardcoded for now
orig = orig.to_array().reindex(variable=varnames)
intp = intp.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)

# calc metrics
corr_intp = xr.corr(orig, intp, dim=('lat','lon','time'))
corr_fill = xr.corr(orig, fill, dim=('lat','lon','time'))
rmse_intp = calc_rmse(orig, intp, dim=('lat','lon','time'))
rmse_fill = calc_rmse(orig, fill, dim=('lat','lon','time'))
import IPython; IPython.embed()

# plot
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
x_pos =np.arange(0,2*len(corr_intp),2)
wd = 0.3

ax1.bar(x_pos, corr_intp, width=wd)
ax1.bar(x_pos+wd, corr_fill, width=wd)

ax2.bar(x_pos, rmse_intp, width=wd)
ax2.bar(x_pos+wd, rmse_fill, width=wd)

ax1.set_xticks([])
ax2.set_xticks(x_pos+0.5*wd, varnames, rotation=90)
ax2.set_ylim([0,40]) #TODO debug remove
ax1.set_xlim([-1,16])
ax2.set_xlim([-1,16])

ax1.set_title('pearson correlation of verification pts')
ax2.set_title('RSME of verification pts')

plt.subplots_adjust(bottom=0.3)
plt.show()
