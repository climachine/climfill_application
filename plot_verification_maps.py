"""
NAMESTRING
"""

import argparse
import numpy as np
import regionmask
import cartopy.crs as ccrs
from scipy.spatial.distance import jensenshannon as js
import xarray as xr
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
args = parser.parse_args()
testcase = args.testcase

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
verification_year = slice('2004','2005')

def calc_rmse(dat1, dat2, dim):
    return np.sqrt(((dat1 - dat2)**2).mean(dim=dim))

# NOTE:
# JS not possible bec needs times where all 8 vars are observed (i.e. in the
#      verification set) at the same time. ditched for now

# read data
orig = xr.open_dataset(f'{esapath}data_orig.nc')
#fill = xr.open_dataset(f'{esapath}{testcase}/verification/data_climfilled.nc')
#intp = xr.open_dataset(f'{esapath}{testcase}/verification/data_interpolated.nc')
#intp0 = xr.open_dataset(f'{esapath}{testcase}/verification/set0/data_interpolated_tmp.nc') #DEBUG
#fill0 = xr.open_dataset(f'{esapath}{testcase}/verification/set0/data_climfilled_tmp.nc') #DEBUG
intp = xr.open_mfdataset(f'{esapath}{testcase}/verification/set?/data_interpolated_tmp.nc').load()
fill = xr.open_mfdataset(f'{esapath}{testcase}/verification/set?/data_climfilled_tmp.nc').load()
init = xr.open_mfdataset(f'{esapath}{testcase}/verification/set?/data_initguess_tmp.nc').load()

# select verification year
orig = orig.sel(time=verification_year).load()

# sort data
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','temperature_obs','precipitation_obs',
            'snow_cover_fraction','diurnal_temperature_range','burned_area'] 
varnames_plot = ['SM','LST','PSAT', #for order of plots
            'TWS', 'T2M','P2M',
            'SCF', 'DTR', 'BA']
orig = orig.to_array().reindex(variable=varnames)
intp = intp.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)
init = init.to_array().reindex(variable=varnames)

intp_set = intp.mean(dim='veriset')
fill_set = fill.mean(dim='veriset')
init_set = init.mean(dim='veriset')

## DEBUG
varname = 'soil_moisture'
lat = -25
lon = 124
for veriset in range(10):
    orig.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='black')
    intp_set.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='orange')
    #init_set.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').plot(color='red')
    #fill_set.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').plot(color='blue')
    #intp.sel(variable=varname, veriset=veriset).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='yellow')
    #fill.sel(variable=varname, veriset=veriset).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='lightblue')
    #init.sel(variable=varname, veriset=veriset).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='lightcoral')
    plt.show()
quit()

# (optional) calculate anomalies
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
intp = intp.groupby('time.month') - intp.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# normalise values such that RMSEs (i.e. negative skill scores) are comparable
# and rounding has same effect on every variable
datamean = orig.mean()
datastd = orig.std()
orig = (orig - datamean) / datastd
intp = (intp - datamean) / datastd
fill = (fill - datamean) / datastd

# plot
proj = ccrs.Robinson()
transf = ccrs.PlateCarree()
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(331, projection=proj)
ax2 = fig.add_subplot(332, projection=proj)
ax3 = fig.add_subplot(333, projection=proj)
ax4 = fig.add_subplot(334, projection=proj)
ax5 = fig.add_subplot(335, projection=proj)
ax6 = fig.add_subplot(336, projection=proj)
ax7 = fig.add_subplot(337, projection=proj)
ax8 = fig.add_subplot(338, projection=proj)
ax9 = fig.add_subplot(339, projection=proj)
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]
levels = np.arange(-1,1.1,0.1)
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))
obsmask = xr.open_dataset(f'{esapath}landmask.nc').landmask
cmap = plt.get_cmap('seismic_r')
cmap.set_over('aliceblue')
cmap.set_bad('lightgrey')

#varnames = ['burned_area'] #DEBUG
for v, (varname, ax) in enumerate(zip(varnames, axes)):

    # avoid very small skill scores by very small differences; scf they are -inf now
    orig = np.round(orig, decimals=5)
    intp = np.round(intp, decimals=5)
    fill = np.round(fill, decimals=5)

    # calculate rmse
    rmseintp = calc_rmse(orig.sel(variable=varname), intp.sel(variable=varname), dim=('time'))
    rmsefill = calc_rmse(orig.sel(variable=varname), fill.sel(variable=varname), dim=('time'))

    # calculate skill score
    skillscore = 1 - (rmsefill/rmseintp)
    skillscore = xr.corr(orig.sel(variable=varname),intp.sel(variable=varname), dim='time') #DEBUG

    # mask regions not included
    skillscore = skillscore.where(obsmask) # not obs dark grey
    skillscore = skillscore.where(np.logical_not(landmask), 10) # ocean blue

    # plot
    im = skillscore.plot(ax=ax, cmap=cmap, vmin=-1, vmax=1, transform=transf, 
                  levels=levels, add_colorbar=False)
    ax.set_title(varnames_plot[v])

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.6]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('RMSE Skill Score')

plt.savefig('verification_maps.png')
