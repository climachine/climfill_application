"""
NAMESTRING
"""

import argparse
import numpy as np
import regionmask
import cartopy.crs as ccrs
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
mask_initguess = xr.open_dataset(f'{esapath}{testcase}/mask_initguess.nc')
mask_cubes = xr.open_dataset(f'{esapath}{testcase}/verification/mask_cubes.nc')
#init = xr.open_mfdataset(f'{esapath}{testcase}/verification/set?/data_initguess_del.nc').load()
intp = xr.open_mfdataset(f'{esapath}{testcase}/verification/set?/data_interpolated_del.nc').load()
fill = xr.open_mfdataset(f'{esapath}{testcase}/verification/set?/data_climfilled_del.nc').load()

# select verification year
orig = orig.sel(time=verification_year).load()

# select only verification cubes
mask_cubes = mask_cubes.sel(veriset=0) # DEBUG
intp = intp.where(np.logical_not(mask_cubes))
fill = fill.where(np.logical_not(mask_cubes))
orig = orig.where(np.logical_not(mask_cubes))

# average over all set
intp = intp.mean(dim='veriset').load()
fill = fill.mean(dim='veriset').load()

# mask still missing after init guess
intp = intp.where(np.logical_not(mask_initguess))
fill = fill.where(np.logical_not(mask_initguess))

# normalise values such that RMSEs (i.e. negative skill scores) are comparable
# and rounding has same effect on every variable
#datamean = orig.mean()
#datastd = orig.std()
#orig = (orig - datamean) / datastd
#intp = (intp - datamean) / datastd
#fill = (fill - datamean) / datastd

# sort data
varnames = ['soil_moisture','surface_temperature','precipitation',
            'terrestrial_water_storage','temperature_obs','precipitation_obs',
            'snow_cover_fraction','diurnal_temperature_range','burned_area'] 
varnames_plot = ['SM','LST','PSAT', #for order of plots
            'TWS', 'T2M','P2M',
            'SCF', 'DTR', 'BA']
orig = orig.to_array().reindex(variable=varnames)
#init = init.to_array().reindex(variable=varnames)
intp = intp.to_array().reindex(variable=varnames)
fill = fill.to_array().reindex(variable=varnames)

#init_set = init_set.to_array().reindex(variable=varnames)
#intp_set = intp_set.to_array().reindex(variable=varnames)
#fill_set = fill_set.to_array().reindex(variable=varnames)

## DEBUG
#varname = 'precipitation'
#lat = -25
#lon = 124
##import IPython; IPython.embed()
#for veriset in range(10):
#    orig.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='black')
#    intp_set.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='orange')
#    fill_set.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='blue')
#    init_set.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').plot(color='red')
##    #fill_set.sel(variable=varname).sel(lat=lat, lon=lon, method='nearest').plot(color='blue')
#    #intp.sel(variable=varname, veriset=str(veriset)).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='yellow')
#    #fill.sel(variable=varname, veriset=str(veriset)).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='lightblue')
#    #init.sel(variable=varname, veriset=str(veriset)).sel(lat=lat, lon=lon, method='nearest').fillna(0).plot(color='lightcoral')
#    plt.show()
##quit()
#init = init_set
#intp = intp_set
#fill = fill_set

# (optional) calculate anomalies
orig = orig.groupby('time.month') - orig.groupby('time.month').mean()
#init = init.groupby('time.month') - init.groupby('time.month').mean()
intp = intp.groupby('time.month') - intp.groupby('time.month').mean()
fill = fill.groupby('time.month') - fill.groupby('time.month').mean()

# now here: regional averages such that init guess is included
landmask = regionmask.defined_regions.natural_earth_v5_0_0.land_110.mask(orig.lon, orig.lat)
regions = regionmask.defined_regions.ar6.land.mask(orig.lon, orig.lat)
regions = regions.where(~np.isnan(landmask))
obsmask = xr.open_dataset(f'{esapath}landmask.nc').landmask

orig = orig.groupby(regions).mean()
intp = intp.groupby(regions).mean()
fill = fill.groupby(regions).mean()

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
cmap = plt.get_cmap('seismic_r')
cmap.set_under('aliceblue')
cmap.set_bad('lightgrey')
fs = 15

def expand_to_worldmap(data,regions):
    test = xr.full_like(regions, np.nan)
    for region, r in zip(range(int(regions.max().item())), data):
        test = test.where(regions != region, r)
    return test

#varnames = ['burned_area'] #DEBUG
for v, (varname, ax) in enumerate(zip(varnames, axes)):

    # avoid very small skill scores by very small differences; scf they are -inf now
    #orig = np.round(orig, decimals=5)
    #intp = np.round(intp, decimals=5)
    #fill = np.round(fill, decimals=5)

    # calculate rmse
    #rmseintp = calc_rmse(orig.sel(variable=varname), intp.sel(variable=varname), dim=('time'))
    #rmsefill = calc_rmse(orig.sel(variable=varname), fill.sel(variable=varname), dim=('time'))

    ## calculate skill score
    #skillscore = 1 - (rmsefill/rmseintp)
    skillscore = xr.corr(orig.sel(variable=varname),intp.sel(variable=varname), dim='time') #DEBUG
    #skillscore = rmsefill # DEBUG

    # expand to worldmap
    skillscore = expand_to_worldmap(skillscore, regions)

    # mask regions not included
    skillscore = skillscore.where(obsmask) # not obs dark grey
    print(varname, skillscore.median().item())
    skillscore = skillscore.where(np.logical_not(landmask), -10) # ocean blue

    # plot
    im = skillscore.plot(ax=ax, cmap=cmap, vmin=-1, vmax=1, transform=transf, 
                  levels=levels, add_colorbar=False)
    ax.set_title(varnames_plot[v], fontsize=fs)
    regionmask.defined_regions.ar6.land.plot(line_kws=dict(color='black', linewidth=1), 
                                                 ax=ax, add_label=False, projection=transf)

cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.6]) # left bottom width height
cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
cbar.set_label('Pearson correlation coefficient', fontsize=fs)
fig.suptitle('(b) Pearson correlation coefficient on anomalies', fontsize=20)

plt.savefig('correlation_maps.png')
