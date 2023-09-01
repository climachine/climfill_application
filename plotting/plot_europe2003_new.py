"""
NAMESTRING
"""

import matplotlib.pyplot as plt
import xarray as xr

# open files
year = 2003
largefilepath = '/net/so4/landclim/bverena/large_files/'
savepath = f'{largefilepath}2003_2020/'
orig = xr.open_dataarray(f'{savepath}datacube_original_2003_2009.nc')
lost = xr.open_dataarray(f'{savepath}datacube_lost_2003_2009.nc')
fill = xr.open_dataarray(f'{savepath}data_climfilled_{year}.nc')

# select 2003
timeslice = slice(f'{year}-01-01',f'{year}-09-01')
orig = orig.sel(time=timeslice).load()
lost = lost.sel(time=timeslice).load()
fill = fill.sel(time=timeslice).load()

# select europe
orig = orig.sel(lat=slice(30,60), lon=slice(-10,30))
lost = lost.sel(lat=slice(30,60), lon=slice(-10,30))
fill = fill.sel(lat=slice(30,60), lon=slice(-10,30))

# convert precip to mm/day
#orig.loc['tp'] = orig.loc['tp']*100
#lost.loc['tp'] = lost.loc['tp']*100
#fill.loc['tp'] = fill.loc['tp']*100

# convert temp to C
orig.loc['skt'] = orig.loc['skt']-273.15
lost.loc['skt'] = lost.loc['skt']-273.15
fill.loc['skt'] = fill.loc['skt']-273.15

# create regional mean
orig = orig.mean(dim=('lat','lon'))
lost = lost.mean(dim=('lat','lon'))
fill = fill.mean(dim=('lat','lon'))

# select skt
orig_skt = orig.sel(variable='skt')
lost_skt = lost.sel(variable='skt')
fill_skt = fill.sel(variable='skt')
orig_swvl1 = orig.sel(variable='swvl1')
lost_swvl1 = lost.sel(variable='swvl1')
fill_swvl1 = fill.sel(variable='swvl1')

col_fill = 'darkgrey'
col_miss = 'indianred'
col_ismn = 'black'
col_erag = 'black'

# plot
fs = 15
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

im = orig_skt.plot(ax=ax1, color='black', label='ERA5')
lost_skt.plot(ax=ax1, color='darkgrey', label='Satellite-observable ERA5')
fill_skt.plot(ax=ax1, color='indianred', label='CLIMFILL')

im = orig_swvl1.plot(ax=ax2, color='black', label='ERA5')
lost_swvl1.plot(ax=ax2, color='darkgrey', label='Satellite-observable ERA5')
fill_swvl1.plot(ax=ax2, color='indianred', label='CLIMFILL')

#ax.set_ylabel('surface layer soil moisture, \ndeviations from 1996-2020 \naverage $[m^{3}\;m^{-3}]$', fontsize=fs)
ax1.set_ylabel('Daily mean \nground temperature [K]')
ax1.set_xticklabels([])
ax1.set_xlabel('')
ax2.set_ylabel('Surface layer \nsoil moisture $[m^{3}\;m^{-3}]$')
#ax.set_xlabel('time')#, fontsize=fs)
#fig.suptitle('(b) Northern Extratropics', fontsize=fs)
#ax.legend(fontsize=fs, loc='lower right')
ax1.legend(loc='lower right')
ax1.set_title('')
ax2.set_title('')
ax1.set_ylim([-5,30])
ax2.set_ylim([0.1,0.35])
#ax.set_ylim([-0.02,0.015])
plt.savefig('europe_2003_3.pdf')
