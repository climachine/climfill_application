"""
create ESA-CCI datacube.

from MH email:

- GPM_IMERG_L3/v06/0.1deg_lat-lon_1m/original
-> aktualisiert

- ESA-CCI-LST_MULTISENSOR-IRCDR/v2.00/0.01deg_lat-lon_1m/original/
-> neu

- ESA-CCI-Fire_AVHRR-LTDR/v1.1/0.25deg_lat-lon_1m/original/
-> neu

- ESA-CCI-Snow_SWE/v2.0/0.1deg_lat-lon_1d/original/
-> neu (hier wechselt der Sensor über die Zeit, von SMMR-NIMBUS7 zu SSMI-DMSP zu SSMIS-DMSP)

- ESA-CCI-LC_Land-Cover-Maps/v2.0.7cds/300m_plate-carree_1y/original
-> schon vorhanden

- GRACE/rl06v2/0.5deg_lat-lon_1m/original/
-> schon vorhanden
"""

### dataset included action list:
# fire: DONE
# land cover: regridding problems
# land surface temperature: monthly regridding myself?
# snow water equivalent: DONE
# soil moisture: DONE
# terrestrial water storage: DONE
# precipitation: DONE
# temperature from obs: DONE, use n_obs for further deleting?
# precipitation from obs: DONE, use n_obs for further deleting?
# lat, lon: DONE
# permafrost extent: not yet downloaded
# above-ground biomass: not yet downloaded
# topography: not yet downloaded (suggestion: NASA SRTM Van Zyl, 2001, Guisan, 1999)
# surface net radiation from ESA CCI cloud: validation by DWD awaiting, not yet downloaded

import numpy as np
import xarray as xr
import xesmf as xe

landclimstoragepath = '/net/exo/landclim/data/dataset/'
esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
timeslice = slice('1995','2020')
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.75,90, 0.5)), # same as cdo_weights.nc
                     'lon': (['lon'], np.arange(-180, 180,0.5))})

# flags for running scripts
itopo = False
ifire = False
ipermafrost = False
ilandcover = False
ilstpre = False
ilstpost = False
ilstpostinit = False
idtr = False
idtrinit = False
isnowpre = False
isnowpreinit = False
isnowpost = False
isnowpostinit = False
ism = False
isminit = False
itws = False
iprecip = False
iobs = True
iobsinit = False
inetrad = False

# topo
if itopo:
    data = xr.open_mfdataset(f'{esapath}topography.nc')

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data.to_netcdf(f'{esapath}topography.nc')

# fire
if ifire:
    filepath = landclimstoragepath + \
               'ESA-CCI-Fire_AVHRR-LTDR/v1.1/0.25deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*.nc')['burned_area']
    data = data.sel(time=timeslice)

    assert np.isnan(data).sum().values.item() == 0 # no missing values

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='burned_area')
    data.to_netcdf(f'{esapath}burned_area.nc')

# permafrost
if ipermafrost:
    filepath = landclimstoragepath + \
               'ESA-CCI-Permafrost/v3.0/1km_polar-stereographic_1y/processed/regrid0.5deg/'
    data = xr.open_mfdataset(f'{filepath}*.nc')['PFR']

    data = data.sel(time=timeslice)

    # assumption: since "perma", frozen whole year
    data = data.resample(time='MS').ffill()

    data = data.to_dataset(name='permafrost')
    data.to_netcdf(f'{esapath}permafrost.nc')

# land cover
if ilandcover:
    filepath = landclimstoragepath + \
               'ESA-CCI-LC_Land-Cover-Maps/v2.0.7cds/300m_plate-carree_1y/processed/regrid0.5deg/'
    data = xr.open_mfdataset(f'{filepath}*.nc')['lccs_class']

    data = data.sel(time=timeslice)

    data = data.resample(time='MS').ffill()

    # add missing years at the end by copying 2015 forward
    # needs to happen after resample otherwise year 2020 monthly is lost
    data = data.reindex(time=xr.date_range(start='1995-01-01', end='2021-01-01', freq='MS'), method='ffill')

    # remove 2021-01-01 again
    data = data.sel(time=timeslice)

    # regridding not necesssary, already done by MH
    # update: regridding necessary since not same grid description was used
    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='landcover')
    data.to_netcdf(f'{esapath}landcover.nc')

# land surface temperature
# missing values: cloud cover, swaths
if ilstpre:
    import glob
    import os
    filepath = landclimstoragepath + \
               'ESA-CCI-LST_MULTISENSOR-IRCDR/v2.00/0.01deg_lat-lon_1d/original/'
    filenames = glob.glob(f'{filepath}*.nc')
    if not os.path.exists('cdo_weights.nc'): 
        os.system(f'cdo genbil,cdo_grid_description.txt {filenames[0]} cdo_weights.nc')

    # regrid all files individually
    # Remap weights not used because missing values are different for each timestep
    # https://code.mpimet.mpg.de/projects/cdo/wiki/FAQ (search "remap weights")
    # although error "weights not used" is faster with weights
    for filename in filenames:
        savename = f'{esapath}lst_daily/lst_{filename.split("-")[-2][:8]}_{filename.split("-")[-3].split("_")[-1]}.nc'
        if not os.path.exists(f'{savename}'):
            cmd = f'cdo -P 20 remap,r720x360,cdo_weights.nc -selname,lst {filename} {savename} \n'
            with open('run_lst_regrid_cdo.sh','a') as file:
                file.write(cmd)

# note that 1995-08-01 is manually removed after ilstpre because cdo regrids wrongly
if ilstpost:
    # second part: read in files, coordinate transformation, save as one file 
    #data = xr.open_mfdataset(f'{esapath}lst_monthly/lst_*.nc')['lst'] # OLD
    data = xr.open_mfdataset(f'{esapath}lst_daily/lst*DAY.nc')['lst']

    # convert coords
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')

    # convert kelvin to celcius
    data = data - 273.15

    # fill missing days in data as fully nan
    data = data.resample(time='D').asfreq()

    # create mask for less than 15 days of data
    mask = np.isnan(data).astype(float).resample(time='MS').sum() # how many nans
    import IPython; IPython.embed()
    mask = mask <= 5 # where less than 15 nans
    data = data.resample(time='MS').mean()
    data = data.where(mask) # keep where less than 15 nans

    data = data.sel(time=timeslice)

    data = data.to_dataset(name='surface_temperature')
    data.to_netcdf(f'{esapath}surface_temperature_tmp.nc')

if ilstpostinit:
    # second part: read in files, coordinate transformation, save as one file 
    #data = xr.open_mfdataset(f'{esapath}lst_monthly/lst_*.nc')['lst'] # OLD
    data = xr.open_mfdataset(f'{esapath}lst_daily/lst*DAY.nc')['lst']

    # convert coords
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')

    # convert kelvin to celcius
    data = data - 273.15

    # fill missing days in data as fully nan
    data = data.resample(time='D').asfreq()

    #  take all vars for init guess
    data = data.resample(time='MS').mean()

    data = data.sel(time=timeslice)

    data = data.to_dataset(name='surface_temperature')
    data.to_netcdf(f'{esapath}surface_temperature_init.nc')

if idtr:
    day = xr.open_mfdataset(f'{esapath}lst_daily/lst*DAY.nc')['lst'] #DEBUG
    night = xr.open_mfdataset(f'{esapath}lst_daily/lst*NIGHT.nc')['lst'] # DEBUG

    # calculate DTR
    data = np.abs(day - night)

    # convert coords
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')

    # fill missing days in data as fully nan
    data = data.resample(time='D').asfreq()

    # create mask for less than 15 days of data
    mask = np.isnan(data).astype(float).resample(time='MS').sum()
    mask = mask <= 15
    data = data.resample(time='MS').mean()
    data = data.where(mask)

    data = data.sel(time=timeslice)

    data = data.to_dataset(name='diurnal_temperature_range')
    data.to_netcdf(f'{esapath}diurnal_temperature_range.nc')

if idtrinit:
    day = xr.open_mfdataset(f'{esapath}lst_daily/lst*DAY.nc')['lst'] #DEBUG
    night = xr.open_mfdataset(f'{esapath}lst_daily/lst*NIGHT.nc')['lst'] # DEBUG

    # calculate DTR
    data = np.abs(day - night)

    # convert coords
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')

    # fill missing days in data as fully nan
    data = data.resample(time='D').asfreq()

    # monthly agg
    data = data.resample(time='MS').mean()

    data = data.sel(time=timeslice)

    data = data.to_dataset(name='diurnal_temperature_range')
    data.to_netcdf(f'{esapath}diurnal_temperature_range_init.nc')

# snow water equivalent
#if isnow:
#    snowpath = landclimstoragepath + 'ESA-CCI-Snow_SWE/v2.0/0.1deg_lat-lon_1d/original/'
#    data = xr.open_mfdataset(f'{snowpath}*.nc')['swe']
#     
#    # all negative values are masks (ocean, mountain, glacier) set to nan
#    mask = data >= 0 # keep those values
#
#    # resample to month: checked that all months have 21 valid days everywhere (bec only constant missingness)
#    data = data.resample(time='MS').mean()
#    mask = mask.resample(time='MS').mean()
#    data = data.sel(time=timeslice)
#
#    # set missing values (bec all in summer from missing files, checked)
#    # to 0 because they imply NO snow
#    data = data.fillna(0)
#
#    # insert missing values from mountains and glaciers again
#    data = data.where(mask)
#
#    # set SH values to 0, not NAN, because we assume no snow in SH
#    data = data.where(data.lat > 0, 0)
#
#    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
#    data = regridder(data)
#
#    data = data.to_dataset(name='snow_water_equivalent')
#    data.to_netcdf(f'{esapath}snow_water_equivalent.nc')

#if isnowpre:
#    import glob
#    import os
#    filepath = landclimstoragepath + \
#               'ESA-CCI-Snow_SCFG/v2.0/5km_lat-lon_1d/original/AVHRR/'
#    filenames = glob.glob(f'{filepath}*.nc')
#    if not os.path.exists('cdo_weights_snow.nc'): 
#        os.system(f'cdo genbil,cdo_grid_description.txt {filenames[0]} cdo_weights_snow.nc')
#
#    # regrid all files individually
#    # Remap weights not used because missing values are different for each timestep
#    # https://code.mpimet.mpg.de/projects/cdo/wiki/FAQ (search "remap weights")
#    # although error "weights not used" is faster with weights
#    for filename in filenames:
#        savename = f'{esapath}snow_daily/snow_{filename.split("/")[-1][:8]}.nc'
#        if not os.path.exists(f'{savename}'):
#            cmd = f'cdo -P 20 remap,r720x360,cdo_weights_snow.nc -selname,scfg {filename} {savename} \n'
#            with open('run_snow_regrid_cdo.sh','a') as file:
#                file.write(cmd)

# snow cover fraction
if isnowpre: 
    filepath = landclimstoragepath + \
               'ESA-CCI-Snow_SCFG/v2.0/5km_lat-lon_1d/original/AVHRR/'
    years = np.arange(1995,2021)
    
    for year in years:
        print(year)
        try:
            data = xr.open_mfdataset(f'{filepath}/{year}*.nc')['scfg']
        except OSError as e:
            print(e)
            continue

        # fill missing days in data as fully nan
        data = data.resample(time='D').asfreq()

        # missing values are above 200
        data = data.where(data <= 100, np.nan)

        # create mask for less than 15 days of data
        mask = np.isnan(data).astype(float).resample(time='MS').sum()
        mask = mask <= 15
        data = data.resample(time='MS').mean()
        data = data.where(mask)

        # regrid
        regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
        data = regridder(data)

        # save
        data = data.to_dataset(name='snow_cover_fraction')
        data.to_netcdf(f'{esapath}snow_yearly/snow_cover_fraction_{year}.nc')

if isnowpreinit: 
    filepath = landclimstoragepath + \
               'ESA-CCI-Snow_SCFG/v2.0/5km_lat-lon_1d/original/AVHRR/'
    years = np.arange(1995,2021)
    
    for year in years:
        print(year)
        try:
            data = xr.open_mfdataset(f'{filepath}/{year}*.nc')['scfg']
        except OSError as e:
            print(e)
            continue

        # fill missing days in data as fully nan
        data = data.resample(time='D').asfreq()

        # missing values are above 200
        data = data.where(data <= 100, np.nan)

        # mean over all avail points
        data = data.resample(time='MS').mean()

        # regrid
        regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
        data = regridder(data)

        # save
        data = data.to_dataset(name='snow_cover_fraction')
        data.to_netcdf(f'{esapath}snow_yearly/snow_cover_fraction_{year}_init.nc')

if isnowpost:
    data = xr.open_mfdataset(f'{esapath}snow_yearly/*[0-9].nc')
    data.to_netcdf(f'{esapath}snow_cover_fraction.nc')

if isnowpostinit:
    data = xr.open_mfdataset(f'{esapath}snow_yearly/*init.nc')
    data.to_netcdf(f'{esapath}snow_cover_fraction_init.nc')

# soil moisture:
# missing values: below dense vegetation, swaths(?)
if ism:
    filepath = landclimstoragepath + \
    'ESA-CCI-SM_combined/v07.1/0.25deg_lat-lon_1d/processed/netcdf/'
    data = xr.open_mfdataset(f'{filepath}*.nc')['sm']

    data = data.sel(time=timeslice)

    # create mask for less than 15 days of data
    mask = np.isnan(data).astype(float).resample(time='MS').sum()
    mask = mask <= 15
    data = data.resample(time='MS').mean()
    data = data.where(mask)

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='soil_moisture')
    data.to_netcdf(f'{esapath}soil_moisture.nc')

# intial guess insted interp is any measured value this month, mean
if isminit:
    filepath = landclimstoragepath + \
    'ESA-CCI-SM_combined/v07.1/0.25deg_lat-lon_1d/processed/netcdf/'
    data = xr.open_mfdataset(f'{filepath}*.nc')['sm']

    data = data.sel(time=timeslice)

    # mean of all available variables
    data = data.resample(time='MS').mean()

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='soil_moisture')
    data.to_netcdf(f'{esapath}soil_moisture_init.nc')

# terrestrial water storage
# missing values: monthly slices missing
if itws:
    filepath = landclimstoragepath + \
    'GRACE/rl06v2/0.5deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*.nc')['lwe_thickness']

    data = data.sel(time=timeslice)
    data = data.resample(time='MS').mean() # some slices missing

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='terrestrial_water_storage')
    data.to_netcdf(f'{esapath}terrestrial_water_storage.nc')

# precipitation
# missing values: in high latitudes
if iprecip:
    filepath = landclimstoragepath + \
    'GPM_IMERG_L3/v06/0.1deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*.nc4')['precipitation']

    # convert DatetimeJulian to datetimeindex
    data['time'] = data.indexes['time'].to_datetimeindex()

    data = data.sel(time=timeslice)

    # convert from [mm/hour] to [mm/month]
    data = data*24*30.5

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='precipitation')
    data.to_netcdf(f'{esapath}precipitation.nc')

# from obs
# missing values: ocean
if iobs:
    # temperature
    filepath = landclimstoragepath + \
    'CRUTS/v4.06/0.5deg_lat-lon_1m/original/'
    maskpath = landclimstoragepath + \
    'CRUTS_stations/v3.26/0.5deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*tmp*.nc')['tmp']
    mask = xr.open_mfdataset(f'{maskpath}*tmp.st0.nc')['st0']

    data = data.resample(time='MS').mean() # M-mid to MS
    mask = mask.resample(time='MS').mean() # M-mid to MS

    # sel timeperiod
    #data = data.sel(time=slice('1995','2013')) #DEBUG
    #mask = mask.sel(time=slice('1995','2013')) #DEBUG
    data = data.sel(time=timeslice).load()
    mask = mask.sel(time=timeslice).load()

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)
    mask = regridder(mask)

    # elongate mask DEBUG
    mask = mask.reindex(time=data.time, method='ffill')

    # mask areas with 0 station influence
    #data = data.where(mask != 0, np.nan)

    # mask areas with 0 station count
    data = data.where(mask != 0, np.nan)

    data = data.to_dataset(name='temperature_obs')
    data.to_netcdf(f'{esapath}temperature_obs.nc')

    # precip
    filepath = landclimstoragepath + \
    'CRUTS/v4.06/0.5deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*pre*.nc')['pre']
    mask = xr.open_mfdataset(f'{maskpath}*pre.st0.nc')['st0']

    data = data.resample(time='MS').mean().load() # M-mid to MS
    mask = mask.resample(time='MS').mean().load() # M-mid to MS

    # sel timeperiod
    #data = data.sel(time=slice('1995','2013')) #DEBUG
    #mask = mask.sel(time=slice('1995','2013')) #DEBUG
    data = data.sel(time=timeslice)
    mask = mask.sel(time=timeslice)

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)
    mask = regridder(mask)

    # elongate mask DEBUG
    mask = mask.reindex(time=data.time, method='ffill')

    # mask areas with 0 station influence
    #data = data.where(mask != 0, np.nan)

    # mask areas with 0 station count
    data = data.where(mask != 0, np.nan)

    data = data.to_dataset(name='precipitation_obs')
    data.to_netcdf(f'{esapath}precipitation_obs.nc')

if iobsinit:
    # temperature
    filepath = landclimstoragepath + \
    'CRUTS/v4.06/0.5deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*tmp*.nc')['tmp']

    data = data.sel(time=timeslice)
    data = data.resample(time='MS').mean() # M-mid to MS

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='temperature_obs')
    data.to_netcdf(f'{esapath}temperature_obs_init.nc')

    # precip
    filepath = landclimstoragepath + \
    'CRUTS/v4.02/0.5deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*pre*.nc')['pre']

    data = data.sel(time=timeslice)
    data = data.resample(time='MS').mean() # M-mid to MS

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='precipitation_obs')
    data.to_netcdf(f'{esapath}precipitation_obs_init.nc')

if inetrad:
    filepath = landclimstoragepath + \
               'ESA-CCI-Cloud_AVHRR-PM/v3.0/0.5deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*.nc')
    import IPython; IPython.embed()
