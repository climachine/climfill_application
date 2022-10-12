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
# temperature from obs: DONE
# precipitation from obs: DONE
# lat, lon: to be calc from regridded
# permafrost extent: not yet downloaded
# above-ground biomass: not yet downloaded
# topography: not yet downloaded
# surface net radiation from ESA CCI cloud: validation by DWD awaiting

import numpy as np
import xarray as xr
import xesmf as xe

landclimstoragepath = '/net/exo/landclim/data/dataset/'
esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
timeslice = slice('1995','2020')
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.75,90, 0.5)), # same as cdo_weights.nc
                     'lon': (['lon'], np.arange(-180, 180,0.5))})

# flags for running scripts
ifire = False
ilstpre = False
ilstpost = False
isnow = False
ism = False
itws = False
iprecip = False
iobs = True

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

# land cover
#if ilandcover:
#    filepath = landclimstoragepath + \
#               'ESA-CCI-LC_Land-Cover-Maps/v2.0.7cds/300m_plate-carree_1y/original/'
#    data = xr.open_mfdataset(f'{filepath}*.nc')['lccs_class']
#regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) #gives TypeError: buffer is too small for requested array
#data = regridder(data)
#cdo remaplaf,r360x180 /net/exo/landclim/data/dataset/ESA-CCI-LC_Land-Cover-Maps/v2.0.7cds/300m_plate-carree_1y/original/ESACCI-LC-L4-LCCS-Map-300m-P1Y-2003-v2.0.7cds.nc test.nc # remaplaf is for land cover classes; fulden says consider aggregating the classes for more useful results

# land surface temperature
# missing values: cloud cover, in whole month? TODO investigate
#if ilst:
#    filepath = landclimstoragepath + \
#               'ESA-CCI-LST_MULTISENSOR-IRCDR/v2.00/0.01deg_lat-lon_1m/original/'
#    import IPython; IPython.embed()
#    data = xr.open_mfdataset(f'{lstpath}*DAY*.nc')['lst']
#data = data.sel(time=timeslice)
#regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
#data = regridder(data) #TypeError: buffer is too small for requested array
# with cdo
# create weights
# cdo genbil,cdo_grid_description.txt /net/exo/landclim/data/dataset/ESA-CCI-LST_MULTISENSOR-IRCDR/v2.00/0.01deg_lat-lon_1m/original/ESACCI-LST-L3S-LST-IRCDR_-0.01deg_1MONTHLY_DAY-20200901000000-fv2.00.nc cdo_weights.nc
# loop over files
# cdo remap,r720x360,cdo_weights.nc /net/exo/landclim/data/dataset/ESA-CCI-LST_MULTISENSOR-IRCDR/v2.00/0.01deg_lat-lon_1m/original/ESACCI-LST-L3S-LST-IRCDR_-0.01deg_1MONTHLY_DAY-20200901000000-fv2.00.nc /net/so4/landclim/bverena/large_files/climfill_esa/lsttest.nc
if ilstpre:
    import glob
    import os
    filepath = landclimstoragepath + \
               'ESA-CCI-LST_MULTISENSOR-IRCDR/v2.00/0.01deg_lat-lon_1m/original/'
    filenames = glob.glob(f'{filepath}*DAY*.nc')
    if not os.path.exists('cdo_weights.nc'):
        os.system(f'cdo genbil,cdo_grid_description.txt {filenames[0]} cdo_weights.nc')

    # regrid all files individually
    for filename in filenames:
        if not os.path.exists(f'{esapath}lst_monthly/lst_{filename.split("-")[-2][:8]}.nc'):
            print(f'{esapath}lst_monthly/lst_{filename.split("-")[-2][:8]}.nc')
            os.system(f'cdo -P 10 remap,r720x360,cdo_weights.nc {filename} {esapath}lst_monthly/lst_{filename.split("-")[-2][:8]}.nc')

# TODO from daily
# note that 1995-08-01 is manually removed after ilstpre because cdo regrids wrongly
if ilstpost:
    # second part: read in files, coordinate transformation, save as one file 
    data = xr.open_mfdataset(f'{esapath}lst_monthly/lst_*.nc')['lst']

    # convert coords
    data.coords['lon'] = (data.coords['lon'] + 180) % 360 - 180
    data = data.sortby('lon')

    data = data.sel(time=timeslice)

    data = data.to_dataset(name='surface_temperature')
    data.to_netcdf(f'{esapath}surface_temperature.nc')

# snow water equivalent
if isnow:
    snowpath = landclimstoragepath + 'ESA-CCI-Snow_SWE/v2.0/0.1deg_lat-lon_1d/original/'
    data = xr.open_mfdataset(f'{snowpath}*.nc')['swe']

    assert np.isnan(data).sum().values.item() == 0 # no missing values

    data = data.where(data >= 0, 0) # all masks (ocean, mountain, glacier) to zero snow

    data = data.resample(time='MS').mean()
    data = data.sel(time=timeslice)

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='snow_water_equivalent')
    data.to_netcdf(f'{esapath}snow_water_equivalent.nc')

# soil moisture:
# missing values: below dense vegetation, swaths(?)
# TODO wait for MH to reply which values are taken into monthly vals
# or if my own monthly agg is necessary
if ism:
    filepath = landclimstoragepath + \
    'ESA-CCI-SM_combined/v07.1/0.25deg_lat-lon_1d/processed/netcdf/'
    data = xr.open_mfdataset(f'{filepath}*.nc')['sm']

    data = data.sel(time=timeslice)

    # create mask for less than 21 days of data
    mask = np.isnan(data).astype(float).resample(time='MS').sum()
    mask = mask > 21
    data = data.resample(time='MS').mean()
    data = data.where(mask)
    import IPython; IPython.embed()

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='soil_moisture')
    data.to_netcdf(f'{esapath}soil_moisture.nc')

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

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='precipitation')
    data.to_netcdf(f'{esapath}precipitation.nc')

# from obs
# missing values: ocean
# TODO: set some values to nan because define threshold below obs
if iobs:
    filepath = landclimstoragepath + \
    'CRUTS/v4.06/0.5deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*tmp*.nc')['tmp']

    data = data.sel(time=timeslice)
    data = data.resample(time='MS').mean() # M-mid to MS

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='temperature_obs')
    data.to_netcdf(f'{esapath}temperature_obs.nc')

    filepath = landclimstoragepath + \
    'CRUTS/v4.02/0.5deg_lat-lon_1m/original/'
    data = xr.open_mfdataset(f'{filepath}*pre*.nc')['pre']

    data = data.sel(time=timeslice)
    data = data.resample(time='MS').mean() # M-mid to MS

    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    data = data.to_dataset(name='precipitation_obs')
    data.to_netcdf(f'{esapath}precipitation_obs.nc')
