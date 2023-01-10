# run this in keras_env
# note: if run several times in different windows, error may occurr:
# ValueError: ESMC_FieldRegridStoreFile() failed with rc = 22.    Please check the log files (named "*E
# might be related to xesmf trying to store weights file at the same time
# solution: run python script from different folder (e.g. ../preprocess...)
"""
NAMESTRING
"""

from glob import glob
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-89.75,90, 0.5)),
                     'lon': (['lon'], np.arange(-180, 180,0.5))})

filepath = ('/net/exo/landclim/data/dataset/MODIS_Land-Surface-'
            'Temperature-MYD11C1/v006/0.05deg_lat-lon_1d/original/')
years = list(np.arange(2002,2021))

##################### surface temp ########################################
#for year in years:
#    print(year)
#
#    # get timestamp from file
#    filenames = f'{filepath}MYD11C1.A{year}*.hdf'
#    filenames = glob(filenames)
#    filenames = sorted(filenames)
#    dates = [filename[112:119] for filename in filenames]
#    times = []
#    pattern = '%Y%j'
#    for date in dates:
#        times.append(datetime.strptime(date, pattern))
#
#    # open data and reformat
#    data = xr.open_mfdataset(filenames, concat_dim='time', combine='nested')
#    data = data['LST_Day_CMG'].load()
#    time = data.dims[0]
#    ydim = data.dims[1]
#    xdim = data.dims[2]
#    data = data.assign_coords(**{ydim: np.arange(90,-90,-0.05)})
#    data = data.assign_coords(**{xdim: np.arange(-180,180,0.05)})
#    data = data.assign_coords(**{time: times})
#    data = data.rename({ydim:'lat',xdim:'lon'})
#    data = data.load()
#
#    # regrid
#    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
#    data = regridder(data)
#
#    # convert kelvin to celcius
#    data = data - 273.15
#
#    # fill missing days in data as fully nan
#    data = data.resample(time='D').asfreq()
#
#    # create mask for less than 15 days of data
#    mask = np.isnan(data).astype(float).resample(time='MS').sum() # how many nans
#    mask = mask <= 15 # where less than 15 nans
#    data = data.resample(time='MS').mean()
#    data = data.where(mask) # keep where less than 15 nans
#
#    # save
#    data = data.to_dataset(name='surface_temperature')
#    data.to_netcdf(f'{esapath}modis_yearly/surface_temperature_MODIS_{year}.nc')


################## DTR #######################################################
#for year in years:
#    print(year)
#
#    # get timestamp from file
#    filenames = f'{filepath}MYD11C1.A{year}*.hdf'
#    filenames = glob(filenames)
#    filenames = sorted(filenames)
#    dates = [filename[112:119] for filename in filenames]
#    times = []
#    pattern = '%Y%j'
#    for date in dates:
#        times.append(datetime.strptime(date, pattern))
#
#    # open data and reformat
#    data = xr.open_mfdataset(filenames, concat_dim='time', combine='nested')
#    data = data['LST_Day_CMG'] - data['LST_Night_CMG']
#    data = data.load()
#    time = data.dims[0]
#    ydim = data.dims[1]
#    xdim = data.dims[2]
#    data = data.assign_coords(**{ydim: np.arange(90,-90,-0.05)})
#    data = data.assign_coords(**{xdim: np.arange(-180,180,0.05)})
#    data = data.assign_coords(**{time: times})
#    data = data.rename({ydim:'lat',xdim:'lon'})
#    data = data.load()
#
#    # regrid
#    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
#    data = regridder(data)
#
#    # fill missing days in data as fully nan
#    data = data.resample(time='D').asfreq()
#
#    # create mask for less than 15 days of data
#    mask = np.isnan(data).astype(float).resample(time='MS').sum() # how many nans
#    mask = mask <= 15 # where less than 15 nans
#    data = data.resample(time='MS').mean()
#    data = data.where(mask) # keep where less than 15 nans
#
#    # save
#    data = data.to_dataset(name='diurnal_temperature_range')
#    data.to_netcdf(f'{esapath}modis_yearly/diurnal_temperature_range_MODIS_{year}.nc')

##################### surface temp init ########################################
for year in years:
    print(year)

    # get timestamp from file
    filenames = f'{filepath}MYD11C1.A{year}*.hdf'
    filenames = glob(filenames)
    filenames = sorted(filenames)
    dates = [filename[112:119] for filename in filenames]
    times = []
    pattern = '%Y%j'
    for date in dates:
        times.append(datetime.strptime(date, pattern))

    # open data and reformat
    data = xr.open_mfdataset(filenames, concat_dim='time', combine='nested')
    data = data['LST_Day_CMG'].load()
    time = data.dims[0]
    ydim = data.dims[1]
    xdim = data.dims[2]
    data = data.assign_coords(**{ydim: np.arange(90,-90,-0.05)})
    data = data.assign_coords(**{xdim: np.arange(-180,180,0.05)})
    data = data.assign_coords(**{time: times})
    data = data.rename({ydim:'lat',xdim:'lon'})
    data = data.load()

    # regrid
    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
    data = regridder(data)

    # convert kelvin to celcius
    data = data - 273.15

    # fill missing days in data as fully nan
    data = data.resample(time='D').asfreq()

    # create mask for less than 15 days of data
    mask = np.isnan(data).astype(float).resample(time='MS').sum() # how many nans
    mask = mask > 15 # where less than 15 nans
    data = data.resample(time='MS').mean()
    data = data.where(mask) # keep where less than 15 nans

    # save
    data = data.to_dataset(name='surface_temperature')
    data.to_netcdf(f'{esapath}modis_yearly/surface_temperature_MODIS_{year}_init.nc')

################## DTR init ####################################################
#for year in years:
#    print(year)
#
#    # get timestamp from file
#    filenames = f'{filepath}MYD11C1.A{year}*.hdf'
#    filenames = glob(filenames)
#    filenames = sorted(filenames)
#    dates = [filename[112:119] for filename in filenames]
#    times = []
#    pattern = '%Y%j'
#    for date in dates:
#        times.append(datetime.strptime(date, pattern))
#
#    # open data and reformat
#    data = xr.open_mfdataset(filenames, concat_dim='time', combine='nested')
#    data = data['LST_Day_CMG'] - data['LST_Night_CMG']
#    data = data.load()
#    time = data.dims[0]
#    ydim = data.dims[1]
#    xdim = data.dims[2]
#    data = data.assign_coords(**{ydim: np.arange(90,-90,-0.05)})
#    data = data.assign_coords(**{xdim: np.arange(-180,180,0.05)})
#    data = data.assign_coords(**{time: times})
#    data = data.rename({ydim:'lat',xdim:'lon'})
#    data = data.load()
#
#    # regrid
#    regridder = xe.Regridder(data, ds_out, 'bilinear', reuse_weights=False) 
#    data = regridder(data)
#
#    # fill missing days in data as fully nan
#    data = data.resample(time='D').asfreq()
#
#    # create mask for less than 15 days of data
#    mask = np.isnan(data).astype(float).resample(time='MS').sum() # how many nans
#    mask = mask > 15 # where less than 15 nans
#    data = data.resample(time='MS').mean()
#    data = data.where(mask) # keep where less than 15 nans
#
#    # save
#    data = data.to_dataset(name='diurnal_temperature_range')
#    data.to_netcdf(f'{esapath}modis_yearly/diurnal_temperature_range_MODIS_{year}_init.nc')
