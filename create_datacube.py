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
# fire: ready
# land cover: ready
# land surface temperature: ready
# snow water equivalent: ready
# soil moisture: ready
# terrestrial water storage: ready
# precipitation: ready
# temperature from obs: ready
# precipitation from obs: ready
# lat, lon: to be calc from regridded
# permafrost extent: not yet downloaded
# above-ground biomass: not yet downloaded
# topography: not yet downloaded
# surface net radiation from ESA CCI cloud: email sent

import xarray as xr

landclimstoragepath = '/net/exo/landclim/data/dataset/'
firepath = 'ESA-CCI-Fire_AVHRR-LTDR/v1.1/0.25deg_lat-lon_1m/original/'
landcoverpath = 'ESA-CCI-LC_Land-Cover-Maps/v2.0.7cds/300m_plate-carree_1y/original'
# and so on but first check whether one script per variable is better structure

fire = xr.open_mfdataset(f'{firepath}*.nc')
