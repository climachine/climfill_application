"""
NAMESTRING
"""

from datetime import datetime
import argparse
import xarray as xr
from sklearn.cluster import MiniBatchKMeans

parser = argparse.ArgumentParser()
parser.add_argument('--testcase', '-t', dest='testcase', type=str)
parser.add_argument('--n_clusters', '-c', dest='n_clusters', type=int)
parser.add_argument('--set', '-s', dest='veriset', type=str)
parser.set_defaults(veriset=None)
args = parser.parse_args()
testcase = args.testcase
n_clusters = args.n_clusters
veriset = args.veriset

esapath = '/net/so4/landclim/bverena/large_files/climfill_esa/'

# read data
print(f'{datetime.now()} read data...')
if veriset is None:
    data = xr.open_dataset(f'{esapath}{testcase}/datatable.nc').to_array().T
    mask = xr.open_dataset(f'{esapath}{testcase}/masktable.nc').to_array().T
else:
    data = xr.open_dataset(f'{esapath}{testcase}/verification/set{veriset}/datatable.nc').to_array().T
    mask = xr.open_dataset(f'{esapath}{testcase}/verification/set{veriset}/masktable.nc').to_array().T

# clustering
print(f'{datetime.now()} clustering...')
labels = MiniBatchKMeans(n_clusters=n_clusters, verbose=0, batch_size=1000, 
                         random_state=0).fit_predict(data)

for c in range(n_clusters):

    print(f'{datetime.now()} create cluster {c}...')

    # select cluster
    data_c = data[labels == c,:]
    mask_c = mask[labels == c,:]

    # to dataset
    data_c = data_c.to_dataset(name='data')
    mask_c = mask_c.to_dataset(name='data')

    # save data
    if veriset is None:
        data_c.to_netcdf(f'{esapath}{testcase}/clusters/datacluster_init_c{c:02d}.nc')
        
        # save mask
        # mask needs explicit bool otherwise lostmask is saved as int (0,1) and 
        # numpy selects all datapoints as missing in imputethis since 0 and 1 
        # are treated as true and no false are found
        mask_c.to_netcdf(f'{esapath}{testcase}/clusters/maskcluster_init_c{c:02d}.nc', 
                         encoding={'data':{'dtype':'bool'}}) 
    else:
        data_c.to_netcdf(f'{esapath}{testcase}/verification/set{veriset}/clusters/datacluster_init_c{c:02d}.nc')
        mask_c.to_netcdf(f'{esapath}{testcase}/verification/set{veriset}/clusters/maskcluster_init_c{c:02d}.nc',
                         encoding={'data':{'dtype':'bool'}}) 
