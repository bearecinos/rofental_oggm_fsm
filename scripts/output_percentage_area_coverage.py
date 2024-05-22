"""
This script produces a raster of % glaciated area coverage per year
 simulation for 3 ssp scenarios.
The percentage of area coverage represents how much a grid cell from the
coarse grid is covered by ice. In other words aggregates pixels from the oggm grid
mask produced when there is ice, elements equal to 1; meaning there is ice.
1) Takes as input data the raster compiled by Hinter_post_processin.py from the working dir
2) Interpolates oggm ice thickness data into a coarser grid. e.g. INCA DEM
grid resolution and Alpine projection!.
3) Formulates a mask of ice thickness > 0 m. for a particular year
and calculates for each pixel of the coarse grid; how much that grid cell is covered by ice.

It outputs a raster with % glaciated area coverage per year per simulation in the right projection.

"""
# Module logger
from __future__ import division
import logging
import argparse
import os
import sys
from configobj import ConfigObj
import xarray as xr
import salem
from pyproj import Transformer
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from oggm import utils

log = logging.getLogger(__name__)

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-simulation_index",
                    type=int,
                    default=0, help="Simulation index")

args = parser.parse_args()

config_file = args.conf
simulation_index = args.simulation_index

config = ConfigObj(os.path.expanduser(config_file))

sys.path.append(config['main_repo_path'])
from tools.mix_gis import interp_with_griddata

working_dir = os.path.join(config['main_repo_path'],
                           'output_data/02_all_rofental')

# Getting the data in working dir
oggm_data_fp = []
topo_data_fp = []

# If more scenarios are un probably we need a list read from a file
for ssp in ['ssp126', 'ssp370', 'ssp585']:
    # Simulations containing ice thickness per ssp for all years merged into a single file.
    experiment_files = 'distributed_data' + ssp + '/all_simulations_merged_for_' + ssp + '.nc'
    # Topo file will be transfer to CEH data folder as is needed by Nathan
    topo_files = 'distributed_data' + ssp + '/all_merged_for_' + ssp + '_topo_data.nc'
    # Constructing the paths
    oggm_path = os.path.join(working_dir, experiment_files)
    topo_path = os.path.join(working_dir, topo_files)
    # Saving them on a list
    oggm_data_fp.append(oggm_path)
    topo_data_fp.append(topo_path)

print('we are processing the following simulation', oggm_data_fp[simulation_index])

# The following lines are uncomment as I changed and save the right projection for
# the INCA.DEM. This process is necessary if your coarse grid does not have the right pyproj.srs
# which needs to be added it to the netcdf file before reading with Salem.

# Got the right code for the Alps projection
# https://epsg.io/31287
# from_online = config['pyproj_Alps']
# inca_original = config['path_coarse_grid_original']
# proj = pyproj.Proj(from_online)
# di = xr.open_dataset(inca_original)
# di.attrs['pyproj_srs'] = proj.srs
# for v in di.variables:
#     di[v].attrs['pyproj_srs'] = proj.srs
# di.to_netcdf(config['path_coarse_grid'])

# Read the coarser grid
inca_fp = config['path_coarse_grid']
# Inca DEM
df_coarse = salem.open_xr_dataset(inca_fp)

# OGGM thickness
df_oggm = salem.open_xr_dataset(oggm_data_fp[simulation_index])
# OGGM topo
df_topo = salem.open_xr_dataset(topo_data_fp[simulation_index])

# Get the grids
grid_oggm = df_oggm.salem.grid
grid_coarse = df_coarse.salem.grid

# we crop the coarser grid to oggm grid bounds (makes the code faster)
sub = df_coarse.salem.subset(grid=grid_oggm, margin=0)
grid_coarse_sub = sub.salem.grid

x_oggm = df_oggm.x.values[:]
y_oggm = df_oggm.y.values[:]

# Lets get the grid size from oggm
dx_oggm = x_oggm[1] - x_oggm[0]
print('oggm has a grid size of', dx_oggm)
xx, yy = np.meshgrid(x_oggm, y_oggm)  # lets gridded

# Get coarse grid coordinates
x_coarse = sub.x.values[:]
y_coarse = sub.y.values[:]
# Transform to projection (maybe is easier with salme) TODO: ask Fabien!
transformer = Transformer.from_crs(grid_oggm.proj.srs,
                                   grid_coarse_sub.proj.srs, always_xy=True)

x_oggm_proj, y_oggm_proj = transformer.transform(xx, yy)

# Some tricks
xstack = np.vstack(x_oggm_proj).T
ystack = np.vstack(y_oggm_proj).T
new_oggm_x = xstack[:, 0]
new_oggm_y = ystack[0, :]

# We grid them again to use griddata
x_oggm_plot, y_oggm_plot = np.meshgrid(new_oggm_x, new_oggm_y)

# Create array to store interpolated data
new_thickness = np.zeros(df_oggm.simulated_thickness.data.shape)

new_thickness = new_thickness[0:4, :, :]

# PARALLELIZATION OF GRIDDATA!
# We need to know how many years there are in the simulation
# to divide the cpus accordingly for the interpolation
years = df_oggm.time[0:4].values.astype(int)
no_yrs = len(years)

dfs = [df_oggm.simulated_thickness.sel(time=year) for year in years]

# We need coordinates for each timestep
# Pool needs this data duplicated to give it to each cpu the same data
x_ori = np.concatenate([[x_oggm_proj]] * no_yrs, axis=0)
y_ori = np.concatenate([[y_oggm_proj]] * no_yrs, axis=0)
x_to_int = np.concatenate([[x_oggm_plot]] * no_yrs, axis=0)
y_to_int = np.concatenate([[y_oggm_plot]] * no_yrs, axis=0)

result = []

if __name__ == '__main__':
    no_of_cpus = no_yrs / 2
    p = ThreadPool(int(no_of_cpus))
    result = p.starmap(interp_with_griddata, zip(dfs, x_ori, y_ori, x_to_int, y_to_int))

    p.close()
    p.join()

for i, array in enumerate(result):
    print(array.shape)
    print(i)
    new_thickness[i, :, :] = array

# Let's make a directory for CEH data and file formats
output_dir = os.path.join(config['main_repo_path'],
                          'output_data/03_data_for_CEH')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

filename, ext = os.path.splitext(os.path.basename(oggm_data_fp[simulation_index]))
filename_f = filename + '_projected_and_aggregated' + ext
netcdf_fp = os.path.join(output_dir, filename_f)
print('Data will be stored here', netcdf_fp)

with utils.ncDataset(netcdf_fp, 'w', format='NETCDF4') as nc:
    nc.createDimension('x', len(new_oggm_x))
    nc.createDimension('y', len(new_oggm_y))
    nc.createDimension('time', len(df_oggm.time[0:4].values))

    v = nc.createVariable('x', 'f4', ('x',), zlib=True)
    v.units = 'm'
    v.long_name = 'x coordinate of projection'
    v.standard_name = 'projection_x_coordinate'
    v[:] = new_oggm_x

    v = nc.createVariable('y', 'f4', ('y',), zlib=True)
    v.units = 'm'
    v.long_name = 'y coordinate of projection'
    v.standard_name = 'projection_y_coordinate'
    v[:] = new_oggm_y

    v = nc.createVariable('time', 'f4', ('time',), zlib=True)
    v.units = 'years'
    v[:] = df_oggm.time[0:4].values

    vn = "simulated_thickness"
    v = nc.createVariable(vn, 'f4', ('time', 'y', 'x',), zlib=True)

df_new = xr.open_dataset(netcdf_fp)
df_new.attrs = df_oggm.attrs
df_new['simulated_thickness'].data = new_thickness

# Calculate the area coverage per oggm pixel
df_new['area_mask'] = (df_new.simulated_thickness > 0)
area_coverage = df_new.area_mask.assign_coords({'x': ('x', new_oggm_x),
                                                'y': ('y', new_oggm_y)})

area_coverage_perc = area_coverage / dx_oggm ** 2

# Now we make the bins for which we will groupby for
# This are the edges and the center of each grid point from INCA_DEM.nc
#
dx_inca = x_coarse[1] - x_coarse[0]
half_pixel = dx_inca / 2

x_edges = np.arange(x_coarse[0] - half_pixel, x_coarse[-1] + half_pixel, dx_inca)
y_edges = np.arange(y_coarse[0] - half_pixel, y_coarse[-1] + half_pixel, dx_inca)

aggregated_area_perc = (
    area_coverage_perc
    .groupby_bins('x', bins=x_edges, labels=x_coarse[:-1])
    .sum(dim="x")
    .groupby_bins('y', bins=y_edges, labels=y_coarse[:-1])
    .sum(dim="y")
)

df_new['area_perc_in_coarse'] = aggregated_area_perc

df_new['area_perc_in_coarse'].attrs = {'long_name': '% of glaciated area coverage on the coarse grid'}

# Delete the intermediate file and save the complete one
os.remove(netcdf_fp)
df_new.to_netcdf(netcdf_fp)
