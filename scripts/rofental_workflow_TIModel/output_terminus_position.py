# Module logger
from __future__ import division
import logging
import argparse
import os
import sys
import geopandas as gpd
from configobj import ConfigObj
import xarray as xr
import pandas as pd
import salem
from collections import defaultdict
import numpy as np
import multiprocessing
import re
import shutil

# Time
import time
from oggm import cfg, tasks, utils, workflow
from oggm.utils import write_centerlines_to_shape
log = logging.getLogger(__name__)
start = time.time()

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-run_mode", type=bool, default=False, help="pass running mode")
parser.add_argument("-simulation_index",
                    type=int,
                    default=0, help="Simulation index")
args = parser.parse_args()

config_file = args.conf
run_mode = args.run_mode
config = ConfigObj(os.path.expanduser(config_file))
simulation_index = args.simulation_index

working_dir = os.path.join(config['main_repo_path'],
                           'output_data/02_all_rofental')

sys.path.append(config['main_repo_path'])
from tools.mix_gis import extract_terminus_position_per_year

# OGGM configuration Params
cfg.initialize(logging_level='ERROR')
cfg.PATHS['working_dir'] = utils.mkdir(working_dir)
print(cfg.PATHS['working_dir'])
cfg.PARAMS['border'] = 80
if run_mode:
    cfg.PARAMS['use_multiprocessing'] = False
else:
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['mp_processes'] = 16
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_compression'] = True
cfg.PARAMS['use_tar_shapefiles'] = True
cfg.PATHS['rgi_version'] = '62'
cfg.PARAMS['use_temp_bias_from_file'] = True
cfg.PARAMS['compress_climate_netcdf'] = False
cfg.PARAMS['store_model_geometry'] = True
cfg.PARAMS['store_fl_diagnostics'] = True

fr = utils.get_rgi_region_file(11, version='62', reset=False)
gdf = gpd.read_file(fr)

# Selecting the glaciers that belong to the Rofental catchment
minlat = 46.690
maxlat = 47.170
minlon = 10.6
maxlon = 11.3
rof = gdf[gdf['CenLat'].between(minlat, maxlat) & gdf['CenLon'].between(minlon, maxlon)]

rof = rof.sort_values('Area', ascending=False)

if run_mode:
    selection = rof[rof.Name == 'Hintereisferner']
else:
    # Remove one error:
    list_id_sel = ['RGI60-11.00439']
    keep_indexes = [(i not in list_id_sel) for i in rof.RGIId]
    selection = rof[keep_indexes]
    print('Running this many glaciers', len(selection))

# TODO see if it is best to re-start directories from tar files
# More information is here : https://oggm.org/tutorials/
# store_and_compress_glacierdirs.html
# #store-the-single-glacier-directories-into-tar-files

gdirs = workflow.init_glacier_directories(selection)

# Let's make a directory for CEH data and file formats
output_dir = os.path.join(config['main_repo_path'],
                          'output_data/04_run_off_terminus_position')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sims = ['ssp126', 'ssp370', 'ssp585']
intermediate_files_dir = output_dir + '/intermediate_files/' + sims[simulation_index]
if not os.path.exists(intermediate_files_dir):
    os.makedirs(intermediate_files_dir)

if not os.path.exists(os.path.join(output_dir, 'Rofental_Centerlines.shp')):
    # We recompute geometry in each glacier dir,
    # so we can get centerlines in a shapefile
    list_talks = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
    ]
    for task in list_talks:
        # The order matters!
        workflow.execute_entity_task(task, gdirs)

    write_centerlines_to_shape(gdirs,  # The glaciers to process
                               path=os.path.join(output_dir, 'Rofental_Centerlines.shp'),  # The output file
                               to_tar=False,  # set to True to put everything into one single tar file
                               to_crs=selection.crs,  # Write into the projection of the original inventory
                               keep_main_only=True,  # Write only the main flowline and discard the tributaries
                               )


# We read the data that we need
# Shapefile with all centrelines
centerlines = gpd.read_file(os.path.join(output_dir, 'Rofental_Centerlines.shp'))
centerlines['coords'] = centerlines.geometry.apply(lambda geom: list(geom.coords))

# Simulation rasters with thickness per year and elevation
sim_output_paths = defaultdict(lambda: defaultdict(list))
topo_output_paths = defaultdict(lambda: defaultdict(list))

for ssp in ['ssp126', 'ssp370', 'ssp585']:
    distributed_name = 'distributed_data' + ssp + '/' + 'all_simulations_merged_for_' + ssp + '.nc'
    topo_name = 'distributed_data' + ssp + '/' + 'all_merged_for_' + ssp + '_topo_data' + '.nc'
    full_path_sim = os.path.join(working_dir, distributed_name)
    full_path_topo = os.path.join(working_dir, topo_name)
    assert os.path.exists(full_path_sim)
    assert os.path.exists(full_path_topo)
    sim_output_paths[ssp] = full_path_sim
    topo_output_paths[ssp] = full_path_topo

# We process a single simulation at the time
# OGGM thickness
doggm = salem.open_xr_dataset(sim_output_paths[sims[simulation_index]])

# OGGM topo
doggm_elevation = salem.open_xr_dataset(topo_output_paths[sims[simulation_index]])

glacier_mask = doggm_elevation.glacier_mask
topo_smooth = doggm_elevation.topo_smoothed

doggm['area_mask'] = (doggm.simulated_thickness > 0)
print('Glaciated area mask per year computed')

# Let's prepare arrays to deploy in a multiprocessing workflow per year
years = doggm.time.values.astype(int)
dfs = [topo_smooth.where(doggm.area_mask.sel(time=year) == 1) for year in years]

gpd_file = os.path.join(output_dir, 'Rofental_Centerlines.shp')
geopandas_file = np.repeat(gpd_file, len(years))

# For each terminus position tracking per year over the entire Rofental area
# we will output an intermediate file per year and then add the terminus coordinates
# evolution to the main runoff netcdf
file_names = []
for y in years:
    file_names.append(os.path.join(intermediate_files_dir,
                                   'terminus_tracking_' + str(y) + '_' + sims[simulation_index] + '.csv'))


print('We are about to deploy multiprocessing')

if __name__ == '__main__':
    with multiprocessing.Pool(processes=4) as pool:
        result = pool.starmap(extract_terminus_position_per_year,
                              zip(dfs, geopandas_file, file_names))
        pool.close()
        pool.join()


matching_files = []
for filename in os.listdir(output_dir):
    if re.match('run_off_yearly_and_terminus_position_', filename):
        matching_files.append(filename)

matching_files = sorted(matching_files)

file_to_change = os.path.join(output_dir, matching_files[simulation_index])

df_new = xr.open_dataset(file_to_change)

df_new = df_new.rename_dims({'calendar_year': 'time'})
rgi_ids = df_new.RGIID.values

i = np.arange(len(years))

for y, f, i in zip(years, file_names, i):
    df = pd.read_csv(f)
    for rgiid in df_new.RGIID.values:
        row_index = df.index.get_loc(df[df['RGIID'] == str(rgiid)].index[0])
        dpg = df.iloc[[row_index]]
        df_new['lat'].loc[dict(time=i, RGIID=str(rgiid))] = dpg['lat'].values[0]
        df_new['lon'].loc[dict(time=i, RGIID=str(rgiid))] = dpg['lon'].values[0]

os.remove(file_to_change)
df_new.to_netcdf(file_to_change)

shutil.rmtree(intermediate_files_dir, ignore_errors=True)
