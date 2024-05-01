# Module logger
from __future__ import division
import logging
log = logging.getLogger(__name__)

import argparse
import os
import sys
import geopandas as gpd
from configobj import ConfigObj
import xarray as xr

# Time
import time
import glob
start = time.time()

import oggm
from oggm import cfg, utils, workflow, tasks
from oggm.shop import gcm_climate
from oggm.sandbox import distribute_2d

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-run_mode", type=bool, default=False, help="pass running mode")
args = parser.parse_args()

config_file = args.conf
run_mode = args.run_mode
config = ConfigObj(os.path.expanduser(config_file))

working_dir = os.path.join(config['main_repo_path'],
                           'output_data/02_all_rofental')

sys.path.append(config['main_repo_path'])

## OGGM configuration Params
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

## Selecting the glaciers that belong to the Rofental catchment
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


if len(gdirs) < 2:
    for gdir in gdirs:
        print(os.listdir(gdir.dir))

#for ssp in ['ssp126', 'ssp370', 'ssp585']:
#    rid = '_ISIMIP3b_mri-esm2-0_r1i1p1f1_' + ssp
#    workflow.execute_entity_task(distribute_2d.add_smoothed_glacier_topo, gdirs)
#    workflow.execute_entity_task(distribute_2d.assign_points_to_band, gdirs)
#    workflow.execute_entity_task(distribute_2d.distribute_thickness_from_simulation,
#                                 gdirs,
#                                 input_filesuffix=rid)

from IPython import embed

for ssp in ['ssp126', 'ssp370', 'ssp585']:
    rid = '_ISIMIP3b_mri-esm2-0_r1i1p1f1_' + ssp

    path_for_distributed_data = os.path.join(working_dir,
                                             'distributed_data' + ssp)
    #distribute_2d.merge_simulated_thickness(gdirs,
    #                                        output_folder=path_for_distributed_data,
    #                                        output_filename='all_merged_for_' + ssp,
    #                                        add_topography=True,
    #                                        keep_dem_file=True,
    #                                        use_multiprocessing=True,
    #                                        simulation_filesuffix=rid)

    # Now we merge all those together per scenario
    merged_files = sorted(glob.glob(os.path.join(path_for_distributed_data,
                                                 "all_merged_for_" + ssp + "_*_01.nc")))
    embed()
    f_path = os.path.join(path_for_distributed_data, "all_simulations_merged_for_" + ssp + ".nc")

    with xr.open_mfdataset(merged_files) as ds:
        final_d = ds.load()

    final_d.to_netcdf(f_path)

