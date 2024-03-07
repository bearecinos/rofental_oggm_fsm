from __future__ import division
import time
import argparse
import os
import geopandas as gpd
from configobj import ConfigObj

# Module logger
import logging

# OGGM imports
from oggm import cfg, utils, workflow, tasks
from oggm.shop import gcm_climate

log = logging.getLogger(__name__)
# Time
start = time.time()

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf", type=str, default="../../../config.ini", help="pass config file")
parser.add_argument("-run_mode", type=bool, default=False, help="pass running mode")
parser.add_argument("-elevation_bands", type=bool, default=False, help="pass running mode")
args = parser.parse_args()

config_file = args.conf
run_mode = args.run_mode
with_elevation_bands = args.elevation_bands

config = ConfigObj(os.path.expanduser(config_file))
MAIN_PATH = config['main_repo_path']


working_dir = os.path.join(config['main_repo_path'],
                           'output_data/01_initial_state')

if with_elevation_bands:
    string_config = 'elev_bands/W5E5_w_data/'
else:
    string_config = 'centerlines/W5E5/'

base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/' + string_config
print('This will be the url end', string_config)

# OGGM configuration
cfg.initialize(logging_level='ERROR')
# Define working directories (either local if run_mode = true)
# or in the cluster environment
if run_mode:
    cfg.PATHS['working_dir'] = utils.get_temp_dir('test-run')
else:
    # Local paths (where to write output and where to download input)
    cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=True)
print('this is our working dir', cfg.PATHS['working_dir'])

# Use multiprocessing
if run_mode:
    cfg.PARAMS['use_multiprocessing'] = False
else:
    # ONLY IN THE CLUSTER!
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['mp_processes'] = 20

# Not sure if the params below are needed
cfg.PARAMS['border'] = 80
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['use_compression'] = True
cfg.PARAMS['use_tar_shapefiles'] = True
# Should this be PATHS or PARAMS?
cfg.PARAMS['rgi_version'] = '62'
cfg.PARAMS['use_temp_bias_from_file'] = True
cfg.PARAMS['compress_climate_netcdf'] = False
cfg.PARAMS['store_model_geometry'] = True

# The Alps
fr = utils.get_rgi_region_file('11', version='62', reset=False)
gdf = gpd.read_file(fr)

# Selecting the glaciers that belong to the Rofental catchment
minlat = 46.690
maxlat = 47.170
minlon = 10.6
maxlon = 11.3

rof = gdf[gdf['CenLat'].between(minlat, maxlat) & gdf['CenLon'].between(minlon, maxlon)]
## TODO: get the largest glaciers to run in run_mode

gdirs = workflow.init_glacier_directories(rof, from_prepro_level=3, prepro_base_url=base_url)

spinup_start_yr = 1979

# Run spinup for glaciers via different methods to compare
# Method 1) Fix geometry spinup
workflow.execute_entity_task(tasks.run_from_climate_data,
                             gdirs,
                             fixed_geometry_spinup_yr=spinup_start_yr,
                             output_filesuffix='_hist_fixed_geom')

# Method 2) run_dynamic_spinup option area
workflow.execute_entity_task(tasks.run_dynamic_spinup,
                             gdirs,
                             spinup_start_yr=spinup_start_yr,
                             minimise_for='area',
                             output_filesuffix='_spinup_dynamic_area',
                             ye=2020)

# Method 3) run_dynamic_spinup option volume
workflow.execute_entity_task(tasks.run_dynamic_spinup,
                             gdirs,
                             spinup_start_yr=spinup_start_yr,
                             minimise_for='volume',
                             output_filesuffix='_spinup_dynamic_volume',
                             ye=2020)

# Method 4) run_dynamic_melt_f_calibration
workflow.execute_entity_task(tasks.run_dynamic_melt_f_calibration,
                             gdirs,
                             ys=spinup_start_yr,
                             ye=2020,
                             output_filesuffix='_dynamic_melt_f')

member = 'mri-esm2-0_r1i1p1f1'

for ssp in ['ssp126', 'ssp370', 'ssp585']:
    # bias correct them
    workflow.execute_entity_task(gcm_climate.process_monthly_isimip_data, gdirs,
                                 # gcm member -> you can choose another one
                                 ssp=ssp,
                                 member=member,
                                 # recognize the climate file for later
                                 output_filesuffix=f'_ISIMIP3b_{member}_{ssp}')

for ssp in ['ssp126', 'ssp370', 'ssp585']:
    rid = f'_ISIMIP3b_{member}_{ssp}'
    workflow.execute_entity_task(tasks.run_from_climate_data, gdirs,
                                 # use gcm_data, not climate_historical
                                 climate_filename='gcm_data',
                                 # use the chosen scenario
                                 climate_input_filesuffix=rid,
                                 # this is important! Start from 2020 glacier
                                 init_model_filesuffix='_dynamic_melt_f',
                                 # recognize the run for later
                                 output_filesuffix=rid)
