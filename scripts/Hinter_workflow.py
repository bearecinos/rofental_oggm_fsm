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
start = time.time()

import oggm
from oggm import cfg, utils, workflow, tasks
from oggm.shop import gcm_climate

# Parameters to pass into the python script form the command line
parser = argparse.ArgumentParser()
parser.add_argument("-conf",
                    type=str,
                    default="../../../config.ini", help="pass config file")
parser.add_argument("-reset_dir", type=bool, default=False, help="reset working dir")
args = parser.parse_args()

config_file = args.conf
config = ConfigObj(os.path.expanduser(config_file))

reset_work_dir = args.reset_dir

working_dir = os.path.join(config['main_repo_path'],
                           'output_data/01_initial_state')

base_url = ('https://cluster.klima.uni-bremen.de/~oggm/'
            'gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/')

sys.path.append(config['main_repo_path'])
from tools.plots import plot_different_spinup_results

## OGGM configuration Params
cfg.initialize(logging_level='ERROR')
cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=True)
print(cfg.PATHS['working_dir'])
cfg.PARAMS['border'] = 80
cfg.PARAMS['use_multiprocessing'] = False
#cfg.PARAMS['mp_processes'] =
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

# Save list of Glaciers in cropped area only if doesn't exist
path_ids = os.path.join(working_dir,
                       'list_of_ids_rofental.txt')

if not os.path.exists(path_ids):
    with open(os.path.join(working_dir,
                           'list_of_ids_rofental.txt'), 'w') as fp:
        for item in rof.RGIId:
            # write each item on a new line
            fp.write("%s\n" % item)

print('Done')

hint = rof[rof.Name == 'Hintereisferner']


gdirs = workflow.init_glacier_directories(hint,
                                          from_prepro_level=3,
                                          prepro_base_url=base_url,
                                          reset=reset_work_dir)

# Tested tasks
task_list = [
    tasks.compute_downstream_line,
    tasks.compute_downstream_bedshape,
    tasks.gridded_attributes,
    tasks.gridded_mb_attributes,
]
for task in task_list:
    workflow.execute_entity_task(task, gdirs)

# Distribute
workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs)

# Test that we have at least 21 variables on gridded_data.nc
gdir = gdirs[0]
with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
    ds = ds.load()
    assert len(ds.count().variables.keys()) == 21

## Initialise with thickness data
bin_variables = ['consensus_ice_thickness', 'millan_ice_thickness']
workflow.execute_entity_task(tasks.elevation_band_flowline,
                             gdirs,
                             bin_variables=bin_variables)
workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline,
                             gdirs,
                             bin_variables=bin_variables)

# create the dynamic flowlines for consensus and millan thickness
tasks.init_present_time_glacier(gdir, filesuffix='_consensus',
                                use_binned_thickness_data='consensus_ice_thickness')
tasks.init_present_time_glacier(gdir, filesuffix='_millan',
                                use_binned_thickness_data='millan_ice_thickness')

## Run the 3 different methods for spinup
spinup_start_yr = 1979
# Method 1: Fixed geometry method
workflow.execute_entity_task(tasks.run_from_climate_data,
                             gdirs,
                             fixed_geometry_spinup_yr=spinup_start_yr,
                             output_filesuffix='_hist_fixed_geom')

# Method 2: Run_dynamic_spinup minimise to match the Area
workflow.execute_entity_task(tasks.run_dynamic_spinup,
                             gdirs,
                             spinup_start_yr=spinup_start_yr,
                             minimise_for='area',
                             output_filesuffix='_spinup_dynamic_area', ye=2020)

# Method 3: Run_dynamic_spinup minimise to match the Volume
workflow.execute_entity_task(tasks.run_dynamic_spinup,
                             gdirs,
                             spinup_start_yr=spinup_start_yr,
                             minimise_for='volume',
                             output_filesuffix='_spinup_dynamic_volume', ye=2020)

# Method 4 (chosen for the rest of this workflow):
# Run_dynamic_spinup calibrating the melting factor at the same time
workflow.execute_entity_task(tasks.run_dynamic_melt_f_calibration,
                             gdirs,
                             ys=spinup_start_yr,
                             ye=2020,
                             output_filesuffix='_dynamic_melt_f')

# Plot spinup results
for gdir in gdirs:
    plot_different_spinup_results(gdir, save_analysis_text=True)


# Download GCM data
member = 'mri-esm2-0_r1i1p1f1'

for ssp in ['ssp126', 'ssp370','ssp585']:
    # bias correct them
    workflow.execute_entity_task(gcm_climate.process_monthly_isimip_data,
                                 gdirs,
                                 # gcm member -> you can choose another one
                                 ssp = ssp,
                                 member=member,
                                 # recognize the climate file for later
                                 output_filesuffix=f'_ISIMIP3b_{member}_{ssp}'
                                 )

# Now run the simulations with hydro output!
for ssp in ['ssp126', 'ssp370', 'ssp585']:
    rid = f'_ISIMIP3b_{member}_{ssp}'
    workflow.execute_entity_task(tasks.run_with_hydro, gdir,
                                 run_task=tasks.run_from_climate_data,
                                 store_monthly_hydro=True,
                                 # use gcm_data, not climate_historical
                                 climate_filename='gcm_data',
                                 # use the chosen scenario
                                 climate_input_filesuffix=rid,
                                 # this is important! Start from 2020 glacier
                                 init_model_filesuffix='_dynamic_melt_f',
                                 # recognize the run for later,
                                 output_filesuffix=rid,
                                 store_fl_diagnostics=True)

if len(gdirs) < 2:
    for gdir in gdirs:
        print(os.listdir(gdir.dir))

## TODO: ask Fabien if is better to tar all this output and re-start glacier dir
## for post processing from tar files!