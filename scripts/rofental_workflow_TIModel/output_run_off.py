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
from collections import defaultdict
import numpy as np

# Time
import time
from oggm import cfg, utils, workflow
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


sim_output_paths = defaultdict(lambda: defaultdict(list))

sims = ['ssp126', 'ssp370', 'ssp585']

for ssp in sims:
    for i, gdir in enumerate(gdirs):
        path_up = os.path.dirname(gdir.get_filepath('model_diagnostics'))
        full_path = os.path.join(path_up, 'model_diagnostics' + '_ISIMIP3b_mri-esm2-0_r1i1p1f1_' + ssp + '.nc')
        sim_output_paths[ssp][gdir.rgi_id] = full_path

dfinal = pd.DataFrame()

for i, rgi_id in enumerate(sim_output_paths[sims[simulation_index]]):
    fpath = sim_output_paths[sims[simulation_index]][rgi_id]
    if len(fpath) > 0:
        with xr.open_dataset(fpath) as dg:
            dg_ssp = dg.load()

        calendar_year = dg_ssp.calendar_year.values
        calendar_month = dg_ssp.calendar_month.values
        hydro_year = dg_ssp.hydro_year.values
        hydro_month = dg_ssp.hydro_month.values
        lat = np.repeat(np.nan, len(dg_ssp.calendar_year.values))
        lon = np.repeat(np.nan, len(dg_ssp.calendar_year.values))
        melt_on_glacier = dg_ssp.melt_on_glacier.values
        id = np.repeat(rgi_id, len(dg_ssp.calendar_year.values))

        row = {'RGIID': id,
               'calendar_year': calendar_year,
               'calendar_month': calendar_month,
               'hydro_year': hydro_year,
               'hydro_month': hydro_month,
               'lat': lat,
               'lon': lon,
               'melt_on_glacier': melt_on_glacier
               }
        df = pd.DataFrame(row)
        df = df.set_index(['RGIID', 'calendar_year'])
        dfinal = pd.concat([dfinal, df])

ds_year = dfinal.to_xarray()

# Monthly runoff
dfinal = pd.DataFrame()

for i, rgi_id in enumerate(sim_output_paths[sims[simulation_index]]):
    fpath = sim_output_paths[sims[simulation_index]][rgi_id]
    if len(fpath) > 0:
        with xr.open_dataset(fpath) as dg:
            dg_ssp = dg.load()

        calendar_year = np.repeat(dg_ssp.calendar_year.values, 12)
        month_2d = np.tile(dg_ssp.month_2d.values, len(dg_ssp.calendar_year.values))
        hydro_year = np.repeat(dg_ssp.hydro_year.values, 12)
        hydro_month_2d = np.tile(dg_ssp.hydro_month_2d.values, len(dg_ssp.calendar_year.values))
        melt_on_glacier_monthly = dg_ssp.melt_on_glacier_monthly.data.ravel()
        lat = np.repeat(np.nan, len(melt_on_glacier_monthly))
        lon = np.repeat(np.nan, len(melt_on_glacier_monthly))
        id = np.repeat(rgi_id, len(melt_on_glacier_monthly))
        month_index = np.arange(0, len(dg_ssp.calendar_year.values) * 12)

        row = {'RGIID': id,
               'month_index': month_index,
               'calendar_year': calendar_year,
               'month_2d': month_2d,
               'hydro_year': hydro_year,
               'hydro_month_2d': hydro_month_2d,
               'lat': lat,
               'lon': lon,
               'melt_on_glacier_monthly': melt_on_glacier_monthly
               }
        df = pd.DataFrame(row)
        df = df.set_index(['RGIID', 'calendar_year', 'month_2d'])
        dfinal = pd.concat([dfinal, df])

ds = dfinal.to_xarray()

# For one glacier we check that the monthly runoff equals yearly
# To be sure we save it right
total = ds_year.sel(RGIID='RGI60-11.00897', calendar_year=2020).melt_on_glacier.values
total_m = ds.sel(RGIID='RGI60-11.00897', calendar_year=2020).melt_on_glacier_monthly.values.sum()
assert total == total_m

# Let's make a directory for CEH data and file formats
output_dir = os.path.join(config['main_repo_path'],
                          'output_data/04_run_off_terminus_position')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


final_path_m = os.path.join(output_dir,
                            'run_off_monthly_and_terminus_position_' + sims[simulation_index] + '.nc')

ds.to_netcdf(final_path_m)


final_path_yr = os.path.join(output_dir,
                             'run_off_yearly_and_terminus_position_' + sims[simulation_index] + '.nc')
ds_year.to_netcdf(final_path_yr)
