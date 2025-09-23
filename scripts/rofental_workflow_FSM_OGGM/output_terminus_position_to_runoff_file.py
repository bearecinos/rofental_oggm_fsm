from __future__ import division
import sys
import configparser
import logging
import os
import glob
import geopandas as gpd
import xarray as xr
import salem
import multiprocessing
import re
import shutil

import numpy as np
import pandas as pd
import pyproj
from salem import wgs84
from oggm.exceptions import InvalidWorkflowError

# Time
import time
from oggm import cfg, tasks, utils, workflow
from oggm.utils import write_centerlines_to_shape
log = logging.getLogger(__name__)
start = time.time()

## Define helper functions
def wait_for_file(path, timeout=60):
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            raise TimeoutError(f"File {path} not found within {timeout} seconds.")
        time.sleep(1)

def wait_for_multiple_files(filepaths, timeout=60):
    for f in filepaths:
        wait_for_file(f, timeout)

def extract_terminus_position_per_year(topo_year,
                                       centerlines_fpath=None,
                                       output_fpath=None,
                                       return_data_set=False):
    """
    Extracts terminus position per year of simulation and stores it in a .csv file for all glaciers
    in the geopandas centerline dataframe.

    :param topo_year: xarray with the elevation of glacier ice covered areas per year
    :param centerlines_fpath: path to geopandas centerlines dataframe (one main centerline per glacier)
    :param output_fpath: paths to store the pickle output file with terminus positions
    :param return_data_set: if true returns the list of terminus positions per year
    :returns a pandas.Dataframe with RGIID, terminus position coords (lat and lon)
    """
    if centerlines_fpath is None:
        raise InvalidWorkflowError('You need to compute centerlines first!')
    if output_fpath is None:
        raise InvalidWorkflowError('Please provide paths to output pickle files')

    centerlines = gpd.read_file(centerlines_fpath)

    centerlines['coords'] = centerlines.geometry.apply(lambda geom: list(geom.coords))

    dfinal = pd.DataFrame()

    for i in centerlines.index:
        shp = centerlines.iloc[[i]]
        rgi_id = shp.RGIID
        raster_proj = pyproj.Proj(topo_year.attrs['pyproj_srs'])

        ds_fls = topo_year.salem.roi(shape=shp)
        x, y = zip(*shp.coords[i])

        # For the entire flowline
        x_all, y_all = salem.gis.transform_proj(wgs84, raster_proj, x, y)
        elev_fls = ds_fls.interp(x=np.array(x_all), y=np.array(y_all), method='nearest')
        yr = topo_year.time.values.tolist()
        if elev_fls is None or elev_fls.count() == 0:
            print(f"Skipping glacier {rgi_id.values[0]}: no valid elevation data in {yr}.")
            continue
        terminus = elev_fls.where(elev_fls == elev_fls.min(skipna=True), drop=True)

        if len(terminus) > 0:
            x_t = terminus.x.values[0]
            y_t = terminus.y.values[0]
            lon, lat = salem.gis.transform_proj(raster_proj, wgs84, x_t, y_t)
        else:
            lon = np.nan
            lat = np.nan

        terminus_coords = {'i': i, 'RGIID': rgi_id, 'lon': lon, 'lat': lat}

        df = pd.DataFrame(terminus_coords)
        df = df.set_index(['i'])
        dfinal = pd.concat([dfinal, df])

    dfinal.to_csv(output_fpath)

    if return_data_set:
        return dfinal

## Define main function
def main(cfg_path):
    # 1) Read configuration file
    cp = configparser.ConfigParser()
    cp.read(cfg_path)
    gen  = cp['General']
    oggm = cp['OGGM']
    fsm  = cp['FSM_OGGM']
    inp  = cp['InputData']
    outp = cp['Output']

    # 2) General settings
    working_dir = gen.get('working_dir')
    # We force reset=False here since this is pure post-processing
    reset = False

    # 3) Initialize OGGM
    cfg.initialize(logging_level='DEBUG')
    cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=reset)
    print(f"Working directory: {cfg.PATHS['working_dir']}")
    print("Reset forced to False for post-processing")

    # 4) OGGM core params
    cfg.PARAMS['use_multiprocessing'] = oggm.getboolean('use_multiprocessing')
    cfg.PARAMS['mp_processes']        = oggm.getint('mp_processes')
    cfg.PARAMS['border']              = oggm.getint('border', fallback=80)

    # 5) FSM parameters (only those relevant to post-processing)
    cfg.PARAMS['FSM_save_runoff']      = fsm.getboolean('FSM_save_runoff')
    cfg.PARAMS['FSM_runoff_frequency'] = fsm.get('FSM_runoff_frequency')
    # (the rest of the FSM params are only used during runs)

    # 6) Standard OGGM flags
    cfg.PARAMS['continue_on_error']       = True
    cfg.PARAMS['use_compression']         = True
    cfg.PARAMS['use_tar_shapefiles']      = True
    cfg.PATHS['rgi_version']              = '62'
    cfg.PARAMS['use_temp_bias_from_file'] = True
    cfg.PARAMS['compress_climate_netcdf'] = False
    cfg.PARAMS['store_model_geometry']    = True
    cfg.PARAMS['store_fl_diagnostics']    = True

    # 7) Define FSM_runoff basename
    _doc = ("A netcdf file containing dates and "
            "ice-based and snow-based runoff volume for each date interval")
    cfg.BASENAMES['FSM_runoff'] = ('FSM_runoff.nc', _doc)

    # 8) Load RGI regions & catchment polygon
    fr  = utils.get_rgi_region_file(11, version='62', reset=False)
    gdf = gpd.read_file(fr)

    catchment_path = inp.get('catchment_path')
    rof_shp = gpd.read_file(catchment_path)
    rof_sel = gdf.clip(rof_shp)
    rof_sel = rof_sel.sort_values('Area', ascending=False)

    rgi_id = inp.get('glacier_rgi_id')
    if rgi_id in ('None', '', None):
        rgi_id = None

    if rgi_id:
        selection = rof_sel[rof_sel.RGIId == rgi_id]
    else:
        selection = rof_sel

    # Here we never need to reset the working directory since we start
    # from a working dir where simulations have been run before
    gdirs = workflow.init_glacier_directories(selection)

    # Let's make a directory for CEH data and file formats
    output_dir = os.path.join(cfg.PATHS['working_dir'],
                              'run_off_terminus_position')
    os.makedirs(output_dir, exist_ok=True)

    shp_path = os.path.join(output_dir, 'Rofental_Centerlines.shp')
    if not os.path.exists(shp_path):
        # We recompute geometry in each glacier dir,
        # so we can get centerlines in a shapefile
        list_talks = [
            tasks.glacier_masks,
            tasks.compute_centerlines,
        ]
        for task in list_talks:
            # The order matters!
            workflow.execute_entity_task(task, gdirs)

        # Remove from gdirs those that dont have thickness distribution due to errors

        write_centerlines_to_shape(gdirs,  # The glaciers to process
                                   path=shp_path,  # The output file
                                   to_tar=False,  # set to True to put everything into one single tar file
                                   to_crs=selection.crs,  # Write into the projection of the original inventory
                                   keep_main_only=True,  # Write only the main flowline and discard the tributaries
                                   )

        print("Shapefile written, waiting for file sync...")
        base = shp_path[:-4]
        wait_for_multiple_files([f"{base}{ext}" for ext in ['.shp', '.dbf', '.shx', '.prj']])

    # We read the data that we need
    # Shapefile with all centrelines
    print("Reading centerlines...")
    centerlines = gpd.read_file(os.path.join(output_dir, 'Rofental_Centerlines.shp'))
    centerlines['coords'] = centerlines.geometry.apply(lambda geom: list(geom.coords))

    # ----------------------
    # 10) Final 2D distribution
    # ----------------------
    simulation_name = outp.get('simulation_name')

    pattern = os.path.join(cfg.PATHS['working_dir'],
                           'distributed_data' + simulation_name,
                           "*all_simulations_merged*")

    matched_files = sorted(glob.glob(pattern))

    topo_file_pattern = os.path.join(cfg.PATHS['working_dir'],
                                     'distributed_data' + simulation_name,
                                     "*topo*")

    matched_dem = sorted(glob.glob(topo_file_pattern))

    # We process a single simulation at the time
    # OGGM thickness
    doggm = salem.open_xr_dataset(matched_files[0])

    # OGGM topo
    doggm_elevation = salem.open_xr_dataset(matched_dem[0])
    topo_smooth = doggm_elevation.topo_smoothed

    doggm['area_mask'] = (doggm.simulated_thickness > 0)
    print('Glaciated area mask per year computed')

    # Let's prepare arrays to deploy in a multiprocessing workflow per year
    years = doggm.time.values.astype(int)
    dfs = []
    for year in years:
        mask = doggm.area_mask.sel(time=year)
        topo_masked = topo_smooth.where(mask == 1)
        dfs.append(topo_masked)

    gpd_file = os.path.join(output_dir, 'Rofental_Centerlines.shp')
    geopandas_file = np.repeat(gpd_file, len(years))

    intermediate_files_dir = os.path.join(output_dir,
                                          'intermediate_files',
                                          simulation_name)
    os.makedirs(intermediate_files_dir, exist_ok=True)

    file_names = []

    # DNG: the issue is that there is a file for ye+1, giving the dist thickness at the END 
    #      of ye. There are no ye+1 entries in the runoff file

    if years[-1] > inp.getint('y1'):
        years=years[:np.where(years==inp.getint('y1')+1)[0][0]]

    for y in years:
        file_names.append(os.path.join(intermediate_files_dir,
                                       'terminus_tracking_' + str(y) + '_' + simulation_name + '.csv'))

    print("Starting multiprocessing" if cfg.PARAMS['use_multiprocessing'] else "Running serial.")
    if cfg.PARAMS['use_multiprocessing']:
        with multiprocessing.Pool(processes=cfg.PARAMS['mp_processes']) as pool:
            result = pool.starmap(extract_terminus_position_per_year, zip(dfs, geopandas_file, file_names))
            print(result)
    else:
        result = [extract_terminus_position_per_year(topo, gdf, fname)
                  for topo, gdf, fname in zip(dfs, geopandas_file, file_names)]

    print("Done outputting terminus position data")

    matching_files = []
    for filename in os.listdir(output_dir):
        if re.match('run_off_daily_and_terminus_position' + simulation_name, filename):
            matching_files.append(filename)
    print(matching_files)
    file_to_change = os.path.join(output_dir, matching_files[0])

    df_new = xr.open_dataset(file_to_change)

    i = np.arange(len(years))
    for year, file, t_index in zip(years, file_names, i):
        df = pd.read_csv(file)

        # the search below is based on the structure of the csv files created by
        #   extract_terminus_position_per_year.
        # result is that the (daily) runoff only gets terminus positions on the 
        #   first of every year, meaning a large number of null values
        for rgiid in df_new.RGIID.values:
            key = str(rgiid)
            if key in df.RGIID.tolist():
                dpg = df.loc[df.RGIID==key]
                ydt = pd.Timestamp(f"{year}-01-01")
                df_new['lat'].loc[dict(time=ydt, RGIID=rgiid)] = dpg['lat'].values[0]
                df_new['lon'].loc[dict(time=ydt, RGIID=rgiid)] = dpg['lon'].values[0]

    os.remove(file_to_change)
    df_new.to_netcdf(file_to_change)
    print("Done saving terminus position data to runoff file")

    #shutil.rmtree(intermediate_files_dir, ignore_errors=True)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python output_terminus_position_to_runoff_file.py <config.ini>")
        sys.exit(1)
    main(sys.argv[1])
