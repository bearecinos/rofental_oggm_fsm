import os
import sys
import configparser
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import glob
from oggm import cfg, utils
from oggm import workflow
from oggm.sandbox import distribute_2d

## Define main function
def main(cfg_path):
    # ----------------------
    # 1) Read configuration file
    # ----------------------
    cp = configparser.ConfigParser()
    cp.read(cfg_path)

    gen_config = cp['General']
    oggm_config = cp['OGGM']
    #fsm_config = cp['FSM_OGGM']
    inp_config = cp['InputData']
    outp_config = cp['Output']

    # ----------------------
    # 2) Parse general settings
    # ----------------------
    working_dir = gen_config.get('working_dir')  # string, may be blank
    # We force reset=False here since this is pure post-processing
    reset = False  # bool

    # ----------------------
    # 3) Initialize OGGM core
    # ----------------------
    cfg.initialize(logging_level='DEBUG')
    if working_dir.strip() == '':
        # If this params.ini working_dir is an empty string
        # we create a tmp dir in the /tmp/ folder of the user
        cfg.PATHS['working_dir'] = utils.gettempdir('FSM_rofental', reset=reset)
    else:
        cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=reset)
    print(f"Working directory: {cfg.PATHS['working_dir']}")
    print(f"Reset = {reset}  (set False to preserve existing directories)")

    # Configure parameters from arguments
    cfg.PARAMS['use_multiprocessing'] = oggm_config.getboolean('use_multiprocessing')
    cfg.PARAMS['mp_processes']        = oggm_config.getint('mp_processes')
    cfg.PARAMS['border']              = oggm_config.getint('border', fallback=80)

    # 5) FSM parameters (only those relevant to post-processing)
    #cfg.PARAMS['FSM_save_runoff']      = fsm_config.getboolean('FSM_save_runoff')
    #cfg.PARAMS['FSM_runoff_frequency'] = fsm_config.get('FSM_runoff_frequency')
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
    #_doc = ("A netcdf file containing dates and "
    #        "ice-based and snow-based runoff volume for each date interval")
    #cfg.BASENAMES['FSM_runoff'] = ('FSM_runoff.nc', _doc)

    # 8) Load RGI regions & catchment polygon
    fr  = utils.get_rgi_region_file(11, version='62', reset=False)
    gdf = gpd.read_file(fr)

    catchment_path = inp_config.get('catchment_path')
    rof_shp = gpd.read_file(catchment_path)
    rof_sel = gdf.clip(rof_shp)
    rof_sel = rof_sel.sort_values('Area', ascending=False)

    rgi_id = inp_config.get('glacier_rgi_id')
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
#    output_dir = os.path.join(cfg.PATHS['working_dir'],
#                              'run_off_terminus_position')

#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)

    # ----------------------
    # 10) Final 2D distribution
    # ----------------------
    simulation_name = outp_config.get('simulation_name')

    path_for_distributed_data = os.path.join(working_dir,
                                             'distributed_data' + simulation_name)

    distribute_2d.merge_simulated_thickness(gdirs,
                                            output_folder=path_for_distributed_data,
                                            output_filename='all_merged_for_',
                                            add_topography='NASADEM',
                                            keep_dem_file=True,
                                            use_multiprocessing=True,
                                            simulation_filesuffix=simulation_name)

    merged_files = sorted(glob.glob(os.path.join(str(path_for_distributed_data),
                                                 'all_merged_for_' + '*_01.nc')))

    f_path = os.path.join(str(path_for_distributed_data),
                          "all_simulations_merged_for" + simulation_name + ".nc")

    with xr.open_mfdataset(merged_files) as ds:
        final_d = ds.load()

    final_d.to_netcdf(f_path)

    print("DONE running distributed thickness postprocessing")
#    print("Now we start post-processing runoff")

#    dfinal = pd.DataFrame()
#
#    for gdir in gdirs:
#        runoff_path = gdir.get_filepath('FSM_runoff')
#        if os.path.exists(runoff_path):
#            with xr.open_dataset(gdir.get_filepath('FSM_runoff')) as dg:
#                dg_sim = dg.load()
#
#            calendar_year = dg_sim['time'].dt.year.values
#            calendar_month = dg_sim['time'].dt.month.values
#            calendar_day = dg_sim['time'].dt.day.values
#            time = dg_sim['time'].values
#            lat = np.repeat(np.nan, len(time))
#            lon = np.repeat(np.nan, len(time))
#            ice_melt_on_glacier = dg_sim.runoff_ice.values
#            snow_melt_on_glacier = dg_sim.runoff_snow.values
#            id = np.repeat(gdir.rgi_id, len(time))

#            row = {'RGIID': id,
#                   'calendar_year': calendar_year,
#                   'calendar_month': calendar_month,
#                   'calendar_day': calendar_day,
#                   'time': time,
#                   'lat': lat,
#                   'lon': lon,
#                   'ice_melt_on_glacier_daily': ice_melt_on_glacier,
#                   'snow_melt_on_glacier_daily': snow_melt_on_glacier,
#                   }

#            df = pd.DataFrame(row)
#            df = df.set_index(['RGIID', 'time'])
#            dfinal = pd.concat([dfinal, df])
#        else:
#            print("No runoff data for check log in gdir to see what went wrong", gdir)
#            continue

#    ds_runoff = dfinal.to_xarray()

#    final_path_daily = os.path.join(output_dir,
#                                    'run_off_daily_and_terminus_position_' + simulation_name + '.nc')

#    ds_runoff.to_netcdf(final_path_daily)

#    print("Done with post processing of thickness and runoff")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python output_distributed_thickness.py <config.ini>")
        sys.exit(1)
    main(sys.argv[1])
