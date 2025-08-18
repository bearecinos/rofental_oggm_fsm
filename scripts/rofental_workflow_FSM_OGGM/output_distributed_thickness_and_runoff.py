import argparse
import os
import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import glob
from oggm import cfg, utils
from oggm import workflow
from oggm.sandbox import distribute_2d

## Define main function
def main(args):
    # In this script reset should be set to only False
    # This is all post processing!
    reset = False

    cfg.initialize(logging_level='DEBUG')

    # Configure parameters from arguments
    cfg.PARAMS['use_multiprocessing'] = args.use_multiprocessing
    cfg.PARAMS['mp_processes'] = args.mp_processes
    cfg.PARAMS['border'] = 80

    print('Reset is set to ', reset)
    print('**Important set this to False to avoid '
          'resetting the glacier directory everytime this is ran!**')

    # this sets a temporary working directory. if you want to use a permanent
    # directory then uncomment and adapt the following line.
    cfg.PATHS['working_dir'] = utils.mkdir(args.working_dir)
    # cfg.PATHS['working_dir'] = '/exports/geos.ed.ac.uk/iceocean/dgoldber/FSM-OGGM'
    print('we are working here', cfg.PATHS['working_dir'])

    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_compression'] = True
    cfg.PARAMS['use_tar_shapefiles'] = True
    cfg.PATHS['rgi_version'] = '62'
    cfg.PARAMS['use_temp_bias_from_file'] = True
    cfg.PARAMS['compress_climate_netcdf'] = False
    cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['store_fl_diagnostics'] = True

    _doc = ('A netcdf file containing dates and ' +
            'ice-based and snow-based runoff volume ' +
            'for each date interval')
    cfg.BASENAMES['FSM_runoff'] = ('FSM_runoff.nc', _doc)

    fr = utils.get_rgi_region_file(11, version='62', reset=False)
    gdf = gpd.read_file(fr)

    catchment_path = args.catchment_path
    rof_shp = gpd.read_file(catchment_path)

    rof_sel = gdf.clip(rof_shp)
    rof_sel = rof_sel.sort_values('Area', ascending=False)

    rgi_id = args.glacier_rgi_id
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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulation_name = args.simulation_name

    path_for_distributed_data = os.path.join(args.working_dir,
                                             'distributed_data' + simulation_name)

    distribute_2d.merge_simulated_thickness(gdirs,
                                            output_folder=path_for_distributed_data,
                                            output_filename='all_merged_for_',
                                            add_topography=True,
                                            keep_dem_file=True,
                                            use_multiprocessing=True,
                                            simulation_filesuffix=simulation_name)

    merged_files = sorted(glob.glob(os.path.join(str(path_for_distributed_data),
                                                 'all_merged_for_' + '*_01.nc')))

    f_path = os.path.join(str(path_for_distributed_data),
                          "all_simulations_merged_for"+ simulation_name + ".nc")

    with xr.open_mfdataset(merged_files) as ds:
        final_d = ds.load()

    final_d.to_netcdf(f_path)

    print("DONE running distributed thickness postprocessing")
    print("Now we start post-processing runoff")

    dfinal = pd.DataFrame()

    for gdir in gdirs:
        runoff_path = gdir.get_filepath('FSM_runoff')
        if os.path.exists(runoff_path):
            with xr.open_dataset(gdir.get_filepath('FSM_runoff')) as dg:
                dg_sim = dg.load()

            calendar_year = dg_sim['time'].dt.year.values
            calendar_month = dg_sim['time'].dt.month.values
            calendar_day = dg_sim['time'].dt.day.values
            time = dg_sim['time'].values
            lat = np.repeat(np.nan, len(time))
            lon = np.repeat(np.nan, len(time))
            ice_melt_on_glacier = dg_sim.runoff_ice.values
            snow_melt_on_glacier = dg_sim.runoff_snow.values
            id = np.repeat(gdir.rgi_id, len(time))

            row = {'RGIID': id,
                   'calendar_year': calendar_year,
                   'calendar_month': calendar_month,
                   'calendar_day': calendar_day,
                   'time': time,
                   'lat': lat,
                   'lon': lon,
                   'ice_melt_on_glacier_daily': ice_melt_on_glacier,
                   'snow_melt_on_glacier_daily': snow_melt_on_glacier,
                   }

            df = pd.DataFrame(row)
            df = df.set_index(['RGIID', 'time'])
            dfinal = pd.concat([dfinal, df])
        else:
            print("No runoff data for check log in gdir to see what went wrong", gdir)
            continue

    ds_runoff = dfinal.to_xarray()

    final_path_daily = os.path.join(output_dir,
                                    'run_off_daily_and_terminus_position_' + 'climate_historical_fsm' + '.nc')

    ds_runoff.to_netcdf(final_path_daily)

    print("Done with post processing of thickness and runoff")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run FSM OGGM model with customizable parameters')
    parser.add_argument('--reset', type=bool, default=True)
    parser.add_argument('--working_dir', type=str, default='')
    parser.add_argument('--use_multiprocessing', type=bool, default=False)
    parser.add_argument('--mp_processes', type=int, default=2)
    parser.add_argument('--border', type=int, default=80)
    parser.add_argument('--asm_x', type=float, default=0.85)
    parser.add_argument('--nbnds', type=int, default=15)
    parser.add_argument('--spinup', type=bool, default=True)
    parser.add_argument('--climate_file', type=str, default='/exports/geos.ed.ac.uk/iceocean/WFDE5_rof/')
    parser.add_argument('--glacier_rgi_id', type=str, default='')
    parser.add_argument('--y0', type=int, default=1980)
    parser.add_argument('--y1', type=int, default=2019)
    parser.add_argument('--catchment_path', type=str, default='')
    parser.add_argument('--simulation_name', type=str, default='')


    args = parser.parse_args()
    main(args)
