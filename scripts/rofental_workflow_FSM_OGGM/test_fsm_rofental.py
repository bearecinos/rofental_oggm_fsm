import argparse
import geopandas as gpd
import xarray as xr
from oggm import cfg, utils
from oggm import workflow, tasks
from oggm.sandbox import distribute_2d
from FSM_oggm_MB import FactorialSnowpackModel, process_wfde5_data, fsm_flowline_model_run

def main(args):
    reset = args.reset

    cfg.initialize(logging_level='DEBUG')

    #Configure parameters from arguments
    cfg.PARAMS['use_multiprocessing'] = args.use_multiprocessing
    cfg.PARAMS['mp_processes'] = args.mp_processes
    cfg.PARAMS['border'] = 80

    cfg.PARAMS['FSM_save_runoff'] = True
    cfg.PARAMS['FSM_runoff_frequency'] = 'D'

    # the following if True means FSM will be run with a fixed # of columns
    # independent on the number of glacier sectoins/elev bands. These
    # will be spaced evenly over the elev range (which will include the
    # downstream region - so im not sure it is a good idea to use at all).
    # The number of columns/bands can be set in
    # the FactorialSnowpackModel constructor as a kwarg, or through the
    # cfg.PARAMS['FSM_Nbnds'] parameter (kwarg overwrites global param)
    # or has a default of 15
    cfg.PARAMS['FSM_interpolate_bnds'] = False
    cfg.PARAMS['FSM_Nbnds'] = args.nbnds

    # if True, this will run FSM for one year when FactorialSnowpackModel is
    # initiated, and the results will be the saved "initial state"
    cfg.PARAMS['FSM_spinup'] = args.spinup

    # Here is how an FSM parameter (asmx) is set. this will create a
    # namelist entry with the value equal to the default
    cfg.PARAMS['FSM_param_asmx'] = args.asm_x

    _doc = ('A netcdf file containing dates and ' +
            'ice-based and snow-based runoff volume ' +
            'for each date interval')
    cfg.BASENAMES['FSM_runoff'] = ('FSM_runoff.nc', _doc)

    FactorialSnowpackModel.create_nml(reset=reset)


    print('Reset is set to ', reset)
    print('**Important set this to False to avoid '
          'resetting the glacier directory everytime this is ran!**')

    # this sets a temporary working directory. if you want to use a permanent
    # directory then uncomment and adapt the following line.
    cfg.PATHS['working_dir'] = utils.mkdir(args.working_dir, reset=reset)
    # cfg.PATHS['working_dir'] = '/exports/geos.ed.ac.uk/iceocean/dgoldber/FSM-OGGM'
    print('we are working here', cfg.PATHS['working_dir'])

    # bespoke path -- needs to be reset
    cfg.PATHS['climate_file'] = args.climate_file
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'

    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_compression'] = True
    cfg.PARAMS['use_tar_shapefiles'] = True
    cfg.PATHS['rgi_version'] = '62'
    cfg.PARAMS['use_temp_bias_from_file'] = True
    cfg.PARAMS['compress_climate_netcdf'] = False
    cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['store_fl_diagnostics'] = True

    base_url = ('https://cluster.klima.uni-bremen.de/~oggm/'
                'gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/')

    fr = utils.get_rgi_region_file(11, version='62', reset=reset)
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

    if reset:
        gdirs = workflow.init_glacier_directories(selection,
                                                  from_prepro_level=3,
                                                  prepro_base_url=base_url,
                                                  reset=reset,
                                                  force=reset)
    else:
        gdirs = workflow.init_glacier_directories(selection)

    elevation_band_task_list = [
        tasks.simple_glacier_masks,
        tasks.elevation_band_flowline,
        tasks.fixed_dx_elevation_band_flowline,
        tasks.compute_downstream_line,
        tasks.compute_downstream_bedshape,
        tasks.gridded_attributes,
        tasks.gridded_mb_attributes,
    ]

    print('multiprocessing' + str(cfg.PARAMS['use_multiprocessing']))

    for task in elevation_band_task_list:
        workflow.execute_entity_task(task, gdirs)

    # Distribute
    workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs)

    # Test that we have at least 21 variables on gridded_data.nc
    gdir = gdirs[0]
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        ds = ds.load()
        assert len(ds.count().variables.keys()) == 21

    y0 = args.y0
    y1 = args.y1

    workflow.execute_entity_task(process_wfde5_data, gdirs, y0=str(y0), y1=str(y1))
    print("DONE PROCESSING wfde5 data")

    workflow.execute_entity_task(tasks.apparent_mb_from_any_mb,
                                 gdirs,
                                 mb_model_class=FactorialSnowpackModel)

    workflow.calibrate_inversion_from_consensus(
        gdirs,
        apply_fs_on_mismatch=True,
        error_on_mismatch=True,  # if you're running many glaciers some might not work
        filter_inversion_output=True,  # this partly filters the over deepening due to
        #    # the equilibrium assumption for retreating glaciers (see. Figure 5 of Maussion et al. 2019)
        volume_m3_reference=None,  # here you could provide your own total volume estimate in m3
    )

    # finally create the dynamic flowlines
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    workflow.execute_entity_task(fsm_flowline_model_run, gdirs,
                                 climate_filename='climate_historical_fsm',
                                 output_filesuffix='_climate_historical_fsm',
                                 ys=y0, ye=y1)

    print("DONE running FSM")

    simulation_name = args.simulation_name

    workflow.execute_entity_task(distribute_2d.add_smoothed_glacier_topo, gdirs)
    workflow.execute_entity_task(distribute_2d.assign_points_to_band, gdirs)
    workflow.execute_entity_task(distribute_2d.distribute_thickness_from_simulation,
                                 gdirs, input_filesuffix=simulation_name)

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
