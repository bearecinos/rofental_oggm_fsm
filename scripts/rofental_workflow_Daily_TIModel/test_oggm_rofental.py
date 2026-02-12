import sys
import configparser
import geopandas as gpd
import xarray as xr
from oggm import cfg, utils, workflow, tasks
from oggm.shop.w5e5 import process_w5e5_data
from oggm.sandbox import distribute_2d
from oggm.core import massbalance
from functools import partial


def main(cfg_path):
    # ----------------------
    # 1) Read configuration file
    # ----------------------
    cp = configparser.ConfigParser()
    cp.read(cfg_path)

    gen_config  = cp['General']
    oggm_config = cp['OGGM']
    inp_config  = cp['InputData']
    outp_config = cp['Output']

    # ----------------------
    # 2) Parse general settings
    # ----------------------
    working_dir = gen_config.get('working_dir')          # string, may be blank
    reset       = gen_config.getboolean('reset')         # bool

    # ----------------------
    # 3) Initialize OGGM core
    # ----------------------
    cfg.initialize(logging_level='DEBUG')
    if working_dir.strip() == '':
        # If this params.ini working_dir is an empty string
        # we create a tmp dir in the /tmp/ folder of the user
        cfg.PATHS['working_dir'] = utils.gettempdir('OGGM_rofental', reset=reset)
    else:
        cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=reset)
    print(f"Working directory: {cfg.PATHS['working_dir']}")
    print(f"Reset = {reset}  (set False to preserve existing directories)")

    # ----------------------
    # 4) OGGM parameters
    # ----------------------
    cfg.PARAMS['use_multiprocessing'] = oggm_config.getboolean('use_multiprocessing')
    cfg.PARAMS['mp_processes']        = oggm_config.getint('mp_processes')
    cfg.PARAMS['border']              = oggm_config.getint('border', fallback=80)

    # ----------------------
    # 6) Climate & I/O paths
    # ----------------------
    catchment_path = inp_config.get('catchment_path')

    # ----------------------
    # 7) OGGM run setup
    # ----------------------
    y0 = inp_config.getint('y0')
    y1 = inp_config.getint('y1')
    simulation_name = outp_config.get('simulation_name')
    useThreeStep = inp_config.getboolean('useThreeStep')
    if useThreeStep is None:
        useThreeStep = False
    useDaily = inp_config.getboolean('useDaily')
    if useDaily is None:
        useDaily = False

    # “Always-on” OGGM flags
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_compression'] = True
    cfg.PARAMS['use_tar_shapefiles'] = True
    cfg.PATHS['rgi_version'] = '62'
    cfg.PARAMS['use_temp_bias_from_file'] = True
    cfg.PARAMS['compress_climate_netcdf'] = False
    cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['store_fl_diagnostics'] = True

    # ----------------------
    # 7) Load RGI & catchment
    # ----------------------
    base_url = (
        "https://cluster.klima.uni-bremen.de/~oggm/"
        "gdirs/oggm_v1.6/L3-L5_files/2023.1/"
        "elev_bands/W5E5_w_data/"
    )
    fr  = utils.get_rgi_region_file(11, version='62', reset=reset)
    gdf = gpd.read_file(fr)

    rof_shp = gpd.read_file(catchment_path)

    rof_sel = gdf.clip(rof_shp)
    rof_sel = rof_sel.sort_values('Area', ascending=False)

    # Grab the raw string (or None if the key is missing)
    rgi_id = inp_config.get('glacier_rgi_id', fallback=None)

    # If the user literally wrote “None”, turn that into a Python None
    if rgi_id == 'None':
        rgi_id = None

    # Now select
    if rgi_id is not None:
        selection = rof_sel[rof_sel.RGIId == rgi_id]
    else:
        selection = rof_sel

    # Build a "safe" DailyTIModel class that disables the calibration-source check
    # TODO: this needs checking with Fabien as there is no docs for this model yet
    # But my thoughts are is it worth re-calibrating a Daily SMB with Hugonnet if this
    # is just a yearly average?? probably not

    DailyTI_nocheck = partial(massbalance.DailyTIModel, check_calib_params=False)

    # new workflow for dev branch
    if not useDaily:
        MBModel = massbalance.MonthlyTIModel
        settings_filesuffix=''
        observations_filesuffix=''
        climate_filename="climate_historical"
    else:
        MBModel = DailyTI_nocheck
        settings_filesuffix='_daily'
        observations_filesuffix='_daily'
        climate_filename="climate_historical_daily"
        cfg.PARAMS['baseline_climate'] = 'GSWP3_W5E5_daily'
        
        if useThreeStep:
            MBModel = DailyTI_nocheck


    # initialize glacier directories
    if reset:
        gdirs = workflow.init_glacier_directories(
            selection,
            from_prepro_level=3,
            prepro_base_url=base_url,
            reset=True, force=True
        )
    else:
        gdirs = workflow.init_glacier_directories(selection)

#    for gdir in gdirs:
#        utils.ModelSettings(gdir, filesuffix=settings_filesuffix, parent_filesuffix='')        

    # ----------------------
    # 8) Preprocessing workflow
    # ----------------------
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

    # ----------------------
    # 9) MB runs
    # ----------------------
    # add climate data to gdir
    # Monthly climate first
    workflow.execute_entity_task(process_w5e5_data, gdirs, daily=False)

    # Daily climate for DailyTIModel
    workflow.execute_entity_task(process_w5e5_data, gdirs, daily=True)


    # 1) if we specify, do an informed three step calibration. If we are using a Daily TIM, 
    #    then this should work (comm. patrick schmitt). If we are using Monthly, then this 
    #    should return the params in the prepro dir

    if useThreeStep:
      workflow.execute_entity_task(
        tasks.mb_calibration_from_hugonnet_mb,
        gdirs,
        settings_filesuffix=settings_filesuffix,
        observations_filesuffix=observations_filesuffix,
        informed_threestep=True,
        overwrite_gdir=True,
        write_to_gdir=True,
        mb_model_class=MBModel)


    # 2) Apparent MB (if you need it)
    workflow.execute_entity_task(
        tasks.apparent_mb_from_any_mb,
        gdirs,
        mb_model_class=MBModel,
    )

    # We do this for FSM so lets do it here too, though it likely is not needed
    workflow.calibrate_inversion_from_consensus(
        gdirs,
        apply_fs_on_mismatch=True,
        error_on_mismatch=True,  # if you're running many glaciers some might not work
        filter_inversion_output=True,  # this partly filters the over deepening due to
        #    # the equilibrium assumption for retreating glaciers (see. Figure 5 of Maussion et al. 2019)
    );


    # I think the step below might redundant if I do a spin up
    # finally create the dynamic flowlines
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    # Fixed-geometry spinup run (historical)
    spinup_start_yr = 1979
    workflow.execute_entity_task(
        tasks.run_from_climate_data,
        gdirs,
        ys=y0,ye=y1,
        climate_filename=climate_filename,
        mb_model_class=MBModel,
        output_filesuffix=simulation_name,
    )

    # Now we do the historical run
    # Forward simulation with hydro diagnostics (control)

    # if we are not looking at runoff, let's not do this for now
    if False:
     workflow.execute_entity_task(
        tasks.run_with_hydro,
        gdirs,
        run_task=tasks.run_from_climate_data,
        climate_filename="climate_historical_daily",
        mb_model_class=MBModel,
        init_model_filesuffix="_historical",
        output_filesuffix="_historical_runhydro",
        ys=y0,ye=y1
        # keep defaults: store_monthly_step=False, mb_elev_feedback='annual'
     )

    workflow.execute_entity_task(distribute_2d.add_smoothed_glacier_topo, gdirs)
    workflow.execute_entity_task(distribute_2d.assign_points_to_band, gdirs)
    workflow.execute_entity_task(distribute_2d.distribute_thickness_from_simulation,
                                 gdirs, input_filesuffix=simulation_name)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_fsm_rofental.py <config.ini>")
        sys.exit(1)
    main(sys.argv[1])
