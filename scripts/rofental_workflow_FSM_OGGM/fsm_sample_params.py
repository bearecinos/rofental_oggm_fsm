import sys
import configparser
import geopandas as gpd
import xarray as xr
from oggm import cfg, utils, workflow, tasks
from oggm.sandbox import distribute_2d
from IPython import embed
import json
import numpy as np

from oggm.cfg import SEC_IN_YEAR
rho = None

def get_WGMS_data(path, years, glac_id, get_mb=True, get_winter_mb=True, get_profile=True, doMean=False):

    # HEF: 491 summer bal non-nan 2013-2025
    # KEF: 507 no summer bal
    # VER: 489 summer bal 1966 on!

    import pandas as pd
    return_dict = dict()

    if (get_mb or get_winter_mb):
        
        df = pd.read_csv (path + '/mass_balance.csv')
        df = df[df.glacier_id==glac_id]
        df = df[df.year.isin(years)]
        bal = df['annual_balance'].values # mwe
        bal_unc = df['annual_balance_unc'].values # mwe
        winter_bal = df['winter_balance'].values # mwe
        winter_bal_unc = df['winter_balance_unc'].values # mwe

    if get_profile:
        assert years[0]>1965, "earlier years have inconsistent bands"
        df = pd.read_csv (path + '/mass_balance_band.csv')
        df = df[df.glacier_id==glac_id]
        df = df[df.year.isin(years)]
        max_lower = 0
        min_upper = 1e5
        for i in range(len(years)):
            lower_band = np.sort(df[df.year==years[i]]['lower_elevation'].values)
            upper_band = np.sort(df[df.year==years[i]]['upper_elevation'].values)
            max_lower = max(min(lower_band),max_lower)
            min_upper = min(max(lower_band),min_upper)
        ind_retain = (lower_band>=max_lower) & (lower_band<=min_upper)
        lower_band = lower_band[ind_retain]
        upper_band = upper_band[ind_retain]
        mb_profile = np.zeros(len(lower_band))
        mb_profile_unc = np.zeros(len(lower_band))

        unc_nonnan=0
        for i in range(len(years)):
            dfyr = df[df.year==years[i]]
            dfyr = dfyr[dfyr['lower_elevation'].isin(lower_band)]
            assert len(dfyr)==len(lower_band), "inconsistent bands"
            mb_profile = mb_profile + \
                dfyr.sort_values(by=["lower_elevation"])['annual_balance'].values
            uncs = dfyr.sort_values(by=["lower_elevation"])['annual_balance_unc'].values
            if not np.isnan(uncs).any():
                unc_nonnan = unc_nonnan+1
                mb_profile_unc = mb_profile_unc + uncs**2
        mb_profile = mb_profile / len(years)
        if unc_nonnan>0:
            mb_profile_unc= np.sqrt(mb_profile_unc / unc_nonnan)
        else:
            mb_profile_unc[:] = 0.05


    if get_mb:
        return_dict['mb_annual_mwe'] = bal
        return_dict['mb_annual_mwe_unc'] = bal_unc
    if get_winter_mb:
        return_dict['mb_winter_mwe'] = winter_bal
        return_dict['mb_winter_mwe_unc'] = winter_bal_unc
    if get_profile:
        return_dict['mb_profile_mwe'] = mb_profile
        return_dict['mb_profile_mwe_unc'] = mb_profile_unc
        return_dict['mb_profile_lower'] = lower_band
        return_dict['mb_profile_upper'] = upper_band

    return return_dict


def main(cfg_path):
    # ----------------------
    # 1) Read configuration file
    # ----------------------
    cp = configparser.ConfigParser()
    cp.read(cfg_path)

    gen_config  = cp['General']
    oggm_config = cp['OGGM']
    fsm_config  = cp['FSM_OGGM']
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
        cfg.PATHS['working_dir'] = utils.gettempdir('FSM_rofental', reset=reset)
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
    # 5) FSM parameters
    # ----------------------
    cfg.PARAMS['FSM_save_runoff']        = fsm_config.getboolean('FSM_save_runoff',fallback=True)
    cfg.PARAMS['FSM_runoff_frequency']   = fsm_config.get('FSM_runoff_frequency',fallback='D')
    cfg.PARAMS['FSM_spinup']             = fsm_config.getboolean('FSM_spinup',fallback=True)
    cfg.PARAMS['FSM_interpolate_bnds']   = fsm_config.getboolean('FSM_interpolate_bnds',fallback=False) # note: nbnds can only be set if interpolate_bnds is True
    cfg.PARAMS['FSM_Nbnds']              = fsm_config.getint('FSM_Nbnds',fallback=None)
    rho = cfg.PARAMS['density_ice']

    # important: parameters for namelist must start with "FSM_param_"
    cpdict = dict(fsm_config)
    sens_params = []

    for key in cpdict.keys():
        if (key[:10] == 'FSM_param_'):

            valstr = cpdict[key]
            val = json.loads(valstr)

            if isinstance(val,list):
                cfg.PARAMS[key] = val[0]
                sens_params = sens_params + [key[10:]]
            else:
                cfg.PARAMS[key] = val

    oggm_fsm_path                        = fsm_config.get('FSM-OGGM_path')
    from FSM_oggm_MB import FactorialSnowpackModel, process_wfde5_data
    sys.path.append(oggm_fsm_path)

    # write/reset the FSM namelist
    FactorialSnowpackModel.create_nml(reset=reset)

    # ----------------------
    # 6) Climate & I/O paths
    # ----------------------
    cfg.PATHS['climate_file']   = inp_config.get('climate_file')
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'
    catchment_path = inp_config.get('catchment_path')

    # ----------------------
    # 7) OGGM run setup
    # ----------------------
    y0 = inp_config.getint('y0')
    y1 = inp_config.getint('y1')
    years_cost = json.loads(inp_config.get('years_cost'))
    simulation_name = outp_config.get('simulation_name')

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
    wgms_id = inp_config.get('glacier_wgms_id', fallback=491)

        # HEF: 491 summer bal non-nan 2013-2025
    # KEF: 507 no summer bal
    # VER: 489 summer bal 1966 on!
    wgms_to_rgi = { 491: "RGI60-11.00897", 
                    507: "RGI60-11.00787",
                    489: "RGI60-11.00719" }

    # Now select
    selection = rof_sel[rof_sel.RGIId == wgms_to_rgi[wgms_id]]

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
    # 9) MB calibration & FSM runs
    # ----------------------

    workflow.execute_entity_task(process_wfde5_data, gdirs, y0=str(y0), y1=str(y1))
    print("DONE PROCESSING wfde5 data")

    gdir = gdirs[0]
    fls = gdir.read_pickle('inversion_flowlines')
    areas = fls[0].bin_area_m2

    mb_model = FactorialSnowpackModel(gdir=gdir)
    years_compute = np.array([years_cost[0]-1] + years_cost)
    mb_output = None
    for i, year in enumerate(years_compute):
        mb_year = mb_model.get_mb(heights=fls[0].surface_h, year=year, fls=fls, reset_state=True, monthly=True)
        if mb_output is None:
            mb_output = mb_year
        else:
            mb_output = np.vstack((mb_output,mb_year))





if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_fsm_rofental.py <config.ini>")
        sys.exit(1)
    main(sys.argv[1])
