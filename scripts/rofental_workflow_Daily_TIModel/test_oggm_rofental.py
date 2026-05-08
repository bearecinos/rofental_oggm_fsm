# Module logger
import logging
import sys
import configparser
import geopandas as gpd
import xarray as xr
from functools import partial
import numpy as np

from oggm import cfg, utils, workflow, tasks, DEFAULT_BASE_URL
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.shop.w5e5 import process_w5e5_data
from oggm.sandbox import distribute_2d
from oggm.core import massbalance
from oggm.core.flowline import FileModel
from oggm import entity_task
from oggm.utils import float_years_timeseries


log = logging.getLogger(__name__)


def daily_storage_template(testing: bool):
    """This creates a 366-element array for storing one year of daily values.
    :testing if False fills unused slots with 0
    if True fill unused slots with NaN
    """
    return np.full(366, np.nan if testing else 0.0)


def daily_storage_map_index(ndays: int, testing: bool):
    """Small helper function that maps a real day index to the
    fixed 366-day storage.
    :ndays: number of days to map
    :testing: Bool
    Return an index-mapping function for daily storage."""

    def map_index(iday: int) -> int:
        if ndays == 365 and testing and iday >= 59:
            return iday + 1
        return iday

    return map_index


def daily_storage_setup(ndays: int, testing: bool):
    """Return daily storage template and index mapper."""
    arr = daily_storage_template(testing)
    map_index = daily_storage_map_index(ndays, testing)
    return arr, map_index


@entity_task(log)
def run_with_hydro_daily(gdir,
                         settings_filesuffix='',
                         run_task=None,
                         fixed_geometry_spinup_yr=None,
                         ref_area_from_y0=False,
                         ref_area_yr=None,
                         ref_geometry_filesuffix=None,
                         Testing=False,
                         store_annual=True,
                         **kwargs,
                         ):
    """Run the flowline model and add daily hydro diagnostics.
    Code addapted from MB-sandbox, Schuster L. et al. (2023)
    Hanus et al. GMD (2024).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix: str
        You can use a different set of settings by providing a filesuffix. This
        is useful for sensitivity experiments. Code-wise the settings_filesuffix
        is set in the @entity-task decorater.
    run_task : func
        any of the `run_*`` tasks in the oggm.flowline module.
        The mass balance model used needs to have the `add_climate` output
        kwarg available though.
    fixed_geometry_spinup_yr : int
        if set to an integer, the model will artificially prolongate
        all outputs of run_until_and_store to encompass all time stamps
        starting from the chosen year. The only output affected are the
        glacier wide diagnostic files - all other outputs are set
        to constants during "spinup"
    ref_area_from_y0 : bool
        overwrite ref_area_yr to the first year of the timeseries
    ref_area_yr : int
        the hydrological output is computed over a reference area, which
        per default is the largest area covered by the glacier in the simulation
        period. Use this kwarg to force a specific area to the state of the
        glacier at the provided simulation year.
    ref_geometry_filesuffix : str
        this kwarg allows to copy the reference area from a previous simulation
        (useful for projections with historical spinup for example).
        Set to a model_geometry file filesuffix that is present in the
        current directory (e.g. `_historical` for pre-processed gdirs).
        If set, ref_area_yr and ref_area_from_y0 refer to this file instead.
    Testing: controls leap-year handling in storage
    store_annual: whether to also save annual aggregates
    **kwargs : all valid kwargs for ``run_task`

    Notes
    -----
    - Geometry is held fixed within each simulation year, as in OGGM's
      monthly `run_with_hydro`.
    - Requires an MB model implementing `get_daily_mb(..., add_climate=True)`.
      This is designed for `DailyTIModel`-like classes.
    - Daily outputs are stored on a 366-day axis. In non-leap years:
      - `Testing=False`: day 366 stays at 0
      - `Testing=True`: Feb 29 is NaN and all subsequent days are shifted by 1
    """

    kwargs['return_value'] = True

    # Check if kwargs are compatible
    if kwargs.get('store_monthly_step', False):
        raise InvalidParamsError(
            'run_with_hydro_daily only compatible with store_monthly_step=False.'
        )
    if kwargs.get('mb_elev_feedback', 'annual') != 'annual':
        raise InvalidParamsError(
            "run_with_hydro_daily only compatible with mb_elev_feedback='annual'."
        )

    model = run_task(gdir, settings_filesuffix=settings_filesuffix, **kwargs)
    if model is None:
        raise InvalidWorkflowError(
            f'The run task ({run_task.__name__}) did not run successfully.'
        )

    # Check if the user requested a spin-up
    do_spinup = fixed_geometry_spinup_yr is not None
    if do_spinup:
        start_dyna_model_yr = model.y0

    mb_mod = model.mb_model

    # The code below is to pick from which geometry we start computing runoff
    suffix = kwargs.get('output_filesuffix', settings_filesuffix)
    if suffix is None:
        suffix = settings_filesuffix

    fmod = FileModel(gdir.get_filepath('model_geometry', filesuffix=suffix))
    years = np.asarray(fmod.years[:-1], dtype=int)
    if len(years) == 0:
        raise InvalidWorkflowError('No model years available in model_geometry.')

    if ref_geometry_filesuffix:
        if not ref_area_from_y0 and ref_area_yr is None:
            raise InvalidParamsError(
                'If `ref_geometry_filesuffix` is set, specify `ref_area_from_y0` '
                'or `ref_area_yr`.'
            )
        fmod_ref = FileModel(
            gdir.get_filepath('model_geometry', filesuffix=ref_geometry_filesuffix)
        )
    else:
        fmod_ref = fmod

    # Check if the first reference year should define the reference area
    if ref_area_from_y0:
        ref_area_yr = int(fmod_ref.years[0])

    if ref_area_yr is not None:
        if ref_area_yr not in fmod_ref.years:
            raise InvalidParamsError('The chosen ref_area_yr is not available.')
        # if specified this moves the reference geometry model to that year
        fmod_ref.run_until(ref_area_yr)

    # pre-flowline arrays where we will store the data
    bin_area_2ds = []
    bin_elev_2ds = []
    ref_areas = []
    snow_buckets = []

    for fl in fmod_ref.fls:
        ref_area = fl.bin_area_m2.copy()
        ref_areas.append(ref_area)
        snow_buckets.append(np.zeros_like(ref_area, dtype=np.float64))

        shape = (len(years), len(ref_area))
        bin_area_2ds.append(np.empty(shape, np.float64)) # glacier area in each bin for each year
        bin_elev_2ds.append(np.empty(shape, np.float64)) # glacier elevation in each bin for each year

    for i, yr in enumerate(years):
        fmod.run_until(int(yr))
        # geometry is fixed within each year
        # geometry only changes between years
        for fl, bin_area_2d, bin_elev_2d in zip(fmod.fls, bin_area_2ds, bin_elev_2ds):
            bin_area_2d[i, :] = fl.bin_area_m2
            bin_elev_2d[i, :] = fl.surface_h
    # If no specific reference year is chosen,
    # the code uses the maximum glacier extent during the whole simulation.
    if ref_area_yr is None:
        for ref_area, bin_area_2d in zip(ref_areas, bin_area_2ds):
            ref_area[:] = np.nanmax(bin_area_2d, axis=0)

    ntime = len(years) + 1
    oshape = (ntime, 366)
    seconds = cfg.SEC_IN_DAY

    out = {
        'off_area': {
            'description': 'Off-glacier area',
            'unit': 'm 2',
            'data': np.zeros(ntime),
        },
        'on_area': {
            'description': 'On-glacier area',
            'unit': 'm 2',
            'data': np.zeros(ntime),
        },
        'melt_off_glacier': {
            'description': 'Off-glacier melt',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
        'melt_on_glacier': {
            'description': 'On-glacier melt',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
        'melt_residual_off_glacier': {
            'description': 'Off-glacier melt due to MB model residual',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
        'melt_residual_on_glacier': {
            'description': 'On-glacier melt due to MB model residual',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
        'liq_prcp_off_glacier': {
            'description': 'Off-glacier liquid precipitation',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
        'liq_prcp_on_glacier': {
            'description': 'On-glacier liquid precipitation',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
        'snowfall_off_glacier': {
            'description': 'Off-glacier solid precipitation',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
        'snowfall_on_glacier': {
            'description': 'On-glacier solid precipitation',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
        'snow_bucket': {
            'description': 'Off-glacier snow reservoir (state variable, end of day)',
            'unit': 'kg',
            'data': np.zeros(oshape),
        },
        'model_mb': {
            'description': 'Annual mass balance from dynamical model',
            'unit': 'kg yr-1',
            'data': np.zeros(ntime),
        },
        'residual_mb': {
            'description': 'Difference (before correction) between MB model and dyn model melt',
            'unit': 'kg d-1',
            'data': np.zeros(oshape),
        },
    }

    fmod.run_until(int(years[0]))
    prev_model_vol = fmod.volume_m3

    for i, yr in enumerate(years):
        yr = int(yr)
        ndays = mb_mod.days_in_year(yr)
        if ndays not in (365, 366):
            raise InvalidWorkflowError(f'Unsupported number of days in year: {ndays}')

        day_fyrs = float_years_timeseries(y0=yr, y1=yr + 1, daily=True)[:-1]
        if len(day_fyrs) != ndays:
            raise InvalidWorkflowError(
                f'Unexpected daily time axis for year {yr}: got {len(day_fyrs)}, expected {ndays}.'
            )

        _, map_index = daily_storage_setup(ndays, Testing)

        off_area_out = 0.0
        on_area_out = 0.0

        for fl_id, (ref_area, snow_bucket, bin_area_2d, bin_elev_2d) in enumerate(
                zip(ref_areas, snow_buckets, bin_area_2ds, bin_elev_2ds)
        ):
            bin_area = bin_area_2d[i, :]
            bin_elev = bin_elev_2d[i, :]
            off_area = utils.clip_min(ref_area - bin_area, 0)

            off_area_out += np.sum(off_area)
            on_area_out += np.sum(bin_area)

            for iday, fyr in enumerate(day_fyrs):
                jday = map_index(iday)

                try:
                    mb_out = mb_mod.get_daily_mb(
                        bin_elev, fl_id=fl_id, year=fyr, add_climate=True
                    )
                    mb, _, _, prcp, prcpsol = mb_out
                except AttributeError as err:
                    raise InvalidWorkflowError(
                        'run_with_hydro_daily requires a daily MB model with '
                        '`get_daily_mb(..., add_climate=True)`.'
                    ) from err
                except ValueError as err:
                    if 'too many values to unpack' in str(err):
                        raise InvalidWorkflowError(
                            'run_with_hydro_daily requires a MB model able to add '
                            'climate info to `get_daily_mb`.'
                        ) from err
                    raise

                mb = np.asarray(mb, dtype=np.float64)
                prcp = np.asarray(prcp, dtype=np.float64)
                prcpsol = np.asarray(prcpsol, dtype=np.float64)

                mb_mass = mb * seconds * gdir.settings['ice_density']
                mb_bias = mb_mod.bias * seconds / mb_mod.sec_in_year(fyr)

                liq_prcp_on_g = (prcp - prcpsol) * bin_area
                liq_prcp_off_g = (prcp - prcpsol) * off_area

                prcpsol_on_g = prcpsol * bin_area
                prcpsol_off_g = prcpsol * off_area

                melt_on_g = (prcpsol - mb_mass) * bin_area
                melt_off_g = (prcpsol - mb_mass) * off_area

                if mb_mod.bias == 0:
                    melt_on_g = utils.clip_min(melt_on_g, 0)
                    melt_off_g = utils.clip_min(melt_off_g, 0)

                bias_on_g = mb_bias * bin_area
                bias_off_g = mb_bias * off_area

                snow_bucket += prcpsol_off_g
                melt_off_g = np.minimum(utils.clip_min(melt_off_g, 0), snow_bucket)
                snow_bucket -= melt_off_g

                out['melt_off_glacier']['data'][i, jday] += np.sum(melt_off_g)
                out['melt_on_glacier']['data'][i, jday] += np.sum(melt_on_g)
                out['melt_residual_off_glacier']['data'][i, jday] += np.sum(bias_off_g)
                out['melt_residual_on_glacier']['data'][i, jday] += np.sum(bias_on_g)
                out['liq_prcp_off_glacier']['data'][i, jday] += np.sum(liq_prcp_off_g)
                out['liq_prcp_on_glacier']['data'][i, jday] += np.sum(liq_prcp_on_g)
                out['snowfall_off_glacier']['data'][i, jday] += np.sum(prcpsol_off_g)
                out['snowfall_on_glacier']['data'][i, jday] += np.sum(prcpsol_on_g)
                out['snow_bucket']['data'][i, jday] += np.sum(snow_bucket)

        out['off_area']['data'][i] = off_area_out
        out['on_area']['data'][i] = on_area_out

        if do_spinup and yr < start_dyna_model_yr:
            model_mb = (
                    out['snowfall_on_glacier']['data'][i, :].sum()
                    - out['melt_on_glacier']['data'][i, :].sum()
            )
            residual_total = 0.0
        else:
            fmod.run_until(yr + 1)
            model_mb = (fmod.volume_m3 - prev_model_vol) * gdir.settings['ice_density']
            prev_model_vol = fmod.volume_m3

            reconstructed_mb = (
                    out['snowfall_on_glacier']['data'][i, :].sum()
                    - out['melt_on_glacier']['data'][i, :].sum()
            )
            residual_total = model_mb - reconstructed_mb

        g_melt = out['melt_on_glacier']['data'][i, :].copy()
        valid = np.arange(366) if ndays == 366 else np.array(
            [map_index(d) for d in range(ndays)], dtype=int
        )

        if residual_total == 0:
            residual_series = np.zeros(366)
        else:
            asum = np.nansum(g_melt[valid])
            if asum > 1e-7 and (residual_total / asum < 1):
                fac = 1 - residual_total / asum
                corrected = g_melt[valid] * fac
                residual_series = np.zeros(366)
                residual_series[valid] = g_melt[valid] - corrected
                out['melt_on_glacier']['data'][i, valid] = corrected
            else:
                residual_daily = residual_total / ndays
                residual_series = np.zeros(366)
                residual_series[valid] = residual_daily
                tmp = g_melt[valid] - residual_daily
                out['snowfall_on_glacier']['data'][i, valid] -= utils.clip_max(tmp, 0)
                out['melt_on_glacier']['data'][i, valid] = utils.clip_min(tmp, 0)

        out['model_mb']['data'][i] = model_mb
        out['residual_mb']['data'][i, :] = residual_series

        if ndays == 365 and Testing:
            for var in [
                'melt_off_glacier',
                'melt_on_glacier',
                'melt_residual_off_glacier',
                'melt_residual_on_glacier',
                'liq_prcp_off_glacier',
                'liq_prcp_on_glacier',
                'snowfall_off_glacier',
                'snowfall_on_glacier',
                'snow_bucket',
                'residual_mb',
            ]:
                out[var]['data'][i, 59] = np.nan

    out_vars = gdir.settings['store_diagnostic_variables']
    ods = xr.Dataset()
    ods.coords['time'] = fmod.years
    ods.coords['day_2d'] = ('day_2d', np.arange(1, 367))
    ods.coords['calendar_day_2d'] = ('day_2d', np.arange(1, 367))

    sm = gdir.settings['hydro_month_' + mb_mod.hemisphere]
    hydro_start = utils.date_to_floatyear(2001, sm, 1)
    hydro_start_day = utils.floatyear_to_date(hydro_start, return_day=True)[2]
    ods.coords['hydro_day_2d'] = (
        'day_2d',
        ((np.arange(366) + hydro_start_day - 1) % 366) + 1,
    )

    for varname, meta in out.items():
        data = meta['data']
        attrs = {k: v for k, v in meta.items() if k != 'data'}

        if varname not in out_vars:
            continue

        if data.ndim == 2:
            if store_annual:
                if varname == 'snow_bucket':
                    annual = np.full(ntime, np.nan)
                    annual[:-1] = np.nanmax(data[:-1, :], axis=1)
                    ods[varname] = ('time', annual)
                else:
                    annual = np.nansum(data, axis=1)
                    annual[-1] = np.nan
                    ods[varname] = ('time', annual)
                    ods[varname].attrs.update(attrs)

            daily_name = f'{varname}_daily'
            ods[daily_name] = (('time', 'day_2d'), data)
            ods[daily_name].attrs.update(attrs)

            if varname == 'snow_bucket' and varname in ods:
                ods[varname].attrs.update(attrs)
        else:
            annual = data.copy()
            annual[-1] = np.nan
            ods[varname] = ('time', annual)
            ods[varname].attrs.update(attrs)

    fpath = gdir.get_filepath('model_diagnostics', filesuffix=suffix)
    ods.to_netcdf(fpath, mode='a')
    return ods


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

    # And we want to always use daily MB TIM by default (but user can decide)
    useDaily = inp_config.getboolean('useDaily', fallback=True)
    if useDaily is None:
        useDaily = True

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

    print('multiprocessing' + str(cfg.PARAMS['use_multiprocessing']))

#TODO: check if the lines below are needed
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
    for task in elevation_band_task_list:
        workflow.execute_entity_task(task, gdirs)

    # Distribute
    workflow.execute_entity_task(tasks.distribute_thickness_per_altitude, gdirs)

    # TODO: this needs to be remove if we dont do elevation_band_task_list
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

    # If we want a DailyTIModel
    if useDaily:
        # Daily climate for DailyTIModel
        workflow.execute_entity_task(process_w5e5_data, gdirs, daily=True)

        # We re-do the calibration starting with the precipitation factor
        # and temperature bias from the glacier directory calibration
        # We only tune the melt factor to match Hugonnet reference mb
        # and period used in the MonthlyTIM calibration
        for gdir in gdirs:
            mb_calib = gdir.read_json('mb_calib')

            tasks.mb_calibration_from_scalar_mb(
                gdir,
                ref_mb=mb_calib['reference_mb'],
                ref_mb_err=mb_calib['reference_mb_err'],
                ref_mb_period=mb_calib['reference_period'],
                settings_filesuffix=settings_filesuffix,
                observations_filesuffix=observations_filesuffix,
                write_to_gdir=True,
                overwrite_gdir=True,
                overwrite_observations=True,
                calibrate_param1='melt_f',
                calibrate_param2=None,
                calibrate_param3=None,
                prcp_fac=mb_calib['prcp_fac'],
                temp_bias=mb_calib['temp_bias'],
                mb_model_class=MBModel,
                filename=climate_filename
            )

    # compute apparent MB
    workflow.execute_entity_task(tasks.apparent_mb_from_any_mb,
                                 gdirs,
                                 mb_model_class=MBModel,
    )

    # calibrate ice dynamic params
    workflow.calibrate_inversion_from_consensus(
        gdirs,
        apply_fs_on_mismatch=True,
        error_on_mismatch=True,  # if you're running many glaciers some might not work
        filter_inversion_output=True,  # this partly filters the over deepening due to
        #    # the equilibrium assumption for retreating glaciers (see. Figure 5 of Maussion et al. 2019)
    );

    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    cfg.PARAMS['store_model_geometry'] = True
    if useDaily:
        workflow.execute_entity_task(
            run_with_hydro_daily,
            gdirs,
            settings_filesuffix=settings_filesuffix,
            run_task=tasks.run_from_climate_data,
            ys=y0, ye=y1,
            climate_filename=climate_filename,
            climate_input_filesuffix='',
            mb_model_class=DailyTI_nocheck,
            output_filesuffix=simulation_name,
            mb_elev_feedback='annual',
            store_monthly_step=False,
            store_annual=True,
        )
    else:
        # Monthly OGGM hydro
        workflow.execute_entity_task(
            tasks.run_with_hydro,
            gdirs,
            settings_filesuffix=settings_filesuffix,
            run_task=tasks.run_from_climate_data,
            ys=y0,
            ye=y1,
            climate_filename=climate_filename,
            mb_model_class=MBModel,
            store_monthly_hydro=True,
            output_filesuffix=simulation_name,
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
