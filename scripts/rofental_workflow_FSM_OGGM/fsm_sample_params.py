import sys
import os
import configparser
import geopandas as gpd
import xarray as xr
from oggm import cfg, utils, workflow, tasks
from oggm.sandbox import distribute_2d
import json
import numpy as np
import pandas as pd
from SALib import ProblemSpec
from scipy.interpolate import interp1d


from oggm.cfg import SEC_IN_YEAR
rho = 1000.

def get_WGMS_data(path, years, glac_id, get_mb=True, get_winter_mb=True, get_profile=True):

    """
    A function to retrieve WGMS mass balance data for a reference glacier.

    Parameters
    ----------
    path: 	path to WGMS FoG data folder containing csv files.
    years:	the years for which CF will be evaluated.
    glac_id:    the wgms ID
    get_mb, get_winter_mb, get_profile: flags to return different meas types

    """

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

    return_dict['mb_years'] = years

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

def get_cost(mb_output, mb_output_years, wgms_data, areas, elevs, \
	profile_cost=True, mb_cost=True, winter_mb_cost=True, doMean=False):

    profile_misfit = None
    mb_misfit = None
    winter_mb_misfit = None 

    mb_output_mon = np.repeat(mb_output_years,12)

    inds = np.isin(mb_output_mon,wgms_data['mb_years'])

    if profile_cost:
	
    	# take average over all mb_output
        mb_profile = np.mean(mb_output[inds,:],0)
	    # convert from m/s to mwe per year
        mb_profile = mb_profile * SEC_IN_YEAR * rho / 1000.
    	# interpolate to wgms_bands
        func = interp1d(elevs, mb_profile, bounds_error=False)
        mb_interp = func(.5*(wgms_data['mb_profile_lower']+wgms_data['mb_profile_upper']))
    	# sum squared difference
        profile_misfit = np.sqrt(np.nanmean( (mb_interp-wgms_data['mb_profile_mwe'])**2 / wgms_data['mb_profile_mwe_unc']**2 ))

    if mb_cost:

        mb_annual = (mb_output[inds,:]).reshape(-1,12,mb_output.shape[1]).mean(axis=1) # result is in m/s
        mb_series = np.matmul(mb_annual, areas) # result is m3/s
        mb_series = mb_series * rho / 1000. * SEC_IN_YEAR / areas.sum() # result in mwe/yr
        if not doMean:
            mb_misfit = np.sqrt(( (mb_series - wgms_data['mb_annual_mwe'])**2 / wgms_data['mb_annual_mwe_unc']**2 ).mean())
        else:
            mb_misfit =  np.abs(  mb_series.mean() - np.mean(wgms_data['mb_annual_mwe']) ) / wgms_data['mb_annual_mwe_unc'][0]

    if winter_mb_cost:

        mb_winter = None
        for i in range(1,len(mb_output_years)):

            indwin = np.array([-3,-2,-1,0,1,2,3])+12*i
            mb_season = mb_output[indwin,:].mean(axis=0)
            if mb_winter is None:
                mb_winter = mb_season
            else:
                mb_winter = np.vstack((mb_winter,mb_season))

        mb_series = np.matmul(mb_winter, areas) # result is m3/s
        mb_series = mb_series * rho / 1000. * SEC_IN_YEAR / areas.sum() # result in mwe/yr

        if not doMean:
            winter_mb_misfit = np.sqrt( np.nanmean( (mb_series - wgms_data['mb_winter_mwe'])**2 / wgms_data['mb_winter_mwe_unc']**2 ) )
        else:
            winter_mb_misfit =  np.abs(  mb_series.mean() - np.mean(wgms_data['mb_winter_mwe']) ) / wgms_data['mb_winter_mwe_unc'][0]
	
    return profile_misfit, mb_misfit, winter_mb_misfit

def main(cfg_path):
    # ----------------------
    # 1) Read configuration file
    # ----------------------
    cp = configparser.ConfigParser()
    cp.optionxform = str
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

    _doc = ("A netcdf file containing dates and "
            "ice‐based and snow‐based runoff volume for each date interval")
    cfg.BASENAMES['FSM_runoff'] = ('FSM_runoff.nc', _doc)

    global rho
    rho = cfg.PARAMS['ice_density']

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

    # important: parameters for namelist must start with "FSM_param_"
    cpdict = dict(fsm_config)
    sens_params = []
    sens_bounds = []

    for key in cpdict.keys():
        if (key[:10] == 'FSM_param_'):

            valstr = cpdict[key]
            val = json.loads(valstr)

            if isinstance(val,list):
                cfg.PARAMS[key] = val[0]
                sens_params.append(key)
                sens_bounds.append(val)
            else:
                cfg.PARAMS[key] = val

    oggm_fsm_path                        = fsm_config.get('FSM-OGGM_path')
    sys.path.append(oggm_fsm_path)
    from FSM_oggm_MB import FactorialSnowpackModel, process_wfde5_data

    # write/reset the FSM namelist
    FactorialSnowpackModel.create_nml(reset=reset)

    # ----------------------
    # 6) Climate & I/O paths
    # ----------------------
    cfg.PATHS['climate_file']   = inp_config.get('climate_file')
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'
    catchment_path = inp_config.get('catchment_path')
    wgms_path = inp_config.get('wgms_path')
    parameter_sample_file = inp_config.get('parameter_sample_file',fallback=None)
    overwrite_sample_file = inp_config.getboolean('overwrite_sample',fallback=False)

    # ----------------------
    # 7) OGGM run setup
    # ----------------------
    y0 = inp_config.getint('y0')
    y1 = inp_config.getint('y1')
    num_sample = inp_config.getint('num_samples',fallback=100)
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
    # 9) Parameter Sample Generation
    # ----------------------

    sens_params
    fsm_sp = ProblemSpec({
        "names": sens_params,
        "groups": None,
        "bounds": sens_bounds,
        "outputs": ["CF"],
    })


    if (overwrite_sample_file) or (parameter_sample_file is None):
        fsm_sp.sample_sobol(num_sample, calc_second_order=True)
        sample_arr = fsm_sp.samples
        results_arr = -1*np.ones((sample_arr.shape[0],3))
        sample_results_arr = np.hstack((sample_arr,results_arr))
        dfsample = pd.DataFrame(data=sample_results_arr, columns=sens_params+['cost_profile','cost_mb','cost_wmb'])
        if parameter_sample_file is None:
            parameter_sample_file = os.getcwd() + '/analysis.csv'
        dfsample.to_csv(parameter_sample_file)
    else:
        dfsample = pd.read_csv(parameter_sample_file,index_col=0)
        arr = dfsample.to_numpy()
        sample_arr = arr[:,:-3]
        results_arr = arr[:,-3:]
        fsm_sp.set_samples(sample_arr)
        print ('restarting from file ' + parameter_sample_file)

    if len(np.where(results_arr[:,0]==-1)[0])==0:
        print('sample complete, no new results needed')
        isample_start = sample_arr.shape[0]
    else:
        profile_start = np.where(results_arr[:,0]==-1)[0][0]
        mb_start = np.where(results_arr[:,1]==-1)[0][0]
        winter_mb_start = np.where(results_arr[:,2]==-1)[0][0]
        isample_start = min(profile_start, mb_start, winter_mb_start)
        print ('beginning at sample number ' + str(isample_start))

    # ----------------------
    # 10) Process wfde5 data
    # ----------------------
    
        workflow.execute_entity_task(process_wfde5_data, gdirs, y0=str(y0), y1=str(y1))
        print("DONE PROCESSING wfde5 data")

    # ----------------------
    # 11) Run/gather results
    # ----------------------

    gdir = gdirs[0]
    fls = gdir.read_pickle('inversion_flowlines')
    areas = fls[0].bin_area_m2
    elevs = fls[0].surface_h

    mb_model = FactorialSnowpackModel(gdir=gdir)
    wgms_dict = get_WGMS_data(wgms_path, years_cost, wgms_id)
    
    years_compute = np.array([years_cost[0]-1] + years_cost)


    for isample in range(isample_start,sample_arr.shape[0]): 
        
        mb_output = None

        for i, name in enumerate(sens_params):
            cfg.PARAMS[name] = sample_arr[isample, i]

        FactorialSnowpackModel.create_nml(reset=False)
            
        for i, year in enumerate(years_compute):    
            mb_year = mb_model.get_mb(heights=fls[0].surface_h, year=year, fls=fls, reset_state=True, monthly=True)
            if mb_output is None:
                mb_output = mb_year
            else:
                mb_output = np.vstack((mb_output,mb_year))

        profile_err, mb_err, wmb_err = get_cost(mb_output, years_compute, wgms_dict, areas, elevs)
        results_arr[isample,:] = [profile_err, mb_err, wmb_err]

        if (np.mod(isample,10)==0):
            print (str(isample) + ' samples done')
        if (np.mod(isample,100)==0):
            sample_results_arr = np.hstack((sample_arr,results_arr))
            pd.DataFrame(data=sample_results_arr, columns=sens_params+['cost_profile','cost_mb','cost_wmb']).to_csv(parameter_sample_file)


    sample_results_arr = np.hstack((sample_arr,results_arr))
    pd.DataFrame(data=sample_results_arr, columns=sens_params+['cost_profile','cost_mb','cost_wmb']).to_csv(parameter_sample_file)
    print("DONE WITH SAMPLE")

    # ----------------------
    # 12) Analyse 
    # ----------------------

    strs = ['profile cost','mb cost','winter mb cost']

    for i in range(3):

        fsm_sp.set_results(results_arr[:,i])
 
        fsm_sp.analyze_sobol()
        print('analysis for ' + strs[i] + ':')
        print(fsm_sp)

    print('analysis for cf sum')

    fsm_sp.set_results(results_arr.sum(axis=1))
    fsm_sp.analyze_sobol()
    print(fsm_sp)




 





if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_fsm_rofental.py <config.ini>")
        sys.exit(1)
    main(sys.argv[1])
