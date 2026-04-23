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
from IPython import embed

used_keys={}

def cfg_get(config_section, option, fallback=None):
    val = config_section.get(option,fallback=fallback)
    used_keys.setdefault(config_section._name, {})[option] = str(val)
    return val

def cfg_getint(config_section, option, fallback=None):
    val = config_section.getint(option,fallback=fallback)
    used_keys.setdefault(config_section._name, {})[option] = str(val)
    return val

def cfg_getboolean(config_section, option, fallback=None):
    val = config_section.getboolean(option,fallback=fallback)
    used_keys.setdefault(config_section._name, {})[option] = str(val)
    return val

from oggm.cfg import SEC_IN_YEAR
# rho is defined here so it can be used in the cost function 
# but it is initialised in main with cfg.PARAMS
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

        # there might be empty uncertainties. Since may unc's are 0.1 mwe, 
        # we will use this so we don't discount years with data
        bal_unc[np.isnan(bal_unc)] = 0.1
        winter_bal_unc[np.isnan(winter_bal_unc)] = 0.1

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
        mb_profile_areas = None
        

        unc_nonnan=0
        for i in range(len(years)):
            dfyr = df[df.year==years[i]]
            dfyr = dfyr[dfyr['lower_elevation'].isin(lower_band)]
            assert len(dfyr)==len(lower_band), "inconsistent bands"
            if mb_profile_areas is None:
                mb_profile_areas = dfyr["area"].values
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
        return_dict['mb_profile_areas'] = mb_profile_areas

    return return_dict

def get_cost(mb_output, mb_output_years, wgms_data, areas, elevs, \
	profile_cost=True, mb_cost=True, winter_mb_cost=True, doMean=False, make_plots=False):

    """
    A function to define misfit cost function

    Parameters
    ----------
    mb_output:		an (Nx12)xM array where N=len(mb_output_years), M=no. of bands
    				each row is mean mb per band (in m/s) for a given month
    mb_output_years:
    				the years FSM is run
    wgms_data:		the dict from get_WGMS_data
    areas, elevs: 	arrays from model flowline
    profile_cost, mb_cost_winter_mb_cost: 
    				flags for cf terms
    doMean: 		average mb series before finding misfit
    make_plots:         print comparison plots

    """

    profile_misfit = None
    mb_misfit = None
    winter_mb_misfit = None

    if make_plots:
        import matplotlib.pyplot as plt
        f, axs = plt.subplots(1, 3, figsize=(12, 4))

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
    	# RMSE of time-mean mb profile
        profile_misfit = np.sqrt(np.nansum( wgms_data['mb_profile_areas']*(mb_interp-wgms_data['mb_profile_mwe'])**2 / wgms_data['mb_profile_mwe_unc']**2 ) / np.sum(wgms_data['mb_profile_areas']))

        if make_plots:
            axs[0].plot(mb_interp,.5*(wgms_data['mb_profile_lower']+wgms_data['mb_profile_upper']),label='FSM')
            axs[0].plot(wgms_data['mb_profile_mwe'],.5*(wgms_data['mb_profile_lower']+wgms_data['mb_profile_upper']),label='WGMS')
            axs[0].legend()

    if mb_cost:

        mb_annual = (mb_output[inds,:]).reshape(-1,12,mb_output.shape[1]).mean(axis=1) # result is in m/s
        mb_series = np.matmul(mb_annual, areas) # result is m3/s
        mb_series = mb_series * rho / 1000. * SEC_IN_YEAR / areas.sum() # result in mwe/yr
        if not doMean:
			# RMSE of mass balance time series
            mb_misfit = np.sqrt( np.nanmean( (mb_series - wgms_data['mb_annual_mwe'])**2 / wgms_data['mb_annual_mwe_unc']**2 ) )
        else:
            mb_misfit =  np.abs(  mb_series.mean() - np.mean(wgms_data['mb_annual_mwe']) ) / wgms_data['mb_annual_mwe_unc'][0]

        if make_plots:
            axs[1].plot(wgms_data['mb_years'],mb_series,label='FSM')
            col1 = axs[1].get_lines()[-1].get_color()
            axs[1].plot(wgms_data['mb_years'],wgms_data['mb_annual_mwe'],label='WGMS')
            col2 = axs[1].get_lines()[-1].get_color()
            axs[1].plot(wgms_data['mb_years'],np.nanmean(mb_series)*np.ones(len(wgms_data['mb_years'])),color=col1,linestyle='--')
            axs[1].plot(wgms_data['mb_years'],np.nanmean(wgms_data['mb_annual_mwe'])*np.ones(len(wgms_data['mb_years'])),color=col2,linestyle='--')
            axs[1].legend()

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
			# RMSE of winter mass balance time series
            winter_mb_misfit = np.sqrt( np.nanmean( (mb_series - wgms_data['mb_winter_mwe'])**2 / wgms_data['mb_winter_mwe_unc']**2 ) )
        else:
            nonans=np.logical_not(np.isnan(wgms_data['mb_winter_mwe']))
            winter_mb_misfit =  np.abs(  mb_series[nonans].mean() - np.mean(wgms_data['mb_winter_mwe'][nonans])) / wgms_data['mb_winter_mwe_unc'][0]

        if make_plots:
            axs[2].plot(wgms_data['mb_years'],mb_series,label='FSM')
            col1 = axs[2].get_lines()[-1].get_color()
            axs[2].plot(wgms_data['mb_years'],wgms_data['mb_winter_mwe'],label='WGMS')
            col2 = axs[2].get_lines()[-1].get_color()
            axs[2].plot(wgms_data['mb_years'],np.nanmean(mb_series)*np.ones(len(wgms_data['mb_years'])),color=col1,linestyle='--')
            axs[2].plot(wgms_data['mb_years'],np.nanmean(wgms_data['mb_winter_mwe'])*np.ones(len(wgms_data['mb_years'])),color=col2,linestyle='--')

            axs[2].legend()            
	
    return profile_misfit, mb_misfit, winter_mb_misfit

def main(cfg_path):
    # ----------------------
    # 1) Read configuration file
    # ----------------------
    cp = configparser.ConfigParser()
	# option needed for case sensitivity
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
    working_dir = cfg_get(gen_config,'working_dir')          # string, may be blank
    reset       = cfg_getboolean(gen_config,'reset')         # bool

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
    cfg.PARAMS['use_multiprocessing'] = cfg_getboolean(oggm_config,'use_multiprocessing')
    cfg.PARAMS['mp_processes']        = cfg_getint(oggm_config,'mp_processes')
    cfg.PARAMS['border']              = cfg_getint(oggm_config,'border', fallback=80)

    # ----------------------
    # 5) FSM parameters
    # ----------------------
    cfg.PARAMS['FSM_save_runoff']        = cfg_getboolean(fsm_config,'FSM_save_runoff',fallback=True)
    cfg.PARAMS['FSM_runoff_frequency']   = cfg_get(fsm_config,'FSM_runoff_frequency',fallback='D')
    cfg.PARAMS['FSM_spinup']             = cfg_getboolean(fsm_config,'FSM_spinup',fallback=True)
    cfg.PARAMS['FSM_interpolate_bnds']   = cfg_getboolean(fsm_config,'FSM_interpolate_bnds',fallback=False) # note: nbnds can only be set if interpolate_bnds is True
    cfg.PARAMS['FSM_Nbnds']              = cfg_getint(fsm_config,'FSM_Nbnds',fallback=None)


    # important: parameters for namelist must start with "FSM_param_"
    cpdict = dict(fsm_config)
    sens_params = []
    sens_bounds = []

	# the loop below is to be able to specify FSM parameters without explicitly codeing for them
	# if they begin with "FSM_param_" they will be included in the FSM namelist
    for key in cpdict.keys():
        if (key[:10] == 'FSM_param_'):

            valstr = cpdict[key]
            val = json.loads(valstr)
            # ensure these appear in default ini file
            used_keys.setdefault(fsm_config._name, {})[key] = 'None'

			# if the value of the param is a list (of length 2)
			# then this parameter is used in the sensitivity analysis below
			# and the list gives the range
            if isinstance(val,list):
                cfg.PARAMS[key] = val[0]
                sens_params.append(key)
                sens_bounds.append(val)
            else:
                cfg.PARAMS[key] = val

    oggm_fsm_path                        = cfg_get(fsm_config,'FSM-OGGM_path')
    sys.path.append(oggm_fsm_path)
    from FSM_oggm_MB import FactorialSnowpackModel, process_wfde5_data

    # write/reset the FSM namelist
    FactorialSnowpackModel.create_nml(reset=reset)

    # ----------------------
    # 6) Climate & I/O paths
    # ----------------------
    cfg.PATHS['climate_file']   = cfg_get(inp_config,'climate_file')
    cfg.PARAMS['baseline_climate'] = 'CUSTOM'
    
    # a path to the wgms FoG data files. should be folder containing csv files
    wgms_path = cfg_get(inp_config,'wgms_path') 
    
    # a path to the wgms to rgi mapping available from the oggm-sample-data repo
    wgms_to_rgi_path = cfg_get(inp_config,'wgms_to_rgi_path',fallback=None)
    # base name of the csv file that contains param samples and costs
    parameter_sample_file_base = cfg_get(inp_config,'parameter_sample_file_base',fallback=None)
    # Only set to true if there are no values already saved in the target file
    overwrite_sample_file = cfg_getboolean(inp_config,'overwrite_sample_file',fallback=False)

    # ----------------------
    # 7) OGGM run setup
    # ----------------------
    y0 = cfg_getint(inp_config,'y0')
    y1 = cfg_getint(inp_config,'y1')
    num_sample = cfg_getint(inp_config,'num_samples',fallback=100)

    # for glacier wide MB and Winter MB, error is based on difference in mean
    doMean = cfg_getboolean(inp_config,'calibrate_to_mean',fallback=False)

    # this is a facility to, rather than run a large sample, plot the results
    # from a single sample. the input is the index of the sample array to run.
    # it is most useful if one has already run a large sample.
    # SPECIAL VALUES:
    #    -1: use the row corresponding to minimum total cost
    #    -2: use the values in the .ini file
    one_off_sample = cfg_getint(inp_config,'one_off_sample',fallback=None)
    years_cost = json.loads(cfg_get(inp_config,'years_cost'))
    simulation_name = cfg_get(outp_config,'simulation_name')

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


    # Grab the raw string (or None if the key is missing)
    wgms_id = cfg_getint(inp_config,'glacier_wgms_id', fallback=0)

    configOut = configparser.ConfigParser()
    configOut.read_dict(used_keys)
    with open('used_params.ini', 'w') as f:
        configOut.write(f)

    # HEF: 491 summer bal non-nan 2013-2025
    # KEF: 507 no summer bal
    # VER: 489 summer bal 1966 on!

    assert wgms_to_rgi_path is not None, "specify wgms_to_rgi_path"
    dfwr = pd.read_csv(wgms_to_rgi_path)
    
    # Now select
    if wgms_id==0:
        dfeur = dfwr[dfwr.RGI60_ID.str[6:8]=='11']
        selection=dfeur.RGI60_ID.tolist()
    else:
        selection = dfwr[dfwr.WGMS_ID == wgms_id].RGI60_ID.tolist()
        parameter_sample_file = parameter_sample_file_base + '_' + selection[0] + '.csv'

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

    # ----------------------
    # 10) Process wfde5 data
    # ----------------------

	# we only need to process w5de5 data if there are more samples to calculate
    #    if (isample_start < sample_arr.shape[0]) or one_off_sample is not None:
    cfg.PARAMS['use_multiprocessing'] = False
    for gdir in gdirs:
        if not os.path.isfile(gdir.dir + '/climate_historical_fsm.nc'):
            process_wfde5_data(gdir, y0=str(y0), y1=str(y1))

    cfg.PARAMS['use_multiprocessing'] = cfg_getboolean(oggm_config,'use_multiprocessing')
    print("DONE PROCESSING wfde5 data")

    if wgms_id == 0: # no wgms id specified -- no sampling to do
        print('No sampling to do, exiting..')        
        sys.exit(0)

    # ----------------------
    # 9) Parameter Sample Generation
    # ----------------------

	# ProblemSpec is an SALib type that sets up an analysis
    sens_params
    fsm_sp = ProblemSpec({
        "names": sens_params,
        "groups": None,
        "bounds": sens_bounds,
        "outputs": ["CF"],
    })

	# if the param sample filename not given or overwrite=true:
	# -	We need to define the parameter sample, using sample_sobol
	# - parameter sample saved to analysis.cv
	# - cost funciton columns set to -1 (not calculated)
	# otherwise
	# - we read the param sample from the file specified

	
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
        print ('restarting from file ' + parameter_sample_file )

        # if there is a one-off sample to be run, assert that the one_off_sample
        # is not larger than the sample array

    if one_off_sample is not None:
       assert one_off_sample <= sample_arr.shape[0], "one_off_sample value too large"

	# Here we determine if we need to find values for the entire sample, or
	#   if we can skip some. We find the first row that has not been processed,
	#   i.e. with -1 in it. This is where we start the loop below.

    if one_off_sample is not None:
        print('one off sample, will not generate new sample results')
        isample_start = sample_arr.shape[0]
    elif len(np.where(results_arr[:,0]==-1)[0])==0:
        print('sample complete, no new results needed')
        isample_start = sample_arr.shape[0]
    else:
        profile_start = np.where(results_arr[:,0]==-1)[0][0]
        mb_start = np.where(results_arr[:,1]==-1)[0][0]
        winter_mb_start = np.where(results_arr[:,2]==-1)[0][0]
        isample_start = min(profile_start, mb_start, winter_mb_start)
        print ('beginning at sample number ' + str(isample_start))


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

    # do one-off and then return
    if one_off_sample is not None:

        if one_off_sample == -1:
            sum_results = np.nansum(results_arr,1)
            assert np.min(results_arr[:,0]) != -1, "sampling unfinished"
            one_off_sample = np.argmin(sum_results)

        if one_off_sample != -2:
            for i, name in enumerate(sens_params):
                cfg.PARAMS[name] = sample_arr[one_off_sample, i]

        FactorialSnowpackModel.create_nml(reset=False)

        mb_output = None

        # loop to run FSM for each year, and stack results
        for i, year in enumerate(years_compute):
            mb_year = mb_model.get_mb(heights=fls[0].surface_h, year=year, fls=fls, reset_state=True, monthly=True)
            if mb_output is None:
                mb_output = mb_year
            else:
                mb_output = np.vstack((mb_output,mb_year))

                # evaluation cost functions for row and store them
        profile_err, mb_err, wmb_err = get_cost(mb_output, years_compute, wgms_dict, areas, elevs, make_plots=True, doMean=doMean)

        import matplotlib.pyplot as plt
        plt.savefig('best_fit_' + str(wgms_id) + '.png')
        plt.close('all')

        return

	# main loop for CF evaluation
    for isample in range(isample_start,sample_arr.shape[0]): 
        
        mb_output = None

        for i, name in enumerate(sens_params):
            cfg.PARAMS[name] = sample_arr[isample, i]

        FactorialSnowpackModel.create_nml(reset=False)

		# loop to run FSM for each year, and stack results
        for i, year in enumerate(years_compute):    
            mb_year = mb_model.get_mb(heights=fls[0].surface_h, year=year, fls=fls, reset_state=True, monthly=True)
            if mb_output is None:
                mb_output = mb_year
            else:
                mb_output = np.vstack((mb_output,mb_year))

		# evaluation cost functions for row and store them
        profile_err, mb_err, wmb_err = get_cost(mb_output, years_compute, wgms_dict, areas, elevs, doMean=doMean)
        results_arr[isample,:] = [profile_err, mb_err, wmb_err]

		# notify every 10 steps, save progress to file every 100 steps
        if (np.mod(isample,10)==0):
            print (str(isample) + ' samples done')
        if (np.mod(isample,100)==0):
            sample_results_arr = np.hstack((sample_arr,results_arr))
            pd.DataFrame(data=sample_results_arr, columns=sens_params+['cost_profile','cost_mb','cost_wmb']).to_csv(parameter_sample_file)

	
    sample_results_arr = np.hstack((sample_arr,results_arr))
    pd.DataFrame(data=sample_results_arr, columns=sens_params+['cost_profile','cost_mb','cost_wmb']).to_csv(parameter_sample_file)
    print("DONE WITH SAMPLE")

    # -----------------------------------------
    # 12) Sobol Analysis for each cost function
    # -----------------------------------------

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



