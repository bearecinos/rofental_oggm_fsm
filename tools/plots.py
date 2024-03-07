" Mix code to plot things"
from collections import defaultdict
import xarray as xr
import os
import matplotlib.pyplot as plt
import matplotlib
import logging
from oggm import cfg, tasks, utils
from oggm import entity_task

# Module logger
log = logging.getLogger(__name__)

def get_dmdtda(ds, gdir, ref_period):
    """
    Get change of mass per time per area? as kg m-2 yr-1

    ds: xarray Dataset with change volume
    gdir: Glacier directory which we are gathering the data
    :returns
    Modeled geodetic mass balances for comparison with Hugonnet 2021
    """

    yr0_ref_mb, yr1_ref_mb = ref_period.split('_')
    yr0_ref_mb = int(yr0_ref_mb.split('-')[0])
    yr1_ref_mb = int(yr1_ref_mb.split('-')[0])

    subs = ds.volume_m3.loc[yr1_ref_mb].values - ds.volume_m3.loc[yr0_ref_mb].values
    model_geo_mb = (subs / gdir.rgi_area_m2 /
                    (yr1_ref_mb - yr0_ref_mb) * cfg.PARAMS['ice_density'])

    return model_geo_mb

@entity_task(log)
def plot_different_spinup_results(gdir, save_analysis_text=True):
    """ plot different spinup results

    gdirs: list of Glacier directories to process

    :returns
    Saves a plot in the glacier directory
    """

    id = gdir.rgi_id

    # Define dictionaries to store data
    d_hist_fix_geom = defaultdict(list)
    d_sp_dynamic_area = defaultdict(list)
    d_sp_dynamic_volume = defaultdict(list)
    d_sp_dynamic_melt_f = defaultdict(list)

    with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_hist_fixed_geom')) as ds:
        ds_hist = ds.load()
        d_hist_fix_geom[id] = ds_hist

    with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_spinup_dynamic_area')) as ds:
        ds_dynamic_spinup_area = ds.load()
        d_sp_dynamic_area[id] = ds_dynamic_spinup_area

    with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_spinup_dynamic_volume')) as ds:
        ds_dynamic_spinup_volume = ds.load()
        d_sp_dynamic_volume[id] = ds_dynamic_spinup_volume

    with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_dynamic_melt_f')) as ds:
        ds_sp_dynamic_melt_f = ds.load()
        d_sp_dynamic_melt_f[id] = ds_sp_dynamic_melt_f

    volume_reference = tasks.get_inversion_volume(gdir)
    area_reference = gdir.rgi_area_m2

    ref_period = cfg.PARAMS['geodetic_mb_period']
    # get the data from Hugonnet 2021
    df_ref_dmdtda = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
    # only select the desired period
    df_ref_dmdtda = df_ref_dmdtda.loc[df_ref_dmdtda['period'] == ref_period]

    # get the reference dmdtda and convert into kg m-2 yr-1
    dmdtda_reference = df_ref_dmdtda['dmdtda'].values[0] * 1000
    dmdtda_reference_error = df_ref_dmdtda['err_dmdtda'].values[0] * 1000

    # here we save the original melt_f for later
    melt_f_original = gdir.read_json('mb_calib')['melt_f']

    # Now make a plot for comparision
    y0 = gdir.rgi_date + 1

    # Make a plot per glacier and store this in each glacier dir
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    d_hist_fix_geom[id].volume_m3.plot(ax=ax1, label='Fixed geometry spinup')
    d_sp_dynamic_melt_f[id].volume_m3.plot(ax=ax1, label='Dynamical melt_f calibration',
                                           linewidth=5.0, alpha=0.5)
    d_sp_dynamic_area[id].volume_m3.plot(ax=ax1, label='Dynamical spinup match area')
    d_sp_dynamic_volume[id].volume_m3.plot(ax=ax1, label='Dynamical spinup match volume')
    ax1.set_title('Volume')
    ax1.scatter(y0, volume_reference, c='C3', label='Reference values')
    ax1.legend()

    d_hist_fix_geom[id].area_m2.plot(ax=ax2)
    d_sp_dynamic_melt_f[id].area_m2.plot(ax=ax2, linewidth=5.0, alpha=0.5)
    d_sp_dynamic_area[id].area_m2.plot(ax=ax2)
    d_sp_dynamic_volume[id].area_m2.plot(ax=ax2)
    ax2.set_title('Area')
    ax2.scatter(y0, area_reference, c='C3')

    d_hist_fix_geom[id].length_m.plot(ax=ax3)
    d_sp_dynamic_melt_f[id].length_m.plot(ax=ax3, linewidth=5.0, alpha=0.5)
    d_sp_dynamic_area[id].length_m.plot(ax=ax3)
    d_sp_dynamic_volume[id].length_m.plot(ax=ax3)
    ax3.set_title('Length')
    ax3.scatter(y0, d_hist_fix_geom[id].sel(time=y0).length_m, c='C3')

    plt.tight_layout()
    plt.savefig(os.path.join(gdir.dir,
                             'spinup_comparison' + id + '.png'))

    if save_analysis_text:
        # Print and save modeled geodetic mass balances for comparison
        file_path = os.path.join(gdir.dir, 'spinup_comparison_mb_' + id + '.txt')
        f = open(file_path, 'w')

        f.write(f'Reference dmdtda 2000 to 2020 (Hugonnet 2021): '
                f'{dmdtda_reference:.2f} +/- {dmdtda_reference_error:6.2f} kg m-2 yr-1 \n')
        f.write(f'Fixed geometry spinup dmdtda 2000 to 2020: '
                f'{get_dmdtda(d_hist_fix_geom[id], gdir, ref_period):.2f} kg m-2 yr-1 \n')

        f.write(f'Dynamical spinup match area dmdtda 2000 to 2020: '
                f'{get_dmdtda(d_sp_dynamic_area[id], gdir, ref_period):.2f} kg m-2 yr-1 \n')

        f.write(f'Dynamical spinup match volume dmdtda 2000 to 2020: '
                f'{get_dmdtda(d_sp_dynamic_volume[id], gdir, ref_period):.2f} kg m-2 yr-1 \n')

        f.write(f'Dynamical melt_f calibration dmdtda 2000 to 2020: '
                f'{get_dmdtda(d_sp_dynamic_melt_f[id], gdir, ref_period):.2f} kg m-2 yr-1 \n')

        f.write(f'Original melt_f: '
                f'{melt_f_original:.1f} kg m-2 day-1 °C-1 \n')
        f.write(f"Dynamic melt_f: "
                f"{gdir.read_json('mb_calib')['melt_f']:.1f} kg m-2 day-1 °C-1 \n")

        f.close()
