import logging
import numpy as np
import pickle
import geopandas as gpd
import pandas as pd
import pyproj
import salem
from salem import wgs84
from scipy.interpolate import griddata
from oggm.exceptions import InvalidWorkflowError

# Module logger
log = logging.getLogger(__name__)

def interp_with_griddata_and_pkl(h_year, x_ori, y_ori, x_to_int, y_to_int,
                                year, file_name, return_data_set=False):
    """
    Wrapper for scipy.interpolate.griddata
    """
    # Let's fill all nans with zero before interpolation
    h_year = h_year.fillna(0)
    h_new = griddata((np.ravel(x_ori), np.ravel(y_ori)),
                     np.ravel(h_year), (x_to_int, y_to_int),
                     method='linear')
    print('another year is finish!')

    data = {'simulated_thickness': h_new, 'year': year}

    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

    if return_data_set:
        return h_new


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
