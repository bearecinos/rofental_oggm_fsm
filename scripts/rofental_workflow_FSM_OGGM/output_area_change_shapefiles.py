import sys
import os
import configparser
import numpy as np
import geopandas as gpd
from pyproj import CRS
import salem
from rasterio.transform import from_origin
from rasterio.features import shapes
from shapely.geometry import shape
from shapely import unary_union
from oggm import cfg, utils
from oggm import workflow

# Define helper functions
def ice_thickness_to_outline(ice_array, transform, crs, threshold=0.0):
    """
    Convert an ice thickness NumPy array to a polygon outline.

    Parameters:
    - ice_array: 2D NumPy array representing ice thickness values.
    - transform: Rasterio transform object (needed for georeferencing).
    - threshold: Minimum ice thickness value to consider as ice-covered.

    Returns:
    - GeoDataFrame with ice coverage polygons.
    """

    # Create a binary mask (1 = ice, 0 = no ice)
    mask = ice_array > threshold

    # Convert the mask into vector shapes
    shapes_gen = shapes(mask.astype(np.uint8), mask=mask, transform=transform)

    # Convert shapes to Shapely geometries
    polygons = [shape(geom) for geom, value in shapes_gen if value == 1]

    merged_polygon = unary_union(polygons)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=[merged_polygon], crs=crs)  # Adjust CRS if needed

    return gdf

## Define main function
def main(cfg_path):
    # 1) Read configuration file
    cp = configparser.ConfigParser()
    cp.read(cfg_path)
    gen = cp['General']
    oggm = cp['OGGM']
    fsm = cp['FSM_OGGM']
    inp = cp['InputData']
    outp = cp['Output']

    # 2) General settings
    working_dir = gen.get('working_dir')
    # We force reset=False here since this is pure post-processing
    reset = False

    # 3) Initialize OGGM
    cfg.initialize(logging_level='DEBUG')
    cfg.PATHS['working_dir'] = utils.mkdir(working_dir, reset=reset)
    print(f"Working directory: {cfg.PATHS['working_dir']}")
    print("Reset forced to False for post-processing")

    # 4) OGGM core params
    cfg.PARAMS['use_multiprocessing'] = oggm.getboolean('use_multiprocessing')
    cfg.PARAMS['mp_processes']        = oggm.getint('mp_processes')
    cfg.PARAMS['border']              = oggm.getint('border', fallback=80)

    # 5) FSM parameters (only those relevant to post-processing)
    cfg.PARAMS['FSM_save_runoff'] = fsm.getboolean('FSM_save_runoff')
    cfg.PARAMS['FSM_runoff_frequency'] = fsm.get('FSM_runoff_frequency')

    # 6) Standard OGGM flags
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_compression'] = True
    cfg.PARAMS['use_tar_shapefiles'] = True
    cfg.PATHS['rgi_version'] = '62'
    cfg.PARAMS['use_temp_bias_from_file'] = True
    cfg.PARAMS['compress_climate_netcdf'] = False
    cfg.PARAMS['store_model_geometry'] = True
    cfg.PARAMS['store_fl_diagnostics'] = True

    # 7) Define FSM_runoff basename
    _doc = ("A netcdf file containing dates and "
            "ice-based and snow-based runoff volume for each date interval")
    cfg.BASENAMES['FSM_runoff'] = ('FSM_runoff.nc', _doc)

    # 8) Load RGI regions & catchment polygon
    fr = utils.get_rgi_region_file(11, version='62', reset=False)
    gdf = gpd.read_file(fr)
 
    catchment_path = inp.get('catchment_path')
    rof_shp = gpd.read_file(catchment_path)
    rof_sel = gdf.clip(rof_shp)
    rof_sel = rof_sel.sort_values('Area', ascending=False)

    rgi_id = inp.get('glacier_rgi_id')
    if rgi_id in ('None', '', None):
        rgi_id = None

    if rgi_id:
        selection = rof_sel[rof_sel.RGIId == rgi_id]
    else:
        selection = rof_sel

    # Here we never need to reset the working directory since we start
    # from a working dir where simulations have been run before
    gdirs = workflow.init_glacier_directories(selection)

    simulation_name = outp.get('simulation_name')
    # Let's make a directory for CEH data and file formats
    output_dir = os.path.join(cfg.PATHS['working_dir'],
                              'area_evolution/'+simulation_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        for gdir in gdirs:
            thick_path = gdir.get_filepath('gridded_simulation', filesuffix=simulation_name)
            if os.path.exists(thick_path):
                df_oggm = salem.open_xr_dataset(thick_path)
                grid_oggm = df_oggm.salem.grid
                # Convert to a CRS object
                custom_crs = CRS.from_proj4(df_oggm.pyproj_srs)
                pixel_size = grid_oggm.dx  # Example pixel size in meters

                # Define the transform (assuming upper-left at (0,0))
                transform = from_origin(grid_oggm.x0, grid_oggm.y0, pixel_size, pixel_size)

                years = df_oggm.time.values

                for year in years:
                    thick_ext = df_oggm.sel(time=int(year))
                    thick_array = thick_ext.simulated_thickness.data
                    shape_outline = ice_thickness_to_outline(thick_array,
                                                             transform,
                                                             custom_crs)
                    rgi_id = gdir.rgi_id
                    path_to_shapefile_dir = os.path.join(output_dir, rgi_id)
                    if not os.path.exists(path_to_shapefile_dir):
                        os.makedirs(path_to_shapefile_dir)
                    shape_outline.to_file(f"{path_to_shapefile_dir}/{int(year)}", driver='ESRI Shapefile')
            else:
                continue

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python output_area_change_shapefiles.py <config.ini>")
        sys.exit(1)
    main(sys.argv[1]) 
