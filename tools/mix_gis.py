import logging
import numpy as np
import netCDF4
from scipy.interpolate import griddata

# Module logger
log = logging.getLogger(__name__)

def interp_with_griddata(h_year, x_ori, y_ori, x_to_int, y_to_int):
    """
    Wrapper for scipy.interpolate.griddata
    """
    # Let's fill all nans with zero before interpolation
    h_year = h_year.fillna(0)
    h_new = griddata((np.ravel(x_ori), np.ravel(y_ori)),
                     np.ravel(h_year), (x_to_int, y_to_int),
                     method='linear')

    return h_new

