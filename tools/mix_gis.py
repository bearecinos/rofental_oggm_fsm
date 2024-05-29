import logging
import numpy as np
import pickle
from scipy.interpolate import griddata

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
