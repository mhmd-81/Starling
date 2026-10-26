import numpy as np


def time_split(values, fit_frac=0.8):
    """Split a 1-D array in time order: first part to fit, last part to validate.
    """
    
    values = np.asarray(values)
    if not 0 < fit_frac < 1:
        raise ValueError("fit_frac must be strictly between 0 and 1")
    cut = int(len(values) * fit_frac)
    return values[:cut], values[cut:]