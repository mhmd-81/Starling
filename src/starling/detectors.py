import numpy as np
import pandas as pd


def trailing_deviation(signal, window=101):
    """Absolute deviation of each point from the median of the PAST window."""
    past_median = (
        pd.Series(signal)
        .shift(1)
        .rolling(window=window, min_periods=window)
        .median()
    )
    return np.abs(signal - past_median.to_numpy())