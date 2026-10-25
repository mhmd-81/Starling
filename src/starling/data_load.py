# requirements
import os
import ast
import pandas
import numpy as np
from pathlib import Path
from dotenv import load_dotenv


load_dotenv() 


def load_channel(chan_id, data_dir=None):
    """Loads the data into the workflow"""
    data_dir = Path(data_dir or os.getenv("BASE_DATA"))
    return np.load(data_dir / "train" / f"{chan_id}.npy")[:, 0]

