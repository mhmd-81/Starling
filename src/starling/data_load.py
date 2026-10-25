# requirements
import os
import ast
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv


load_dotenv() 

data_dir = Path(os.getenv("BASE_DATA"))


def load_channel(chan_id, data_dir=None):
    """Loads the data(train, test, meta data) into the workflow"""
    train_data=np.load(data_dir / "train" / f"{chan_id}.npy")[:, 0]
    test_data=np.load(data_dir/ "test" f"{chan_id}.npy")[:,0]
    labels = pd.read_csv(f"{data_dir}/labeled_anomalies.csv")
    rows = labels[labels["chan_id"] == chan_id]
    intervals = []
    for s in rows["anomaly_sequences"]:
        intervals.extend(ast.literal_eval(s))
    return train_data, test_data, intervals


