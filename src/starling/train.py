import argparse
from datetime import datetime
import pandas as pd
from starling.pipeline import run_channel
from starling.tracking import log_run


def labled_channels(data_dir='data'):
    labels = pd.read_csv(f'{data_dir}/labeled_anomalies.csv')
    return sorted(labels["chan_id"].unique())