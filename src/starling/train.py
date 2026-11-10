import argparse
from datetime import datetime
import pandas as pd
from starling.pipeline import run_channel
from starling.tracking import log_run


def labled_channels(data_dir='data'):
    labels = pd.read_csv(f'{data_dir}/labeled_anomalies.csv')
    return sorted(labels["chan_id"].unique())

def main():
    parser = argparse.ArgumentParser(description='running baselines on channel ids')
    parser.add_argument("--methods", nargs='+', default=['roaling_median', 'linear'])
    parser.add_argument("--channels", nargs="*", default=None, help='useing all channels as default')
    parser.add_argument("--q", type=float, default=0.99)
    parser.add_argument("--window", type=int, default=101)
    parser.add_argument("--lag", type=int, default=32)
    parser.add_argument("--fit-frac", type=int, default=0.8)
    parser.add_argument("--data-dir", default='data')
    args = parser.parse_args()