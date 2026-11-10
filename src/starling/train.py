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