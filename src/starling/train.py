import argparse
from datetime import datetime
import pandas as pd
from starling.pipeline import run_channel
from starling.tracking import log_run


def labled_channels(data_dir='data'):
    labels = pd.read_csv(f'{data_dir}/labeled_anomalies.csv')
    return sorted(labels["chan_id"].unique())

def main():
    """accepting different kind of args for model run"""
    parser = argparse.ArgumentParser(description='running baselines on channel ids')
    parser.add_argument("--methods", nargs='+', default=['rolling_median', 'linear'])
    parser.add_argument("--channels", nargs="*", default=None, help='useing all channels as default')
    parser.add_argument("--q", type=float, default=0.99)
    parser.add_argument("--window", type=int, default=101)
    parser.add_argument("--lag", type=int, default=32)
    parser.add_argument("--fit-frac", type=int, default=0.8)
    parser.add_argument("--data-dir", default='data')
    args = parser.parse_args()

    channels = args.channels or labled_channels(args.data_dir)
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"batch {batch_id}: {len(channels)} channels x {len(args.methods)} methods")

    failed = {}
    for method in args.methods:
        for chan in channels:
            try:
                result = run_channel(chan, method=method, window=args.window, lag=args.lag,
                                     q=args.q, fit_frac=args.fit_frac, data_dir=args.data_dir)
                log_run(result, tags={"batch_id": batch_id})
                print(f"  {method:15s} {chan:5s} f1={result['point']['f1']:.3f}  auc={result['auc']:.3f}")
            
            except Exception as e:
                failed[(method, chan)] = str(e)[:80]
                print(f"  {method:15s} {chan:5s} FAILED: {failed[(method, chan)]}")

    print(f"done. batch_id={batch_id} | failed: {len(failed)}")


if __name__ == "__main__":
    main()