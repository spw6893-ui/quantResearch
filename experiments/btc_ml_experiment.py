"""BTC ML Experiment: train LightGBM/XGBoost + Transformer-LSTM on BTC data.
Reuses the existing model pipeline from the main project.
Usage: python experiments/btc_ml_experiment.py [--freq daily|1h|30min|15min|5min] [--horizons 1,7,14]
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.btc_data import (
    load_btc,
    ALL_FREQS,
    triple_barrier_label,
    fetch_trades_ccxt,
    aggregate_trades_to_bars,
    add_microstructure_features,
)
from experiments.btc_signal_scan import add_ta_indicators
from config.settings import LGBM_CONFIG, XGB_CONFIG, MAX_FEATURES, RANDOM_SEED
from models.lgbm_model import LGBMModel
from models.xgb_model import XGBModel
from models.trainer import ModelTrainer
from utils.helpers import set_seed

set_seed(RANDOM_SEED)


def build_features(df, horizon, label_mode='binary', pt_sl=(1.0, 1.0)):
    """Build features and labels for a given horizon."""
    df = add_ta_indicators(df.copy())
    if label_mode == 'triple_barrier':
        tb = triple_barrier_label(df, horizon, pt_sl=pt_sl)
        # Map to binary: +1 -> 1 (up), else -> 0
        df['label'] = (tb == 1).astype(int)
        df['tb_raw'] = tb
    else:
        df['label'] = (df['close'].shift(-horizon) / df['close'] - 1 > 0).astype(int)
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1

    exclude = {'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount',
               'label', 'future_return', 'tb_raw'}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    df = df.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    return df, feature_cols


def create_sequences(df, feature_cols, seq_length=60):
    """Create 3D sequences for deep learning models."""
    from sklearn.preprocessing import RobustScaler

    data = df[feature_cols].values
    labels = df['label'].values

    # Scale (fit on first 70%)
    train_end = int(len(data) * 0.7)
    scaler = RobustScaler()
    scaler.fit(data[:train_end])
    data = scaler.transform(data)
    data = np.clip(data, -5, 5)

    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y)


def run_experiment(df, horizon, model_types, seq_length=60, label_mode='binary', pt_sl=(1.0, 1.0)):
    """Run single horizon experiment with multiple models."""
    df_feat, feature_cols = build_features(df, horizon, label_mode, pt_sl)
    if len(df_feat) < 500:
        print(f"  Insufficient data ({len(df_feat)} rows), skipping")
        return None

    print(f"  Features: {len(feature_cols)}, Samples: {len(df_feat)}")

    # Create sequences
    X, y = create_sequences(df_feat, feature_cols, seq_length)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Label dist: up={y_test.mean()*100:.1f}%")

    results = {'horizon': horizon, 'n_features': len(feature_cols),
               'n_train': len(X_train), 'n_test': len(X_test),
               'test_up_ratio': y_test.mean()}

    trainer = ModelTrainer()

    for mt in model_types:
        t0 = time.time()
        try:
            model, metrics = trainer.train_model(mt, X_train, y_train, X_val, y_val)

            if mt in ('lgbm', 'xgboost'):
                test_probs = model.predict_proba(X_test)
            else:
                import torch
                model.eval()
                test_probs_list = []
                batch_sz = 512
                with torch.no_grad():
                    for bi in range(0, len(X_test), batch_sz):
                        batch = torch.FloatTensor(X_test[bi:bi+batch_sz]).to(trainer.device)
                        probs = torch.sigmoid(model(batch)).cpu().numpy()
                        test_probs_list.append(probs)
                        del batch
                        torch.cuda.empty_cache()
                test_probs = np.concatenate(test_probs_list)

            test_auc = roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else 0.5
            test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))
            elapsed = time.time() - t0

            results[f'{mt}_val_auc'] = metrics.get('val_auc', 0)
            results[f'{mt}_test_auc'] = test_auc
            results[f'{mt}_test_acc'] = test_acc
            results[f'{mt}_time'] = round(elapsed, 1)
            print(f"  {mt}: val_auc={metrics.get('val_auc', 0):.4f}, test_auc={test_auc:.4f}, time={elapsed:.1f}s")
        except Exception as e:
            print(f"  {mt} FAILED: {e}")
            results[f'{mt}_val_auc'] = 0
            results[f'{mt}_test_auc'] = 0
            results[f'{mt}_test_acc'] = 0
            results[f'{mt}_time'] = 0

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTC ML Experiment")
    parser.add_argument('--freq', default='daily', choices=ALL_FREQS)
    parser.add_argument('--horizons', default=None, help='Comma-separated horizons (e.g., 1,7,14)')
    parser.add_argument('--models', default='lgbm,xgboost',
                        help='Comma-separated model types')
    parser.add_argument('--seq-length', type=int, default=60)
    parser.add_argument('--label-mode', default='binary', choices=['binary', 'triple_barrier'],
                        help='Labeling method: binary or triple_barrier (AFML)')
    parser.add_argument('--pt-sl', default='1.0,1.0',
                        help='Profit-take,stop-loss multipliers for triple barrier')
    parser.add_argument('--micro', action='store_true',
                        help='Add microstructure features from ccxt trades (best effort)')
    parser.add_argument('--trades-days', type=int, default=30,
                        help='Trades lookback days ending at last bar time (default 30)')
    parser.add_argument('--vpin-window', type=int, default=50)
    parser.add_argument('--kyle-window', type=int, default=50)
    args = parser.parse_args()

    df = load_btc(args.freq)
    print(f'Data: {len(df)} bars, {df["datetime"].iloc[0]} ~ {df["datetime"].iloc[-1]}')

    if args.micro:
        end_dt = pd.to_datetime(df["datetime"].iloc[-1])
        start_dt = end_dt - pd.Timedelta(days=int(args.trades_days))
        trades = fetch_trades_ccxt(start=start_dt, end=end_dt, verbose=True)
        df = aggregate_trades_to_bars(df, trades)
        df = add_microstructure_features(df, vpin_window=int(args.vpin_window), kyle_window=int(args.kyle_window))
        print(f"Microstructure columns added. Example: ofi/vpin/kyle -> {[c for c in ['ofi', f'vpin_{args.vpin_window}', f'kyle_lambda_{args.kyle_window}'] if c in df.columns]}")

    if args.horizons:
        horizons = [int(x) for x in args.horizons.split(',')]
    elif args.freq == 'daily':
        horizons = [1, 3, 7, 14, 30]
    elif args.freq == '5min':
        horizons = [12, 48, 144, 288]  # 1h, 4h, 12h, 1d
    elif args.freq == '15min':
        horizons = [4, 16, 48, 96]  # 1h, 4h, 12h, 1d
    elif args.freq == '30min':
        horizons = [2, 8, 24, 48]  # 1h, 4h, 12h, 1d
    elif args.freq.startswith('vbar'):
        horizons = [1, 5, 10, 20, 50, 100]  # volume bars
    else:
        horizons = [1, 4, 12, 24, 48]

    model_types = args.models.split(',')

    print(f'\nHorizons: {horizons}')
    print(f'Models: {model_types}')
    print(f'Seq length: {args.seq_length}')

    pt_sl_vals = tuple(float(x) for x in args.pt_sl.split(','))
    if args.label_mode == 'triple_barrier':
        print(f'Label mode: Triple Barrier (pt={pt_sl_vals[0]}, sl={pt_sl_vals[1]})')

    all_results = []
    for h in horizons:
        print(f'\n{"="*70}')
        print(f'  Horizon: {h} bars')
        print(f'{"="*70}')
        result = run_experiment(df, h, model_types, args.seq_length, args.label_mode, pt_sl_vals)
        if result:
            all_results.append(result)

    if not all_results:
        print('\nNo results.')
        return

    # Summary
    df_res = pd.DataFrame(all_results)
    print('\n' + '=' * 90)
    print('RESULTS SUMMARY')
    print('=' * 90)

    cols = ['horizon', 'n_features', 'n_train', 'n_test', 'test_up_ratio']
    for mt in model_types:
        cols.extend([f'{mt}_test_auc', f'{mt}_val_auc', f'{mt}_time'])
    display_cols = [c for c in cols if c in df_res.columns]

    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df_res[display_cols].to_string(index=False))

    # Best horizon per model
    for mt in model_types:
        col = f'{mt}_test_auc'
        if col in df_res.columns and df_res[col].max() > 0:
            best = df_res.loc[df_res[col].idxmax()]
            print(f'\n{mt} best: horizon={int(best["horizon"])}, test_auc={best[col]:.4f}')

    # Save
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'experiments')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'btc_{args.freq}_ml_results.csv')
    df_res.to_csv(save_path, index=False)
    print(f'\nResults saved: {save_path}')


if __name__ == '__main__':
    main()
