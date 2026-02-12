"""BTC TA Signal Mining: discrete win rates + continuous AUC scan across horizons.
Usage: python experiments/btc_signal_scan.py [--freq daily|1h|30min|15min|5min] [--fetch]
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.btc_data import (
    load_btc,
    load_btc_orderflow,
    fetch_btc,
    ALL_FREQS,
    triple_barrier_label,
    fetch_trades_ccxt,
    aggregate_trades_to_bars,
    add_microstructure_features,
)


def add_ta_indicators(df):
    """Add all classic TA indicators to dataframe."""
    # RSI
    for p in [6, 9, 14, 21, 30]:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
        df[f'rsi_{p}'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # MA
    for p in [5, 10, 20, 50, 100, 200]:
        df[f'ma_{p}'] = df['close'].rolling(p).mean()

    # Bollinger
    for p in [10, 20, 50]:
        mid = df['close'].rolling(p).mean()
        std = df['close'].rolling(p).std()
        df[f'boll_upper_{p}'] = mid + 2 * std
        df[f'boll_lower_{p}'] = mid - 2 * std
        df[f'boll_pctb_{p}'] = (df['close'] - df[f'boll_lower_{p}']) / (df[f'boll_upper_{p}'] - df[f'boll_lower_{p}'] + 1e-10)

    # KDJ
    for p in [9, 14, 21]:
        low_p = df['low'].rolling(p).min()
        high_p = df['high'].rolling(p).max()
        df[f'k_{p}'] = 100 * (df['close'] - low_p) / (high_p - low_p + 1e-10)
        df[f'd_{p}'] = df[f'k_{p}'].rolling(3).mean()
        df[f'j_{p}'] = 3 * df[f'k_{p}'] - 2 * df[f'd_{p}']
        df[f'kdj_golden_{p}'] = ((df[f'k_{p}'] > df[f'd_{p}']) & (df[f'k_{p}'].shift(1) <= df[f'd_{p}'].shift(1))).astype(int)

    # CCI
    for p in [14, 20, 50]:
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(p).mean()
        md = tp.rolling(p).apply(lambda x: np.abs(x - x.mean()).mean())
        df[f'cci_{p}'] = (tp - ma_tp) / (0.015 * md + 1e-10)

    # Williams %R
    for p in [14, 21]:
        hh = df['high'].rolling(p).max()
        ll = df['low'].rolling(p).min()
        df[f'willr_{p}'] = -100 * (hh - df['close']) / (hh - ll + 1e-10)

    # MACD
    for fast, slow, sig in [(12, 26, 9), (5, 13, 5), (24, 52, 18)]:
        ema_f = df['close'].ewm(span=fast).mean()
        ema_s = df['close'].ewm(span=slow).mean()
        macd = ema_f - ema_s
        macd_sig = macd.ewm(span=sig).mean()
        df[f'macd_hist_{fast}_{slow}'] = macd - macd_sig
        df[f'macd_golden_{fast}_{slow}'] = ((macd > macd_sig) & (macd.shift(1) <= macd_sig.shift(1))).astype(int)
        df[f'macd_dead_{fast}_{slow}'] = ((macd < macd_sig) & (macd.shift(1) >= macd_sig.shift(1))).astype(int)

    # Momentum
    for p in [3, 5, 7, 14, 30]:
        df[f'mom_{p}'] = df['close'].pct_change(p)

    # Volatility
    for p in [7, 14, 30]:
        df[f'vol_{p}'] = df['close'].pct_change().rolling(p).std()

    # Volume
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-10)
    df['vol_breakout'] = (df['volume'] > 2 * df['vol_ma20']).astype(int)

    # Body/shadows
    df['body'] = (df['close'] - df['open']) / (df['open'] + 1e-10)

    # Consecutive
    df['consec_up'] = 0
    for i in range(8):
        df['consec_up'] = df['consec_up'] + df['close'].diff().shift(i).gt(0).astype(int)
    df['consec_down'] = 8 - df['consec_up']

    return df


def build_discrete_strategies(test):
    """Return list of (name, subset, direction) tuples."""
    return [
        ('MACD(12,26) golden', test[test['macd_golden_12_26'] == 1], 'long'),
        ('MACD(12,26) death', test[test['macd_dead_12_26'] == 1], 'short'),
        ('MACD(24,52) golden', test[test['macd_golden_24_52'] == 1], 'long'),
        ('RSI14<20 long', test[test['rsi_14'] < 20], 'long'),
        ('RSI14<30 long', test[test['rsi_14'] < 30], 'long'),
        ('RSI14>70 short', test[test['rsi_14'] > 70], 'short'),
        ('RSI14>80 short', test[test['rsi_14'] > 80], 'short'),
        ('RSI9<20 long', test[test['rsi_9'] < 20], 'long'),
        ('RSI30<30 long', test[test['rsi_30'] < 30], 'long'),
        ('Boll20 lower long', test[test['close'] < test['boll_lower_20']], 'long'),
        ('Boll20 upper short', test[test['close'] > test['boll_upper_20']], 'short'),
        ('Boll50 lower long', test[test['close'] < test['boll_lower_50']], 'long'),
        ('Boll20 pctB<0.1', test[test['boll_pctb_20'] < 0.1], 'long'),
        ('Boll20 pctB>0.9', test[test['boll_pctb_20'] > 0.9], 'short'),
        ('Price>MA50 long', test[test['close'] > test['ma_50']], 'long'),
        ('Price<MA50 short', test[test['close'] < test['ma_50']], 'short'),
        ('Price>MA200 long', test[test['close'] > test['ma_200']], 'long'),
        ('Price<MA200 short', test[test['close'] < test['ma_200']], 'short'),
        ('MA50>MA200 long', test[test['ma_50'] > test['ma_200']], 'long'),
        ('MA50<MA200 short', test[test['ma_50'] < test['ma_200']], 'short'),
        ('K9<20 long', test[test['k_9'] < 20], 'long'),
        ('K9>80 short', test[test['k_9'] > 80], 'short'),
        ('J9<0 long', test[test['j_9'] < 0], 'long'),
        ('J9>100 short', test[test['j_9'] > 100], 'short'),
        ('CCI14<-100 long', test[test['cci_14'] < -100], 'long'),
        ('CCI14>100 short', test[test['cci_14'] > 100], 'short'),
        ('CCI50<-100 long', test[test['cci_50'] < -100], 'long'),
        ('WR14<-80 long', test[test['willr_14'] < -80], 'long'),
        ('WR14>-20 short', test[test['willr_14'] > -20], 'short'),
        ('Vol breakout+bull', test[(test['vol_breakout'] == 1) & (test['body'] > 0)], 'long'),
        ('Vol breakout+bear', test[(test['vol_breakout'] == 1) & (test['body'] < 0)], 'short'),
        ('5consec down revert', test[test['consec_down'] >= 5], 'long'),
        ('5consec up revert', test[test['consec_up'] >= 5], 'short'),
        ('Mom14>10% long', test[test['mom_14'] > 0.1], 'long'),
        ('Mom14<-10% revert', test[test['mom_14'] < -0.1], 'long'),
        ('Mom30>20% long', test[test['mom_30'] > 0.2], 'long'),
        ('Mom30<-20% revert', test[test['mom_30'] < -0.2], 'long'),
        ('RSI14<30+Boll lower', test[(test['rsi_14'] < 30) & (test['close'] < test['boll_lower_20'])], 'long'),
        ('CCI<-100+RSI<30', test[(test['cci_14'] < -100) & (test['rsi_14'] < 30)], 'long'),
        ('Price<MA200+RSI<30', test[(test['close'] < test['ma_200']) & (test['rsi_14'] < 30)], 'long'),
    ]


def build_continuous_signals(test):
    """Return dict of signal_name -> series."""
    signals = {}
    for col in ['rsi_6', 'rsi_9', 'rsi_14', 'rsi_21', 'rsi_30']:
        signals[col + '_long'] = -test[col]
        signals[col + '_short'] = test[col]
    for p in [10, 20, 50]:
        signals[f'boll_pctb_{p}_long'] = -test[f'boll_pctb_{p}']
        signals[f'boll_pctb_{p}_short'] = test[f'boll_pctb_{p}']
    for p in [9, 14, 21]:
        signals[f'k_{p}_long'] = -test[f'k_{p}']
        signals[f'k_{p}_short'] = test[f'k_{p}']
    for p in [14, 20, 50]:
        signals[f'cci_{p}_long'] = -test[f'cci_{p}']
        signals[f'cci_{p}_short'] = test[f'cci_{p}']
    for f, s in [(12, 26), (5, 13), (24, 52)]:
        signals[f'macd_{f}_{s}_long'] = test[f'macd_hist_{f}_{s}']
        signals[f'macd_{f}_{s}_short'] = -test[f'macd_hist_{f}_{s}']
    for p in [3, 5, 7, 14, 30]:
        signals[f'mom_{p}_trend'] = test[f'mom_{p}']
        signals[f'mom_{p}_revert'] = -test[f'mom_{p}']
    for p in [50, 200]:
        if f'ma_{p}' in test.columns:
            signals[f'price_ma{p}_long'] = -(test['close'] / test[f'ma_{p}'] - 1)
            signals[f'price_ma{p}_short'] = test['close'] / test[f'ma_{p}'] - 1
    signals['vol_ratio'] = test['vol_ratio']
    signals['consec_up_revert'] = test['consec_up'].astype(float)
    signals['consec_down_revert'] = -test['consec_up'].astype(float)
    return signals


def run_discrete_scan(test, horizons, min_count=5, min_excess=2.0):
    """Scan discrete strategies across horizons."""
    strategies = build_discrete_strategies(test)

    print('\n' + '=' * 95)
    print('DISCRETE STRATEGY SCAN')
    print('=' * 95)

    for h in horizons:
        lbl = f'label_{h}'
        if lbl not in test.columns:
            continue
        base = test[lbl].mean() * 100
        print(f'\n--- Horizon: {h} bars, Base WR(long): {base:.1f}% ---')

        results = []
        for name, sub, direction in strategies:
            if len(sub) < min_count:
                continue
            if direction == 'long':
                wr = sub[lbl].mean() * 100
                excess = wr - base
            else:
                wr = (1 - sub[lbl].mean()) * 100
                excess = wr - (100 - base)
            if abs(excess) > min_excess:
                results.append((name, len(sub), wr, excess))

        results.sort(key=lambda x: x[3], reverse=True)
        if results:
            print('%-35s %6s %8s %8s' % ('Strategy', 'Count', 'WinRate', 'Excess'))
            print('-' * 60)
            for name, cnt, wr, exc in results[:20]:
                print('%-35s %6d %7.1f%% %+7.1f%%' % (name, cnt, wr, exc))
        else:
            print('  (no strategy with >%.1f%% excess)' % min_excess)


def run_auc_scan(test, horizons):
    """Scan continuous signals AUC across horizons."""
    signals = build_continuous_signals(test)

    print('\n\n' + '=' * 95)
    print('CONTINUOUS SIGNAL AUC SCAN')
    print('=' * 95)

    for h in horizons:
        lbl = f'label_{h}'
        if lbl not in test.columns:
            continue
        y = test[lbl].values
        print(f'\n--- Horizon: {h} bars ---')

        res = []
        for sname, sig in signals.items():
            s = sig.values
            valid = ~np.isnan(s)
            if valid.sum() < 100:
                continue
            try:
                auc = roc_auc_score(y[valid], s[valid])
            except:
                continue
            res.append((sname, auc))

        res.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        print('%-35s %8s' % ('Signal', 'AUC'))
        print('-' * 45)
        for name, auc in res[:20]:
            print('%-35s %8.4f' % (name, auc))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTC TA Signal Mining")
    parser.add_argument('--freq', default='daily', choices=ALL_FREQS)
    parser.add_argument('--data-mode', default='ohlcv', choices=['ohlcv', 'orderflow'],
                        help='Data source: ohlcv (default) or orderflow (Binance klines taker-buy proxy)')
    parser.add_argument('--fetch', action='store_true', help='Force re-download data')
    parser.add_argument('--label-mode', default='binary', choices=['binary', 'triple_barrier'],
                        help='Labeling method: binary (next bar up/down) or triple_barrier (AFML)')
    parser.add_argument('--pt-sl', default='1.0,1.0',
                        help='Profit-take,stop-loss multipliers for triple barrier (e.g. 1.0,1.0)')
    parser.add_argument('--micro', action='store_true',
                        help='Add microstructure features from ccxt trades (best effort)')
    parser.add_argument('--trades-days', type=int, default=30,
                        help='Trades lookback days ending at last bar time (default 30)')
    parser.add_argument('--vpin-window', type=int, default=50)
    parser.add_argument('--kyle-window', type=int, default=50)
    args = parser.parse_args()

    if args.data_mode == 'orderflow':
        df = load_btc_orderflow(args.freq)
    elif args.fetch:
        df = fetch_btc(args.freq)
    else:
        df = load_btc(args.freq)

    print(f'\nData: {len(df)} bars, {df["datetime"].iloc[0]} ~ {df["datetime"].iloc[-1]}')

    if args.micro:
        end_dt = pd.to_datetime(df["datetime"].iloc[-1])
        start_dt = end_dt - pd.Timedelta(days=int(args.trades_days))
        trades = fetch_trades_ccxt(start=start_dt, end=end_dt, verbose=True)
        df = aggregate_trades_to_bars(df, trades)
        df = add_microstructure_features(df, vpin_window=int(args.vpin_window), kyle_window=int(args.kyle_window))

    # Add indicators
    df = add_ta_indicators(df)

    # Add labels - horizons depend on frequency
    if args.freq == 'daily':
        horizons = [1, 3, 5, 7, 14, 30]
    elif args.freq == '1h':
        horizons = [1, 4, 12, 24, 48, 168]  # 1h, 4h, 12h, 1d, 2d, 1w
    elif args.freq == '30min':
        horizons = [2, 8, 24, 48, 96, 336]  # 1h, 4h, 12h, 1d, 2d, 1w
    elif args.freq == '15min':
        horizons = [4, 16, 48, 96, 192, 672]  # 1h, 4h, 12h, 1d, 2d, 1w
    elif args.freq == '5min':
        horizons = [12, 48, 144, 288, 576, 2016]  # 1h, 4h, 12h, 1d, 2d, 1w
    else:
        horizons = [1, 5, 10, 20, 50, 100]  # volume bars

    if args.label_mode == 'triple_barrier':
        pt_sl_vals = tuple(float(x) for x in args.pt_sl.split(','))
        print(f'Label mode: Triple Barrier (pt={pt_sl_vals[0]}, sl={pt_sl_vals[1]})')
        for h in horizons:
            tb = triple_barrier_label(df, h, pt_sl=pt_sl_vals)
            # For discrete scan: 1=up, 0=down (map -1->0 for compatibility)
            df[f'label_{h}'] = (tb == 1).astype(int)
            # Keep raw triple barrier label for analysis
            df[f'tb_raw_{h}'] = tb
            dist = tb.value_counts().to_dict()
            print(f'  h={h}: +1={dist.get(1.0,0)}, 0={dist.get(0.0,0)}, -1={dist.get(-1.0,0)}')
    else:
        for h in horizons:
            df[f'label_{h}'] = (df['close'].shift(-h) / df['close'] - 1 > 0).astype(int)

    df = df.dropna().reset_index(drop=True)
    split = int(len(df) * 0.7)
    test = df.iloc[split:].copy()
    print(f'Test: {len(test)} bars, {test["datetime"].iloc[0]} ~ {test["datetime"].iloc[-1]}')

    run_discrete_scan(test, horizons)
    run_auc_scan(test, horizons)

    # Save results
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'experiments')
    os.makedirs(save_dir, exist_ok=True)
    print(f'\nDone. Results printed above.')


if __name__ == '__main__':
    main()
