"""BTC data fetcher: daily (yfinance, 10yr), 1h/30min/15min/5min (ccxt/Binance, 5yr)
Also supports Volume Bars: resample raw time bars into volume-based bars.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

# Mapping from our freq names to Binance timeframes and filenames
FREQ_MAP = {
    '5min':  ('5m',  'btc_5min.pkl'),
    '15min': ('15m', 'btc_15min.pkl'),
    '30min': ('30m', 'btc_30min.pkl'),
    '1h':    ('1h',  'btc_1h.pkl'),
    'daily': (None,  'btc_daily.pkl'),
}

# Volume bar configs: threshold in BTC volume per bar
VOLUME_BAR_CONFIGS = {
    'vbar_100':  100,    # ~100 BTC per bar
    'vbar_500':  500,    # ~500 BTC per bar
    'vbar_1000': 1000,   # ~1000 BTC per bar
}

ALL_FREQS = list(FREQ_MAP.keys()) + list(VOLUME_BAR_CONFIGS.keys())


def fetch_btc_daily(start="2016-01-01", end="2026-02-11"):
    import yfinance as yf
    btc = yf.download('BTC-USD', start=start, end=end, interval='1d')
    btc.columns = [c[0].lower() for c in btc.columns]
    btc = btc.reset_index()
    btc.columns = [c.lower() for c in btc.columns]
    btc = btc.rename(columns={'date': 'datetime'})
    btc['amount'] = btc['close'] * btc['volume']
    btc = btc[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    btc = btc.dropna().sort_values('datetime').reset_index(drop=True)
    save_path = os.path.join(DATA_DIR, "btc_daily.pkl")
    btc.to_pickle(save_path)
    print(f"BTC daily saved: {save_path}, {len(btc)} rows, {btc['datetime'].iloc[0]} ~ {btc['datetime'].iloc[-1]}")
    return btc


def _fetch_binance_ohlcv(symbol, timeframe, days_back, save_name):
    """Generic paginated fetcher for Binance via ccxt."""
    import ccxt
    import time as _time
    exchange = ccxt.binance()
    since = exchange.milliseconds() - days_back * 24 * 3600 * 1000
    all_data = []
    batch = 0
    print(f"Fetching {symbol} {timeframe} from Binance, {days_back} days (~{days_back/365:.1f}yr)...")
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception as e:
            print(f"  API error (retrying in 5s): {e}")
            _time.sleep(5)
            continue
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        batch += 1
        if batch % 50 == 0:
            ts = pd.to_datetime(ohlcv[-1][0], unit='ms')
            print(f"  ... {len(all_data):,} candles fetched, up to {ts}")
        if len(ohlcv) < 1000:
            break
        _time.sleep(0.1)  # rate limit
    if not all_data:
        print("  No data returned")
        return pd.DataFrame()
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['amount'] = df['close'] * df['volume']
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    save_path = os.path.join(DATA_DIR, save_name)
    df.to_pickle(save_path)
    print(f"Saved: {save_path}, {len(df):,} rows, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    return df


def make_volume_bars(df, vol_threshold):
    """Convert time-based OHLCV into volume bars.

    Each bar accumulates volume until vol_threshold is reached, then closes.
    Returns a new DataFrame with the same OHLCV+amount columns.
    """
    datetimes = df['datetime'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    volumes = df['volume'].values
    amounts = df['amount'].values

    bars = []
    cum_vol = 0.0
    cum_amount = 0.0
    bar_open = 0.0
    bar_high = -1e18
    bar_low = 1e18
    bar_start_dt = None

    for i in range(len(df)):
        if bar_start_dt is None:
            bar_open = opens[i]
            bar_start_dt = datetimes[i]
            bar_high = highs[i]
            bar_low = lows[i]

        if highs[i] > bar_high:
            bar_high = highs[i]
        if lows[i] < bar_low:
            bar_low = lows[i]
        cum_vol += volumes[i]
        cum_amount += amounts[i]

        if cum_vol >= vol_threshold:
            bars.append((
                bar_start_dt, bar_open, bar_high, bar_low,
                closes[i], cum_vol, cum_amount,
            ))
            cum_vol = 0.0
            cum_amount = 0.0
            bar_start_dt = None

    result = pd.DataFrame(bars, columns=['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'])
    return result


def build_volume_bars(freq, source_freq='5min'):
    """Build volume bars from raw time-bar data.

    Args:
        freq: one of 'vbar_100', 'vbar_500', 'vbar_1000'
        source_freq: base time bars to aggregate from (default '5min')
    """
    vol_threshold = VOLUME_BAR_CONFIGS[freq]
    print(f"Building volume bars: {freq} (threshold={vol_threshold} BTC) from {source_freq} data...")

    # Load source data
    source_df = load_btc(source_freq)
    vbars = make_volume_bars(source_df, vol_threshold)

    save_path = os.path.join(DATA_DIR, f"btc_{freq}.pkl")
    vbars.to_pickle(save_path)
    print(f"Volume bars saved: {save_path}, {len(vbars):,} bars")
    if len(vbars) > 0:
        total_time = (pd.Timestamp(vbars['datetime'].iloc[-1]) - pd.Timestamp(vbars['datetime'].iloc[0]))
        avg_duration = total_time / len(vbars)
        print(f"  Time range: {vbars['datetime'].iloc[0]} ~ {vbars['datetime'].iloc[-1]}")
        print(f"  Avg bar duration: {avg_duration}")
        print(f"  Avg volume/bar: {vbars['volume'].mean():.1f} BTC")
    return vbars


def fetch_btc(freq, days_back=1825):
    """Fetch BTC data for any supported frequency."""
    if freq in VOLUME_BAR_CONFIGS:
        return build_volume_bars(freq)
    if freq == 'daily':
        return fetch_btc_daily()
    timeframe, save_name = FREQ_MAP[freq]
    return _fetch_binance_ohlcv('BTC/USDT', timeframe, days_back, save_name)


def load_btc(freq="daily"):
    """Load cached BTC data, or fetch if not found."""
    if freq in VOLUME_BAR_CONFIGS:
        path = os.path.join(DATA_DIR, f"btc_{freq}.pkl")
        if os.path.exists(path):
            df = pd.read_pickle(path)
            print(f"Loaded {path}: {len(df)} rows")
            return df
        return build_volume_bars(freq)

    if freq not in FREQ_MAP:
        raise ValueError(f"Unsupported freq: {freq}. Choose from {ALL_FREQS}")
    _, fname = FREQ_MAP[freq]
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_pickle(path)
        print(f"Loaded {path}: {len(df)} rows")
        return df
    return fetch_btc(freq)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch BTC OHLCV data")
    parser.add_argument('--freq', default='daily', choices=ALL_FREQS)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--days', type=int, default=1825, help='Days of history (default 1825=5yr)')
    args = parser.parse_args()
    if args.force:
        fetch_btc(args.freq, args.days)
    else:
        load_btc(args.freq)
