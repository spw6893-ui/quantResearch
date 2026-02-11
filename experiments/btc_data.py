"""BTC data fetcher: daily (yfinance, 10yr), hourly (ccxt, 5yr), 5min (ccxt, 5yr)"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)


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


def fetch_btc_hourly(days_back=1825):
    return _fetch_binance_ohlcv('BTC/USDT', '1h', days_back, 'btc_1h.pkl')


def fetch_btc_5min(days_back=1825):
    return _fetch_binance_ohlcv('BTC/USDT', '5m', days_back, 'btc_5min.pkl')


def load_btc(freq="daily"):
    fname = {"daily": "btc_daily.pkl", "1h": "btc_1h.pkl", "5min": "btc_5min.pkl"}[freq]
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path):
        df = pd.read_pickle(path)
        print(f"Loaded {path}: {len(df)} rows")
        return df
    if freq == "daily":
        return fetch_btc_daily()
    elif freq == "1h":
        return fetch_btc_hourly()
    else:
        return fetch_btc_5min()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch BTC OHLCV data")
    parser.add_argument('--freq', default='daily', choices=['daily', '1h', '5min'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.force:
        {"daily": fetch_btc_daily, "1h": fetch_btc_hourly, "5min": fetch_btc_5min}[args.freq]()
    else:
        load_btc(args.freq)
