"""BTC data fetcher: daily (yfinance, 10yr), hourly (ccxt, ~2yr), 5min (ccxt, ~3mo)"""
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


def fetch_btc_hourly(days_back=730):
    import ccxt
    exchange = ccxt.binance()
    since = exchange.milliseconds() - days_back * 24 * 3600 * 1000
    all_data = []
    print(f"Fetching BTC/USDT 1h from Binance, {days_back} days...")
    while True:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['amount'] = df['close'] * df['volume']
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    save_path = os.path.join(DATA_DIR, "btc_1h.pkl")
    df.to_pickle(save_path)
    print(f"BTC 1h saved: {save_path}, {len(df)} rows, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    return df


def fetch_btc_5min(days_back=90):
    import ccxt
    exchange = ccxt.binance()
    since = exchange.milliseconds() - days_back * 24 * 3600 * 1000
    all_data = []
    print(f"Fetching BTC/USDT 5m from Binance, {days_back} days...")
    while True:
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', since=since, limit=1000)
        if not ohlcv:
            break
        all_data.extend(ohlcv)
        since = ohlcv[-1][0] + 1
        if len(ohlcv) < 1000:
            break
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['amount'] = df['close'] * df['volume']
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
    df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    save_path = os.path.join(DATA_DIR, "btc_5min.pkl")
    df.to_pickle(save_path)
    print(f"BTC 5min saved: {save_path}, {len(df)} rows, {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]}")
    return df


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
