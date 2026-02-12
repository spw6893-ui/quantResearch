"""BTC data fetcher: daily (yfinance, 10yr), 1h/30min/15min/5min (ccxt/Binance, 5yr)
Also supports Volume Bars: resample raw time bars into volume-based bars.
"""
import os
import sys
import numpy as np
import pandas as pd
from typing import Optional
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)
TRADES_DIR = os.path.join(DATA_DIR, "trades")
os.makedirs(TRADES_DIR, exist_ok=True)

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


def _symbol_to_slug(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_").replace(" ", "")


def _ensure_datetime(x):
    if x is None:
        return None
    ts = pd.to_datetime(x, utc=True)
    # experiments 里统一用 naive datetime（本地/UTC 无所谓，只要一致），避免 tz 混用导致 merge 问题
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _try_read_trades(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if path.endswith(".csv.gz"):
        df = pd.read_csv(path, compression="gzip")
    else:
        df = pd.read_pickle(path)
    if len(df) == 0:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype("int64")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    return df


def _try_write_trades(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".csv.gz"):
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_pickle(path)


def _trades_cache_dir(exchange_id: str, symbol: str) -> str:
    slug = _symbol_to_slug(symbol)
    return os.path.join(TRADES_DIR, f"{exchange_id}_{slug}")


def _read_trades_meta(cache_dir: str) -> dict:
    meta_path = os.path.join(cache_dir, "_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_trades_meta(cache_dir: str, meta: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "_meta.json")
    tmp_path = meta_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp_path, meta_path)


def _list_trade_parquet_files(cache_dir: str):
    """返回 [(min_ts, max_ts, path), ...]，ts 单位 ms。"""
    if not os.path.isdir(cache_dir):
        return []
    out = []
    for fn in os.listdir(cache_dir):
        if not fn.startswith("trades_") or not fn.endswith(".parquet"):
            continue
        # trades_{min}_{max}.parquet
        try:
            core = fn[len("trades_"):-len(".parquet")]
            a, b = core.split("_", 1)
            min_ts = int(a)
            max_ts = int(b)
            out.append((min_ts, max_ts, os.path.join(cache_dir, fn)))
        except Exception:
            continue
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _write_trades_chunk_parquet(cache_dir: str, df_chunk: pd.DataFrame) -> None:
    if len(df_chunk) == 0:
        return
    os.makedirs(cache_dir, exist_ok=True)
    df_chunk = df_chunk.drop_duplicates(subset=["timestamp", "price", "amount"]).sort_values("timestamp")
    min_ts = int(df_chunk["timestamp"].min())
    max_ts = int(df_chunk["timestamp"].max())
    out_path = os.path.join(cache_dir, f"trades_{min_ts}_{max_ts}.parquet")
    # 不直接覆盖同名文件（避免并发/中断导致损坏）
    if os.path.exists(out_path):
        # 同范围重复写，改名
        out_path = os.path.join(cache_dir, f"trades_{min_ts}_{max_ts}_{np.random.randint(0, 1_000_000)}.parquet")
    df_chunk.to_parquet(out_path, index=False)

    meta = _read_trades_meta(cache_dir)
    meta["max_timestamp"] = int(max(meta.get("max_timestamp", 0), max_ts))
    meta["min_timestamp"] = int(min(meta.get("min_timestamp", min_ts), min_ts)) if meta.get("min_timestamp") else int(min_ts)
    meta["updated_at_utc"] = pd.Timestamp.utcnow().isoformat()
    _write_trades_meta(cache_dir, meta)


def load_trades_from_cache(
    cache_dir: str,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    """从 parquet 分片缓存中加载指定时间范围的 trades。"""
    files = _list_trade_parquet_files(cache_dir)
    if not files:
        return pd.DataFrame(columns=["timestamp", "price", "amount", "side", "datetime"])

    chosen = [p for (a, b, p) in files if not (b < start_ms or a > end_ms)]
    if not chosen:
        return pd.DataFrame(columns=["timestamp", "price", "amount", "side", "datetime"])

    parts = []
    for p in chosen:
        try:
            part = pd.read_parquet(p, columns=["timestamp", "price", "amount", "side"])
            parts.append(part)
        except Exception:
            continue
    if not parts:
        return pd.DataFrame(columns=["timestamp", "price", "amount", "side", "datetime"])

    df = pd.concat(parts, ignore_index=True)
    df["timestamp"] = df["timestamp"].astype("int64")
    df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
    df = df.drop_duplicates(subset=["timestamp", "price", "amount"]).sort_values("timestamp").reset_index(drop=True)
    if len(df) > 0:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
    else:
        df["datetime"] = pd.to_datetime([], utc=True).tz_localize(None)
    return df


def fetch_trades_ccxt(
    symbol: str = "BTC/USDT",
    exchange_id: str = "binance",
    start=None,
    end=None,
    days_back: Optional[int] = None,
    limit: int = 1000,
    sleep_s: float = 0.05,
    max_retries: int = 8,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """用 ccxt 拉取历史成交（trades），并做本地缓存（增量追加）。

    重要限制：
      - 交易所 REST 的历史成交拉取通常会非常慢/有上限；建议先从较短窗口（如 7~30 天）开始。
      - 这里只做“尽力拉取 + 增量缓存”，不保证能一次覆盖多年 tick 级历史。
    """
    import time as _time
    import ccxt

    start_dt = _ensure_datetime(start)
    end_dt = _ensure_datetime(end)
    if end_dt is None:
        end_dt = pd.Timestamp.utcnow().tz_localize(None)
    if start_dt is None:
        if days_back is None:
            raise ValueError("fetch_trades_ccxt 需要提供 start 或 days_back")
        start_dt = end_dt - pd.Timedelta(days=int(days_back))

    if start_dt >= end_dt:
        raise ValueError(f"start({start_dt}) 必须早于 end({end_dt})")

    req_start_dt = start_dt
    req_end_dt = end_dt
    req_start_ms = int(req_start_dt.timestamp() * 1000)
    req_end_ms = int(req_end_dt.timestamp() * 1000)

    # 缓存：默认使用 parquet 分片目录，支持中途中断后继续
    cache_dir = None
    cache_file = None
    if cache_path is None:
        cache_dir = _trades_cache_dir(exchange_id, symbol)
    else:
        _, ext = os.path.splitext(cache_path)
        if ext.lower() in (".pkl", ".pickle", ".csv", ".gz", ".parquet"):
            cache_file = cache_path
        else:
            cache_dir = cache_path

    cached_end = None
    if cache_dir is not None:
        meta = _read_trades_meta(cache_dir)
        if meta.get("max_timestamp"):
            cached_end = pd.to_datetime(int(meta["max_timestamp"]), unit="ms")
            if cached_end >= start_dt:
                start_dt = max(start_dt, cached_end + pd.Timedelta(milliseconds=1))
    elif cache_file is not None:
        cached = _try_read_trades(cache_file)
        if len(cached) > 0:
            cached = cached.sort_values("timestamp").drop_duplicates(subset=["timestamp", "price", "amount"])
            cached_end = pd.to_datetime(int(cached["timestamp"].max()), unit="ms")
            if cached_end >= start_dt:
                start_dt = max(start_dt, cached_end + pd.Timedelta(milliseconds=1))

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})

    since_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_rows = []
    retries = 0
    last_progress_ms = since_ms - 1
    batch = 0
    flush_rows = 50_000
    flush_every_batches = 50

    if verbose:
        print(f"[trades] {exchange_id} {symbol}: {start_dt} ~ {end_dt} (limit={limit})")
        if cached_end is not None:
            print(f"[trades] cache max_ts={cached_end} (incremental)")

    while since_ms < end_ms:
        try:
            trades = exchange.fetch_trades(symbol, since=since_ms, limit=limit)
            retries = 0
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise
            wait = min(5.0, 0.5 * retries)
            if verbose:
                print(f"[trades] API error (retry {retries}/{max_retries}, sleep {wait:.1f}s): {e}")
            _time.sleep(wait)
            continue

        if not trades:
            # 没拿到成交，可能是窗口太早/太晚/交易所限制，做一个小步前进避免死循环
            since_ms = min(end_ms, since_ms + 60_000)  # +1min
            _time.sleep(sleep_s)
            continue

        # 规范化输出
        for t in trades:
            ts = int(t.get("timestamp") or 0)
            if ts <= 0 or ts > end_ms:
                continue
            price = float(t.get("price") or np.nan)
            amount = float(t.get("amount") or np.nan)
            if not np.isfinite(price) or not np.isfinite(amount):
                continue
            side = t.get("side")  # 'buy'/'sell'/None
            all_rows.append((ts, price, amount, side))

        new_last_ms = int(trades[-1].get("timestamp") or since_ms)
        if new_last_ms <= last_progress_ms:
            # 交易所返回不推进（重复页），强制前进一步
            since_ms = min(end_ms, since_ms + 1)
        else:
            since_ms = new_last_ms + 1
            last_progress_ms = new_last_ms

        batch += 1
        if verbose and batch % 200 == 0:
            cur = pd.to_datetime(last_progress_ms, unit="ms")
            print(f"[trades] batches={batch:,}, rows={len(all_rows):,}, up_to={cur}")

        # 定期落盘，避免超长窗口中途被 kill/超时导致丢进度
        if (cache_dir is not None) and (len(all_rows) >= flush_rows or batch % flush_every_batches == 0):
            df_chunk = pd.DataFrame(all_rows, columns=["timestamp", "price", "amount", "side"])
            _write_trades_chunk_parquet(cache_dir, df_chunk)
            if verbose:
                print(f"[trades] flushed chunk: {len(df_chunk):,} rows -> {cache_dir}")
            all_rows.clear()

        _time.sleep(sleep_s)

        # 防止一次跑到天荒地老：用户想拉长窗口时更建议先分段/增量
        if batch >= 200_000:
            if verbose:
                print("[trades] 已达到 batch 上限(200k)，提前停止；建议缩短时间窗或分段拉取。")
            break

    # 写入剩余 chunk
    if len(all_rows) > 0:
        df_tail = pd.DataFrame(all_rows, columns=["timestamp", "price", "amount", "side"])
        if cache_dir is not None:
            _write_trades_chunk_parquet(cache_dir, df_tail)
            if verbose:
                print(f"[trades] flushed tail: {len(df_tail):,} rows -> {cache_dir}")
        elif cache_file is not None:
            cached = _try_read_trades(cache_file)
            merged = pd.concat([cached, df_tail], ignore_index=True) if len(cached) > 0 else df_tail
            merged = merged.drop_duplicates(subset=["timestamp", "price", "amount"]).sort_values("timestamp")
            merged["datetime"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True).dt.tz_localize(None)
            _try_write_trades(merged, cache_file)
            if verbose:
                print(f"[trades] saved cache: {cache_file} ({len(merged):,} rows)")

    # 返回用户请求范围内的 trades（避免把全量缓存一次性塞进内存）
    if cache_dir is not None:
        return load_trades_from_cache(cache_dir, req_start_ms, req_end_ms)
    if cache_file is not None:
        df = _try_read_trades(cache_file)
        if len(df) == 0:
            return df
        df = df[(df["timestamp"] >= req_start_ms) & (df["timestamp"] <= req_end_ms)]
        df = df.drop_duplicates(subset=["timestamp", "price", "amount"]).sort_values("timestamp").reset_index(drop=True)
        return df
    return pd.DataFrame(columns=["timestamp", "price", "amount", "side", "datetime"])


def _infer_trade_sign(trade_prices: np.ndarray, trade_sides: np.ndarray) -> np.ndarray:
    """把成交方向转成 +1/-1（买/卖）。side 缺失时用 tick rule 近似。"""
    sign = np.zeros(len(trade_prices), dtype=np.int8)
    if trade_sides is not None and len(trade_sides) == len(trade_prices):
        # 先用 side
        side_buy = trade_sides == "buy"
        side_sell = trade_sides == "sell"
        sign[side_buy] = 1
        sign[side_sell] = -1
    # side 缺失的，用 tick rule：价格上行视为买，价格下行视为卖，持平沿用上一次
    need = sign == 0
    if need.any():
        px = trade_prices.astype(float)
        d = np.diff(px, prepend=px[0])
        tick = np.where(d > 0, 1, np.where(d < 0, -1, 0)).astype(np.int8)
        # 平盘继承
        last = 1
        for i in range(len(tick)):
            if tick[i] == 0:
                tick[i] = last
            else:
                last = tick[i]
        sign[need] = tick[need]
    return sign


def aggregate_trades_to_bars(
    bars_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """把成交数据聚合到 bars（支持时间K线/成交量K线）。

    约定：用相邻两根 bar 的 datetime 作为区间边界 [start, next_start)。
    最后一根 bar 的 end 用“中位数 bar 间隔”外推。
    """
    if len(trades_df) == 0 or len(bars_df) == 0:
        out = bars_df.copy()
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.tz_localize(None)
        if len(out) >= 2:
            ts = out["datetime"].astype("int64").values // 1_000_000
            diffs = np.diff(ts)
            median_delta = int(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 60_000
            end_ts = np.r_[ts[1:], ts[-1] + median_delta]
            out["bar_duration_sec"] = (end_ts - ts) / 1000.0
        else:
            out["bar_duration_sec"] = np.nan

        # 保持字段存在，方便下游统一处理（没拿到 trades 时用 0，占位但不中断训练）
        zero_cols = [
            "n_trades", "trade_volume", "trade_amount", "buy_volume", "sell_volume",
            "signed_volume", "signed_amount", "ofi", "dollar_imbalance", "vwap_trades",
            "avg_trade_size",
        ]
        for c in zero_cols:
            out[c] = 0.0
        return out

    bars = bars_df.copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True).dt.tz_localize(None)
    bar_ts = bars["datetime"].astype("int64").values // 1_000_000  # ns->ms
    if len(bar_ts) < 2:
        median_delta = 60_000
    else:
        diffs = np.diff(bar_ts)
        median_delta = int(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 60_000
    bar_end_ts = np.r_[bar_ts[1:], bar_ts[-1] + median_delta]
    bars["bar_duration_sec"] = (bar_end_ts - bar_ts) / 1000.0

    trades = trades_df.copy()
    if "timestamp" not in trades.columns:
        raise ValueError("trades_df 缺少 timestamp(ms) 列")
    t_ts = trades["timestamp"].astype("int64").values
    # 只保留落在 bars 覆盖范围内的成交（含最后一根外推 end）
    valid = (t_ts >= bar_ts[0]) & (t_ts < bar_end_ts[-1])
    trades = trades.loc[valid].reset_index(drop=True)
    if len(trades) == 0:
        return aggregate_trades_to_bars(bars, pd.DataFrame(), eps=eps)

    t_ts = trades["timestamp"].astype("int64").values
    t_px = trades["price"].astype(float).values
    t_amt = trades["amount"].astype(float).values
    t_side = trades["side"].astype(str).values if "side" in trades.columns else None
    t_sign = _infer_trade_sign(t_px, t_side)

    # trade -> bar index (start inclusive)
    bar_idx = np.searchsorted(bar_ts, t_ts, side="right") - 1
    in_range = (bar_idx >= 0) & (bar_idx < len(bar_ts))
    bar_idx = bar_idx[in_range]
    t_px = t_px[in_range]
    t_amt = t_amt[in_range]
    t_sign = t_sign[in_range]

    # 排除落在 end 之后的（主要是最后一根外推 end 的精度问题）
    t_ts2 = t_ts[in_range]
    ok_end = t_ts2 < bar_end_ts[bar_idx]
    bar_idx = bar_idx[ok_end]
    t_px = t_px[ok_end]
    t_amt = t_amt[ok_end]
    t_sign = t_sign[ok_end]

    n_bars = len(bar_ts)
    n_trades = np.bincount(bar_idx, minlength=n_bars).astype(float)
    trade_volume = np.bincount(bar_idx, weights=t_amt, minlength=n_bars).astype(float)
    trade_amount = np.bincount(bar_idx, weights=t_amt * t_px, minlength=n_bars).astype(float)
    signed_volume = np.bincount(bar_idx, weights=t_amt * t_sign, minlength=n_bars).astype(float)
    signed_amount = np.bincount(bar_idx, weights=t_amt * t_px * t_sign, minlength=n_bars).astype(float)
    buy_volume = np.bincount(bar_idx, weights=t_amt * (t_sign > 0), minlength=n_bars).astype(float)
    sell_volume = np.bincount(bar_idx, weights=t_amt * (t_sign < 0), minlength=n_bars).astype(float)

    ofi = signed_volume / (trade_volume + eps)
    dollar_imb = signed_amount / (trade_amount + eps)
    vwap_trades = trade_amount / (trade_volume + eps)
    avg_trade_size = trade_volume / (n_trades + eps)

    bars["n_trades"] = n_trades
    bars["trade_volume"] = trade_volume
    bars["trade_amount"] = trade_amount
    bars["buy_volume"] = buy_volume
    bars["sell_volume"] = sell_volume
    bars["signed_volume"] = signed_volume
    bars["signed_amount"] = signed_amount
    bars["ofi"] = ofi
    bars["dollar_imbalance"] = dollar_imb
    bars["vwap_trades"] = vwap_trades
    bars["avg_trade_size"] = avg_trade_size
    return bars


def add_microstructure_features(
    bars_df: pd.DataFrame,
    vpin_window: int = 50,
    kyle_window: int = 50,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """在已聚合成交的 bars 上，计算微观结构衍生特征（Lopez de Prado 风格的“可用近似”）。"""
    df = bars_df.copy()
    if "buy_volume" not in df.columns or "sell_volume" not in df.columns:
        return df

    vol = (df["buy_volume"].fillna(0) + df["sell_volume"].fillna(0)).astype(float)
    imb = (df["buy_volume"].fillna(0) - df["sell_volume"].fillna(0)).astype(float)
    df["trade_imbalance_abs"] = imb.abs() / (vol + eps)
    df[f"vpin_{vpin_window}"] = df["trade_imbalance_abs"].rolling(vpin_window).mean()

    # Kyle's lambda: r_t = lambda * q_t + e, 其中 q_t 用 signed_amount(更接近 dollar volume)
    logp = np.log(df["close"].astype(float).replace(0, np.nan))
    r = logp.diff()
    q = df.get("signed_amount", pd.Series(np.nan, index=df.index)).astype(float)
    cov_rq = r.rolling(kyle_window).cov(q)
    var_q = q.rolling(kyle_window).var()
    df[f"kyle_lambda_{kyle_window}"] = cov_rq / (var_q + eps)

    # 交易价格偏离：close 相对成交 VWAP
    if "vwap_trades" in df.columns:
        df["close_vwap_trades_ratio"] = df["close"].astype(float) / (df["vwap_trades"].astype(float) + eps)

    return df

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


def fetch_binance_klines_extended(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    start=None,
    end=None,
    days_back: Optional[int] = None,
    limit: int = 1000,
    sleep_s: float = 0.2,
    verbose: bool = True,
) -> pd.DataFrame:
    """拉 Binance 原生 Klines（带 taker buy volume / trades 数等扩展字段）。

    说明：
      - 这是“日级 orderflow 近似”（来自 taker buy volume），不是 L2 orderbook。
      - 相比逐笔 trades，拉取量极小（5 年日线约 1800 行），非常适合做日级微观特征。
    """
    import time as _time
    import ccxt

    end_dt = _ensure_datetime(end)
    if end_dt is None:
        end_dt = pd.Timestamp.utcnow().tz_localize(None)
    start_dt = _ensure_datetime(start)
    if start_dt is None:
        if days_back is None:
            raise ValueError("fetch_binance_klines_extended 需要提供 start 或 days_back")
        start_dt = end_dt - pd.Timedelta(days=int(days_back))

    if start_dt >= end_dt:
        raise ValueError(f"start({start_dt}) 必须早于 end({end_dt})")

    exchange = ccxt.binance({"enableRateLimit": True})

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    rows = []
    cur_ms = start_ms
    page = 0
    if verbose:
        print(f"[klines] binance {symbol} {interval}: {start_dt} ~ {end_dt} (limit={limit})")

    while cur_ms < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cur_ms, "limit": limit}
        # endTime 只是上界提示；部分情况下交易所会忽略/调整，后续再过滤
        params["endTime"] = end_ms

        data = exchange.publicGetKlines(params)
        if not data:
            break

        rows.extend(data)
        last_open_ms = int(data[-1][0])
        if last_open_ms <= cur_ms:
            cur_ms += 1
        else:
            cur_ms = last_open_ms + 1

        page += 1
        if verbose and page % 5 == 0:
            ts = pd.to_datetime(last_open_ms, unit="ms", utc=True)
            print(f"[klines] pages={page}, rows={len(rows)}, up_to={ts}")
        _time.sleep(sleep_s)

        if len(data) < limit:
            break

    if not rows:
        return pd.DataFrame()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "n_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["open_time"] = df["open_time"].astype("int64")
    df = df[(df["open_time"] >= start_ms) & (df["open_time"] <= end_ms)].copy()

    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["n_trades"] = df["n_trades"].astype("int64")

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    df["amount"] = df["quote_volume"]
    df = df[[
        "datetime", "open", "high", "low", "close", "volume", "amount",
        "quote_volume", "n_trades", "taker_buy_base", "taker_buy_quote",
    ]]
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def build_daily_orderflow_features_from_klines(df: pd.DataFrame, eps: float = 1e-10) -> pd.DataFrame:
    """兼容旧名称：等同于 build_orderflow_features_from_klines(keep_raw_fields=True)。"""
    return build_orderflow_features_from_klines(df, keep_raw_fields=True, eps=eps)


def build_orderflow_features_from_klines(
    df: pd.DataFrame,
    keep_raw_fields: bool = True,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """把扩展 Klines 转成（日/小时/分钟）级别的 orderflow/微观结构特征（无需逐笔 trades）。

    这些字段来自 Binance Klines 的扩展列：
      - n_trades
      - taker_buy_base / taker_buy_quote
      - quote_volume（这里也会映射到 amount）
    """
    out = df.copy()
    need = {"taker_buy_base", "taker_buy_quote", "quote_volume", "n_trades", "volume"}
    miss = [c for c in need if c not in out.columns]
    if miss:
        raise ValueError(f"缺少字段 {miss}，请用 fetch_binance_klines_extended 获取")

    out["buy_volume"] = out["taker_buy_base"].astype(float)
    out["sell_volume"] = (out["volume"].astype(float) - out["buy_volume"]).clip(lower=0.0)
    out["signed_volume"] = out["buy_volume"] - out["sell_volume"]
    out["ofi"] = out["signed_volume"] / (out["volume"].astype(float) + eps)

    out["buy_amount"] = out["taker_buy_quote"].astype(float)
    out["sell_amount"] = (out["quote_volume"].astype(float) - out["buy_amount"]).clip(lower=0.0)
    out["signed_amount"] = out["buy_amount"] - out["sell_amount"]
    out["dollar_imbalance"] = out["signed_amount"] / (out["quote_volume"].astype(float) + eps)

    # 成交 VWAP（基于 quote/base），和 OHLC 的 close 不一定一致
    out["vwap_trades"] = out["quote_volume"].astype(float) / (out["volume"].astype(float) + eps)
    out["avg_trade_size"] = out["volume"].astype(float) / (out["n_trades"].astype(float) + eps)

    if not keep_raw_fields:
        keep = [
            "datetime", "open", "high", "low", "close", "volume", "amount",
            "n_trades",
            "buy_volume", "sell_volume", "ofi",
            "buy_amount", "sell_amount", "dollar_imbalance",
            "vwap_trades", "avg_trade_size",
        ]
        out = out[[c for c in keep if c in out.columns]].copy()

    return out


def add_orderflow_microstructure_features(
    df: pd.DataFrame,
    vpin_window: int = 50,
    kyle_window: int = 50,
    eps: float = 1e-10,
) -> pd.DataFrame:
    """基于 orderflow(Klines 扩展字段) 增强微观结构特征。

    目标：在没有历史 L2 orderbook 的情况下，提供接近“盘口微观结构”的可用代理特征：
      - VPIN: 订单流不平衡的平滑度量
      - Kyle's lambda: 价格冲击系数（用 signed_amount 近似）
      - Amihud: 流动性/冲击代理（|ret| / dollar_volume）
      - 交易强度：trades/sec、volume/sec
    """
    out = df.copy()
    if len(out) == 0:
        return out

    # 基础列检查 / 兜底推导
    if "bar_duration_sec" not in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True).dt.tz_localize(None)
        dt = out["datetime"].diff().dt.total_seconds()
        dur = float(dt.median()) if dt.notna().any() else np.nan
        out["bar_duration_sec"] = dur

    if "buy_volume" not in out.columns or "sell_volume" not in out.columns:
        if "taker_buy_base" in out.columns and "volume" in out.columns:
            out["buy_volume"] = out["taker_buy_base"].astype(float)
            out["sell_volume"] = (out["volume"].astype(float) - out["buy_volume"]).clip(lower=0.0)
        else:
            return out

    if "buy_amount" not in out.columns or "sell_amount" not in out.columns:
        if "taker_buy_quote" in out.columns and "amount" in out.columns:
            out["buy_amount"] = out["taker_buy_quote"].astype(float)
            out["sell_amount"] = (out["amount"].astype(float) - out["buy_amount"]).clip(lower=0.0)
        else:
            out["buy_amount"] = out.get("buy_volume", 0.0) * 0.0
            out["sell_amount"] = out.get("sell_volume", 0.0) * 0.0

    out["signed_volume"] = out["buy_volume"].astype(float) - out["sell_volume"].astype(float)
    out["ofi"] = out["signed_volume"] / (out["volume"].astype(float) + eps) if "volume" in out.columns else out.get("ofi", 0.0)

    out["signed_amount"] = out["buy_amount"].astype(float) - out["sell_amount"].astype(float)
    if "amount" in out.columns:
        out["dollar_imbalance"] = out["signed_amount"] / (out["amount"].astype(float) + eps)

    # VPIN（窗口内 |imbalance| 的均值）
    total_vol = (out["buy_volume"].astype(float) + out["sell_volume"].astype(float))
    imb_abs = out["signed_volume"].astype(float).abs() / (total_vol + eps)
    out[f"vpin_{vpin_window}"] = imb_abs.rolling(int(vpin_window)).mean()

    # Kyle's lambda：log-return ~ lambda * signed_amount
    if "close" in out.columns:
        logp = np.log(out["close"].astype(float).replace(0, np.nan))
        r = logp.diff()
        q = out["signed_amount"].astype(float)
        cov_rq = r.rolling(int(kyle_window)).cov(q)
        var_q = q.rolling(int(kyle_window)).var()
        out[f"kyle_lambda_{kyle_window}"] = cov_rq / (var_q + eps)

        # Amihud illiquidity proxy
        if "amount" in out.columns:
            ret = out["close"].astype(float).pct_change()
            out["amihud_illiq"] = ret.abs() / (out["amount"].astype(float) + eps)

    # close 相对成交 VWAP 的偏离（盘口压力代理）
    if "vwap_trades" in out.columns and "close" in out.columns:
        out["close_vwap_trades_ratio"] = out["close"].astype(float) / (out["vwap_trades"].astype(float) + eps)

    # 交易强度
    if "n_trades" in out.columns:
        out["trades_per_sec"] = out["n_trades"].astype(float) / (out["bar_duration_sec"].astype(float) + eps)
    if "volume" in out.columns:
        out["volume_per_sec"] = out["volume"].astype(float) / (out["bar_duration_sec"].astype(float) + eps)

    return out


def load_btc_orderflow(freq: str) -> pd.DataFrame:
    """加载已准备好的 orderflow 数据集（来自 Binance 扩展 Klines）。"""
    if freq == "daily":
        path = os.path.join(DATA_DIR, "btc_daily_orderflow.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 {path}，请先运行 --fetch-daily-orderflow")
        df = pd.read_pickle(path)
        return df
    if freq == "1h":
        path = os.path.join(DATA_DIR, "btc_1h_orderflow.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 {path}，请先运行 --fetch-1h-orderflow")
        df = pd.read_pickle(path)
        return df
    if freq == "5min":
        path = os.path.join(DATA_DIR, "btc_5min_orderflow.parquet")
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 {path}，请先运行 --fetch-5min-orderflow")
        df = pd.read_parquet(path)
        return df
    raise ValueError("orderflow 目前只支持 freq=daily/1h/5min")


def fetch_btc_daily_orderflow(days_back: int = 1825, save_name: str = "btc_daily_orderflow.pkl") -> pd.DataFrame:
    """拉取 5 年（日线）taker buy 量等字段，并保存日级 orderflow 特征。"""
    # 取到最近一个完整 UTC 日（避免当天未收盘造成口径混乱）
    end_dt = pd.Timestamp.utcnow().floor("D").tz_localize(None)
    start_dt = end_dt - pd.Timedelta(days=int(days_back))

    kl = fetch_binance_klines_extended(symbol="BTCUSDT", interval="1d", start=start_dt, end=end_dt, verbose=True)
    if len(kl) == 0:
        return kl
    feat = build_orderflow_features_from_klines(kl, keep_raw_fields=True)
    feat["bar_duration_sec"] = 86400.0
    feat = add_orderflow_microstructure_features(feat)
    save_path = os.path.join(DATA_DIR, save_name)
    feat.to_pickle(save_path)
    print(f"[orderflow] saved: {save_path}, {len(feat):,} rows, {feat['datetime'].iloc[0]} ~ {feat['datetime'].iloc[-1]}")
    return feat


def fetch_btc_1h_orderflow(days_back: int = 1825, save_name: str = "btc_1h_orderflow.pkl") -> pd.DataFrame:
    """拉取 5 年（小时线）taker buy 量等字段，并保存小时级 orderflow 特征。

    说明：5 年小时线大约 1825*24 ≈ 4.4 万行，按 1000/页大约 44 页，实际拉取通常在 1 分钟级别。
    """
    end_dt = pd.Timestamp.utcnow().floor("h").tz_localize(None)
    start_dt = end_dt - pd.Timedelta(days=int(days_back))

    kl = fetch_binance_klines_extended(symbol="BTCUSDT", interval="1h", start=start_dt, end=end_dt, verbose=True)
    if len(kl) == 0:
        return kl
    feat = build_orderflow_features_from_klines(kl, keep_raw_fields=True)
    feat["bar_duration_sec"] = 3600.0
    feat = add_orderflow_microstructure_features(feat)
    save_path = os.path.join(DATA_DIR, save_name)
    feat.to_pickle(save_path)
    print(f"[orderflow] saved: {save_path}, {len(feat):,} rows, {feat['datetime'].iloc[0]} ~ {feat['datetime'].iloc[-1]}")
    return feat


def fetch_btc_5min_orderflow(
    days_back: int = 1825,
    save_name: str = "btc_5min_orderflow.parquet",
    sleep_s: float = 0.15,
) -> pd.DataFrame:
    """拉取 5 年（5分钟线）taker buy 量等字段，并保存 5min 级 orderflow 特征。

    数据量：约 1825*24*12 ≈ 52.6 万行，适合用 parquet 保存（避免单文件过大导致 push 失败）。
    """
    end_dt = pd.Timestamp.utcnow().floor("5min").tz_localize(None)
    start_dt = end_dt - pd.Timedelta(days=int(days_back))

    kl = fetch_binance_klines_extended(
        symbol="BTCUSDT",
        interval="5m",
        start=start_dt,
        end=end_dt,
        sleep_s=sleep_s,
        verbose=True,
    )
    if len(kl) == 0:
        return kl
    feat = build_orderflow_features_from_klines(kl, keep_raw_fields=False)
    feat["bar_duration_sec"] = 300.0
    feat = add_orderflow_microstructure_features(feat)

    # 尽量压小体积，便于存储/推送
    for c in feat.columns:
        if c in ("datetime",):
            continue
        if c == "n_trades":
            feat[c] = feat[c].astype("int32")
        else:
            if pd.api.types.is_float_dtype(feat[c]) or pd.api.types.is_integer_dtype(feat[c]):
                feat[c] = feat[c].astype("float32")

    save_path = os.path.join(DATA_DIR, save_name)
    feat.to_parquet(save_path, index=False, compression="zstd")
    print(f"[orderflow] saved: {save_path}, {len(feat):,} rows, {feat['datetime'].iloc[0]} ~ {feat['datetime'].iloc[-1]}")
    return feat


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




def triple_barrier_label(df, max_horizon, pt_sl=(1.0, 1.0), vol_lookback=20):
    """Triple Barrier Method (Lopez de Prado, AFML Ch.3).

    For each bar, set three barriers:
      - Upper (profit-take): entry + pt_mult * daily_vol
      - Lower (stop-loss):   entry - sl_mult * daily_vol
      - Vertical:            max_horizon bars ahead

    Label: +1 if upper hit first, -1 if lower hit first, 0 if vertical expires.

    Args:
        df: DataFrame with close column.
        max_horizon: vertical barrier in bar counts.
        pt_sl: (profit_take_mult, stop_loss_mult) relative to rolling volatility.
        vol_lookback: lookback for volatility estimate.

    Returns:
        pd.Series of labels {-1, 0, 1}, same index as df.
    """
    close = df["close"].values
    # Rolling volatility (std of log returns)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    vol = log_ret.rolling(vol_lookback).std().values

    pt_mult, sl_mult = pt_sl
    n = len(close)
    labels = np.full(n, np.nan)

    for i in range(vol_lookback, n):
        if np.isnan(vol[i]) or vol[i] < 1e-10:
            continue
        entry = close[i]
        upper = entry * (1 + pt_mult * vol[i])
        lower = entry * (1 - sl_mult * vol[i])
        end = min(i + max_horizon, n - 1)

        label = 0  # default: vertical barrier (no touch)
        for j in range(i + 1, end + 1):
            if close[j] >= upper:
                label = 1
                break
            if close[j] <= lower:
                label = -1
                break
        labels[i] = label

    return pd.Series(labels, index=df.index)

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
    parser.add_argument('--fetch-trades', action='store_true', help='Fetch & cache trades via ccxt')
    parser.add_argument('--trades-days', type=int, default=7, help='Trades history window in days (default 7)')
    parser.add_argument('--build-micro', action='store_true', help='Build microstructure features and save as cached bars')
    parser.add_argument('--fetch-daily-orderflow', action='store_true',
                        help='Fetch 5y daily orderflow features (taker buy volume from Binance klines)')
    parser.add_argument('--fetch-1h-orderflow', action='store_true',
                        help='Fetch 5y 1h orderflow features (taker buy volume from Binance klines)')
    parser.add_argument('--fetch-5min-orderflow', action='store_true',
                        help='Fetch 5y 5min orderflow features (taker buy volume from Binance klines)')
    args = parser.parse_args()
    if args.force:
        fetch_btc(args.freq, args.days)
    else:
        load_btc(args.freq)

    if args.fetch_trades or args.build_micro:
        # 为了避免一次拉太久，这里只默认拉最近 N 天；需要更长窗口可多次运行增量缓存
        fetch_trades_ccxt(days_back=args.trades_days, verbose=True)

    if args.build_micro:
        bars = load_btc(args.freq)
        trades = fetch_trades_ccxt(days_back=args.trades_days, verbose=True)
        bars2 = aggregate_trades_to_bars(bars, trades)
        bars2 = add_microstructure_features(bars2)
        out_path = os.path.join(DATA_DIR, f"btc_{args.freq}_micro.pkl")
        bars2.to_pickle(out_path)
        print(f"[micro] saved: {out_path} ({len(bars2):,} rows)")

    if args.fetch_daily_orderflow:
        fetch_btc_daily_orderflow(days_back=int(args.days))

    if args.fetch_1h_orderflow:
        fetch_btc_1h_orderflow(days_back=int(args.days))

    if args.fetch_5min_orderflow:
        fetch_btc_5min_orderflow(days_back=int(args.days))
