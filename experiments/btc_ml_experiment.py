"""BTC ML Experiment: train LightGBM/XGBoost + Transformer-LSTM on BTC data.
Reuses the existing model pipeline from the main project.
Usage: python experiments/btc_ml_experiment.py [--freq daily|1h|30min|15min|5min] [--horizons 1,7,14]
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset, DataLoader
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.btc_data import (
    load_btc,
    load_btc_orderflow,
    ALL_FREQS,
    triple_barrier_label,
    fetch_trades_ccxt,
    aggregate_trades_to_bars,
    add_microstructure_features,
)
from experiments.btc_signal_scan import add_ta_indicators
from config.settings import LGBM_CONFIG, XGB_CONFIG, MAX_FEATURES, RANDOM_SEED, TRAINING_CONFIG
from config.settings import CV_CONFIG
from data.feature_engineering import FeatureEngineer
from models.lgbm_model import LGBMModel
from models.xgb_model import XGBModel
from models.trainer import ModelTrainer, EarlyStopping, TimeSeriesSplitter
from utils.helpers import set_seed

set_seed(RANDOM_SEED)


def _to_utc_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt.dt.tz_convert(None)


def load_hf_csv(csv_path: str) -> pd.DataFrame:
    """加载你预先计算好的 BTC 高频特征 CSV（通常为小时线）。"""
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError("hf-csv 缺少 datetime 列")
    df["datetime"] = _to_utc_naive_datetime(df["datetime"])
    df = df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    if "amount" not in df.columns and all(c in df.columns for c in ("close", "volume")):
        df["amount"] = df["close"].astype(float) * df["volume"].astype(float)
    # 尽量把 True/False 字段转成 0/1（避免后续特征选择/缩放遇到 object dtype）
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        if c in ("datetime", "bar_end_time", "feature_time"):
            continue
        # 小集合才尝试判断（避免对高基数列浪费时间）
        vals = df[c].dropna().astype(str).unique()
        if len(vals) == 0 or len(vals) > 5:
            continue
        s = set(v.lower() for v in vals)
        if s.issubset({"true", "false"}):
            df[c] = df[c].astype(str).str.lower().map({"true": 1, "false": 0}).astype("int8")
    return df


def _seq_to_tabular_fast(X_rows: np.ndarray, seq_length: int) -> np.ndarray:
    """把 (N_rows, F) 快速转为 (N_samples, 3F)：last + mean + std（窗口长度=seq_length）。

    对齐口径与 create_sequences 一致：用过去 seq_length 根预测“下一根”的 label。
      - window: [s, s+seq_length)
      - label : y[s+seq_length]
    """
    X = np.asarray(X_rows, dtype=np.float32)
    n_rows, n_feat = X.shape
    seq_length = int(seq_length)
    n_samples = n_rows - seq_length
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    csum = np.cumsum(X.astype(np.float64), axis=0)
    csum2 = np.cumsum((X.astype(np.float64) ** 2), axis=0)

    # window 的最后一行 index：seq_length-1 .. n_rows-2
    end = np.arange(seq_length - 1, seq_length - 1 + n_samples)
    start_prev = end - seq_length

    sum_end = csum[end]
    sum2_end = csum2[end]
    sum_prev = np.zeros((n_samples, n_feat), dtype=np.float64)
    sum2_prev = np.zeros((n_samples, n_feat), dtype=np.float64)
    mask = start_prev >= 0
    sum_prev[mask] = csum[start_prev[mask]]
    sum2_prev[mask] = csum2[start_prev[mask]]

    win_sum = sum_end - sum_prev
    win_sum2 = sum2_end - sum2_prev
    mean = win_sum / seq_length
    var = win_sum2 / seq_length - mean ** 2
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    last = X[end]
    return np.concatenate([last, mean.astype(np.float32), std.astype(np.float32)], axis=1).astype(np.float32)


def _seq_to_tabular_nanaware(X_rows: np.ndarray, seq_length: int) -> np.ndarray:
    """与 _seq_to_tabular_fast 相同，但对 NaN 做窗口内忽略（nanmean/nanstd），并保留 NaN。"""
    X = np.asarray(X_rows, dtype=np.float32)
    n_rows, n_feat = X.shape
    seq_length = int(seq_length)
    n_samples = n_rows - seq_length
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    isn = np.isnan(X)
    X0 = np.where(isn, 0.0, X).astype(np.float32)
    cnt = (~isn).astype(np.float32)

    csum = np.cumsum(X0.astype(np.float64), axis=0)
    csum2 = np.cumsum((X0.astype(np.float64) ** 2), axis=0)
    ccnt = np.cumsum(cnt.astype(np.float64), axis=0)

    end = np.arange(seq_length - 1, seq_length - 1 + n_samples)
    start_prev = end - seq_length

    sum_end = csum[end]
    sum2_end = csum2[end]
    cnt_end = ccnt[end]

    sum_prev = np.zeros((n_samples, n_feat), dtype=np.float64)
    sum2_prev = np.zeros((n_samples, n_feat), dtype=np.float64)
    cnt_prev = np.zeros((n_samples, n_feat), dtype=np.float64)
    mask = start_prev >= 0
    sum_prev[mask] = csum[start_prev[mask]]
    sum2_prev[mask] = csum2[start_prev[mask]]
    cnt_prev[mask] = ccnt[start_prev[mask]]

    win_sum = sum_end - sum_prev
    win_sum2 = sum2_end - sum2_prev
    win_cnt = cnt_end - cnt_prev

    mean = np.full((n_samples, n_feat), np.nan, dtype=np.float32)
    std = np.full((n_samples, n_feat), np.nan, dtype=np.float32)
    valid = win_cnt > 0
    mean[valid] = (win_sum[valid] / win_cnt[valid]).astype(np.float32)
    var = np.zeros((n_samples, n_feat), dtype=np.float64)
    var[valid] = win_sum2[valid] / win_cnt[valid] - (win_sum[valid] / win_cnt[valid]) ** 2
    var = np.maximum(var, 0.0)
    std[valid] = np.sqrt(var[valid]).astype(np.float32)

    last = X[end]
    return np.concatenate([last, mean, std], axis=1).astype(np.float32)


def _seq_to_tabular_rolling(
    X_rows: np.ndarray,
    seq_length: int,
    agg: str,
    align: str = "next",
) -> np.ndarray:
    """用滚动和的方式把 (N_rows, F) 转为 tabular（更省内存）。

    对齐口径与 create_sequences 一致：
      - window: [s, s+seq_length)
      - label : y[s+seq_length]
    所以样本数 n_samples = N_rows - seq_length。

    agg:
      - last
      - last_mean
      - last_mean_std
      - last_mean_std_slope
      - last_mean_std_z
      - last_mean_std_slope_z
    """
    X = np.asarray(X_rows, dtype=np.float32)
    n_rows, n_feat = X.shape
    seq_length = int(seq_length)
    align = (align or "next").strip().lower()
    if align not in ("next", "current"):
        raise ValueError("align 仅支持: next / current")
    n_samples = n_rows - seq_length + (1 if align == "current" else 0)
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    agg = (agg or "last_mean_std").strip().lower()
    if agg not in (
        "last",
        "last_mean",
        "last_mean_std",
        "last_mean_std_slope",
        "last_mean_std_z",
        "last_mean_std_slope_z",
    ):
        raise ValueError("tabular_agg 仅支持: last / last_mean / last_mean_std / last_mean_std_slope / last_mean_std_z / last_mean_std_slope_z")
    mult = {
        "last": 1,
        "last_mean": 2,
        "last_mean_std": 3,
        "last_mean_std_slope": 4,
        "last_mean_std_z": 4,
        "last_mean_std_slope_z": 5,
    }[agg]

    out = np.empty((n_samples, n_feat * mult), dtype=np.float32)

    # NaN-aware：忽略缺失值，避免把缺失当 0（XGBoost/LightGBM 可原生处理 NaN）
    win_sum = np.zeros((n_feat,), dtype=np.float64)
    win_sum2 = np.zeros((n_feat,), dtype=np.float64)
    win_cnt = np.zeros((n_feat,), dtype=np.float64)
    init = X[:seq_length]
    m_init = np.isfinite(init)
    init0 = np.where(m_init, init, 0.0).astype(np.float64)
    win_sum[:] = np.sum(init0, axis=0)
    win_sum2[:] = np.sum(init0 ** 2, axis=0)
    win_cnt[:] = np.sum(m_init, axis=0).astype(np.float64)

    for s in range(n_samples):
        end_row = s + seq_length - 1  # align=next/current 都是 window 的最后一行
        last = X[end_row]
        first = X[s]

        if agg == "last":
            out[s, 0:n_feat] = last
        elif agg == "last_mean":
            mean64 = np.full((n_feat,), np.nan, dtype=np.float64)
            valid = win_cnt > 0
            mean64[valid] = win_sum[valid] / win_cnt[valid]
            mean = mean64.astype(np.float32)
            out[s, 0:n_feat] = last
            out[s, n_feat:2 * n_feat] = mean
        elif agg == "last_mean_std":
            mean64 = np.full((n_feat,), np.nan, dtype=np.float64)
            var64 = np.full((n_feat,), np.nan, dtype=np.float64)
            valid = win_cnt > 0
            mean64[valid] = win_sum[valid] / win_cnt[valid]
            var64[valid] = win_sum2[valid] / win_cnt[valid] - mean64[valid] ** 2
            var64 = np.maximum(var64, 0.0)
            out[s, 0:n_feat] = last
            out[s, n_feat:2 * n_feat] = mean64.astype(np.float32)
            out[s, 2 * n_feat:3 * n_feat] = np.sqrt(var64).astype(np.float32)
        elif agg == "last_mean_std_slope":
            mean64 = np.full((n_feat,), np.nan, dtype=np.float64)
            var64 = np.full((n_feat,), np.nan, dtype=np.float64)
            valid = win_cnt > 0
            mean64[valid] = win_sum[valid] / win_cnt[valid]
            var64[valid] = win_sum2[valid] / win_cnt[valid] - mean64[valid] ** 2
            var64 = np.maximum(var64, 0.0)
            out[s, 0:n_feat] = last
            out[s, n_feat:2 * n_feat] = mean64.astype(np.float32)
            out[s, 2 * n_feat:3 * n_feat] = np.sqrt(var64).astype(np.float32)
            out[s, 3 * n_feat:4 * n_feat] = (last - first).astype(np.float32)
        elif agg == "last_mean_std_z":
            mean64 = np.full((n_feat,), np.nan, dtype=np.float64)
            var64 = np.full((n_feat,), np.nan, dtype=np.float64)
            valid = win_cnt > 0
            mean64[valid] = win_sum[valid] / win_cnt[valid]
            var64[valid] = win_sum2[valid] / win_cnt[valid] - mean64[valid] ** 2
            var64 = np.maximum(var64, 0.0)
            std64 = np.sqrt(var64)
            z64 = np.full((n_feat,), np.nan, dtype=np.float64)
            ok = valid & (std64 > 0)
            z64[ok] = (last.astype(np.float64)[ok] - mean64[ok]) / std64[ok]
            out[s, 0:n_feat] = last
            out[s, n_feat:2 * n_feat] = mean64.astype(np.float32)
            out[s, 2 * n_feat:3 * n_feat] = std64.astype(np.float32)
            out[s, 3 * n_feat:4 * n_feat] = z64.astype(np.float32)
        else:  # last_mean_std_slope_z
            mean64 = np.full((n_feat,), np.nan, dtype=np.float64)
            var64 = np.full((n_feat,), np.nan, dtype=np.float64)
            valid = win_cnt > 0
            mean64[valid] = win_sum[valid] / win_cnt[valid]
            var64[valid] = win_sum2[valid] / win_cnt[valid] - mean64[valid] ** 2
            var64 = np.maximum(var64, 0.0)
            std64 = np.sqrt(var64)
            z64 = np.full((n_feat,), np.nan, dtype=np.float64)
            ok = valid & (std64 > 0)
            z64[ok] = (last.astype(np.float64)[ok] - mean64[ok]) / std64[ok]
            out[s, 0:n_feat] = last
            out[s, n_feat:2 * n_feat] = mean64.astype(np.float32)
            out[s, 2 * n_feat:3 * n_feat] = std64.astype(np.float32)
            out[s, 3 * n_feat:4 * n_feat] = (last - first).astype(np.float32)
            out[s, 4 * n_feat:5 * n_feat] = z64.astype(np.float32)

        if s + 1 < n_samples:
            leaving = X[s]
            entering = X[s + seq_length]

            m_leave = np.isfinite(leaving)
            m_enter = np.isfinite(entering)

            if m_leave.any():
                lv = leaving[m_leave].astype(np.float64)
                win_sum[m_leave] -= lv
                win_sum2[m_leave] -= lv ** 2
                win_cnt[m_leave] -= 1.0
            if m_enter.any():
                ev = entering[m_enter].astype(np.float64)
                win_sum[m_enter] += ev
                win_sum2[m_enter] += ev ** 2
                win_cnt[m_enter] += 1.0

    return out

def build_features(
    df,
    horizon,
    label_mode: str = "binary",
    pt_sl: tuple[float, float] = (1.0, 1.0),
    feature_set: str = "ta+hf",
    add_state_features: bool = False,
    state_window: int = 48,
    state_funding_col: str | None = None,
    add_alpha101: bool = False,
    alpha101_list: str | None = None,
    alpha101_rank_window: int = 20,
    alpha101_adv_window: int = 20,
):
    """Build features and labels for a given horizon."""
    feature_set = (feature_set or "ta+hf").strip().lower()
    if feature_set not in ("ta", "hf", "ta+hf"):
        raise ValueError("feature_set 仅支持: ta / hf / ta+hf")

    df = df.copy()
    if "amount" not in df.columns and all(c in df.columns for c in ("close", "volume")):
        df["amount"] = df["close"].astype(float) * df["volume"].astype(float)

    # ta 基线：只保留 OHLCV(+amount)，避免把 hf 特征混进来
    if feature_set == "ta":
        keep = [c for c in ["datetime", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        df = df[keep].copy()
        df = add_ta_indicators(df)
    elif feature_set == "ta+hf":
        df = add_ta_indicators(df)
    else:
        # hf-only: 不添加 TA，只使用 CSV 自带的特征
        pass

    # 连续“状态变量”(state)特征：不做 regime 分桶，让模型自己用连续阈值/交互学习
    if bool(add_state_features):
        if "close" in df.columns:
            w = max(int(state_window), 2)
            close = pd.to_numeric(df["close"], errors="coerce")
            ret1 = close.pct_change()
            df[f"state_trend_{w}"] = close.pct_change(w)
            df[f"state_vol_{w}"] = ret1.rolling(w, min_periods=max(w // 2, 2)).std()
        # funding 类：如果不指定列名，则优先选常见字段
        fc = (str(state_funding_col).strip() if state_funding_col is not None else "")
        cand = [fc] if fc else []
        cand += ["funding_pressure", "funding_rate", "funding_annualized"]
        for c in cand:
            if c and c in df.columns:
                df[f"state_{c}"] = pd.to_numeric(df[c], errors="coerce")
                break

    # Alpha101（单品种近似版）：结构化价格/成交量衍生因子
    if bool(add_alpha101):
        try:
            from data.alpha101_factors import Alpha101Config, compute_alpha101_single
        except Exception as e:
            raise RuntimeError(f"导入 alpha101_factors 失败：{e}") from e

        if alpha101_list:
            alphas = [int(x) for x in str(alpha101_list).split(",") if str(x).strip()]
        else:
            alphas = [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        cfg = Alpha101Config(rank_window=int(alpha101_rank_window), adv_window=int(alpha101_adv_window), scale_window=int(alpha101_rank_window))
        a_df = compute_alpha101_single(df, alphas=alphas, cfg=cfg, prefix="alpha101_")
        df = pd.concat([df, a_df], axis=1)

    # 先计算 future_return，再构造 label；对最后 horizon 行显式置为 NaN 并丢弃，避免把未知标签误当成 0
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    if label_mode == 'triple_barrier':
        tb = triple_barrier_label(df, horizon, pt_sl=pt_sl)
        df['tb_raw'] = tb
        label = (tb == 1).astype('float32')
        label[df['future_return'].isna() | pd.isna(tb)] = np.nan
        df['label'] = label
    else:
        label = (df['future_return'] > 0).astype('float32')
        label[df['future_return'].isna()] = np.nan
        df['label'] = label

    exclude = {'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount',
               'label', 'future_return', 'tb_raw'}
    # 注意：高频特征里普遍存在缺失值，不能用 dropna(feature_cols) 否则样本会被大量丢弃。
    # 这里仅丢弃 label 为 NaN 的行；特征缺失交给后续 nan_to_num + RobustScaler 处理。
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    # 将 bool 转为 0/1，确保全是数值 dtype（FeatureEngineer.select_features 会用 numpy 直接取 values）
    for c in feature_cols:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype("int8")
        elif not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    return df, feature_cols


def create_sequences(df, feature_cols, seq_length=60, align: str = "next"):
    """Create 3D sequences for deep learning models.

    align:
      - next:  用过去 seq_length 根（到 t-1）预测标签 t（默认，最严格因果）
      - current: 用过去 seq_length 根（到 t）预测标签 t（更常见的 bar-close 口径）
    """
    align = (align or "next").strip().lower()
    if align not in ("next", "current"):
        raise ValueError("align 仅支持: next / current")
    data = df[feature_cols].values
    labels = df['label'].values

    # Replace inf/nan（高频特征缺失较多，必须先处理，否则 RobustScaler 直接报错）
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Scale (fit on first 70%)
    train_end = int(len(data) * 0.7)
    scaler = RobustScaler()
    scaler.fit(data[:train_end])
    data = scaler.transform(data)
    data = np.clip(data, -5, 5)

    X, y = [], []
    if align == "next":
        start = seq_length
        for i in range(start, len(data)):
            X.append(data[i - seq_length:i])
            y.append(labels[i])
    else:
        start = seq_length - 1
        for i in range(start, len(data)):
            X.append(data[i - seq_length + 1:i + 1])
            y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y)


def create_tabular(
    df,
    feature_cols,
    seq_length=60,
    tabular_agg: str = "last_mean_std",
    tabular_scale: str = "robust",
    tabular_clip: float | None = 5.0,
    align: str = "next",
):
    """Create 2D tabular features for GBDT models: last + mean + std.

    重要：对齐口径与深度模型一致（用过去 seq_length 预测下一根的 label）。
    """
    X_rows = df[feature_cols].values.astype(np.float32, copy=False)
    y_rows = df["label"].values.astype(np.int64)
    # 先把 inf 规范成 NaN（方便后续 missing 逻辑统一处理）
    X_rows = np.where(np.isfinite(X_rows), X_rows, np.nan).astype(np.float32, copy=False)

    n_rows = len(X_rows)
    seq_length = int(seq_length)
    align = (align or "next").strip().lower()
    if align not in ("next", "current"):
        raise ValueError("align 仅支持: next / current")
    n_samples = n_rows - seq_length + (1 if align == "current" else 0)
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    # 划分按样本比例，映射回行做 scaler fit，避免泄露
    train_end_s = int(n_samples * 0.7)
    # 训练样本覆盖到的最后一行 index = (train_end_s-1)+seq_length-1 = train_end_s+seq_length-2
    fit_end_row = max(train_end_s + seq_length - 2, 0)
    tabular_scale = (tabular_scale or "robust").strip().lower()
    if tabular_scale not in ("robust", "none"):
        raise ValueError("tabular_scale 仅支持: robust / none")

    if tabular_scale == "robust":
        # RobustScaler 不支持 NaN：这里用 0 填充缺失值（也可未来扩展 median/ffill）
        X_rows = np.nan_to_num(X_rows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        scaler = RobustScaler()
        scaler.fit(X_rows[:fit_end_row + 1])
        X_rows = scaler.transform(X_rows).astype(np.float32)

    if tabular_clip is not None:
        clip_v = float(tabular_clip)
        if clip_v > 0:
            X_rows = np.clip(X_rows, -clip_v, clip_v)

    # 用 rolling 版本生成 tabular，避免 _seq_to_tabular_fast 的 cumsum/cumsum2 大矩阵引发 OOM
    X_tab = _seq_to_tabular_rolling(X_rows, seq_length=seq_length, agg=tabular_agg, align=align)
    y = y_rows[seq_length:] if align == "next" else y_rows[seq_length - 1:]
    return X_tab, y


def _parse_json_dict(s: str | None, arg_name: str) -> dict | None:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"{arg_name} 解析失败，请提供 JSON 对象字符串。例如: '{{\"max_depth\":4,\"min_child_weight\":10}}'。原错误: {e}") from e
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ValueError(f"{arg_name} 需要是 JSON 对象(dict)，而不是 {type(obj)}")
    return obj


def _compute_regime_row_labels(
    df_feat: pd.DataFrame,
    mode: str,
    window: int,
    bins: int,
    train_end_idx: int,
    regime_col: str | None = None,
) -> np.ndarray:
    """计算每一行(bar)的 regime 标签（row-level），后续再对齐到 sample-level。

    mode:
      - vol_quantile:  按 rolling 波动率分位数分桶（0..bins-1）
      - trend_sign:    按 rolling 收益（window）正负分桶（0/1）
      - funding_sign:  按资金费率(或指定列)正负分桶（0/1）
    返回:
      shape (n_rows,), int，缺失/不可用为 -1
    """
    mode = (mode or "none").strip().lower()
    window = int(window)
    bins = int(bins)
    train_end_idx = int(train_end_idx)
    n = len(df_feat)
    out = np.full((n,), -1, dtype=np.int16)

    if mode in ("none", ""):
        return out

    if mode == "vol_quantile":
        if "close" not in df_feat.columns:
            raise ValueError("regime=vol_quantile 需要 df_feat 包含 close 列")
        ret1 = pd.to_numeric(df_feat["close"], errors="coerce").pct_change()
        vol = ret1.rolling(window, min_periods=max(window // 2, 2)).std()
        vol_tr = vol.iloc[:train_end_idx].dropna()
        if len(vol_tr) < 100:
            raise ValueError("regime=vol_quantile: 训练段有效样本太少，无法计算分位点")
        qs = np.linspace(0.0, 1.0, bins + 1)
        edges = vol_tr.quantile(qs).to_numpy()
        # 处理分位点重复（极端情况下 vol 很平）
        edges = np.unique(edges)
        if len(edges) - 1 < bins:
            # 退化：用中位数二分
            med = float(vol_tr.median())
            edges = np.array([float("-inf"), med, float("inf")], dtype=float)
            bins = 2
        # digitize: 用内部边界，把每个值映射到 0..bins-1
        cut = edges[1:-1]
        vv = vol.to_numpy()
        ok = np.isfinite(vv)
        out[ok] = np.digitize(vv[ok], cut, right=False).astype(np.int16)
        return out

    if mode == "trend_sign":
        if "close" not in df_feat.columns:
            raise ValueError("regime=trend_sign 需要 df_feat 包含 close 列")
        mom = pd.to_numeric(df_feat["close"], errors="coerce").pct_change(window)
        vv = mom.to_numpy()
        ok = np.isfinite(vv)
        out[ok] = (vv[ok] >= 0).astype(np.int16)
        return out

    if mode == "funding_sign":
        col = (regime_col or "").strip()
        if not col:
            for c in ("funding_pressure", "funding_rate", "funding_annualized"):
                if c in df_feat.columns:
                    col = c
                    break
        if not col or col not in df_feat.columns:
            raise ValueError("regime=funding_sign 需要 funding_pressure/funding_rate/funding_annualized 之一，或用 --regime-col 指定")
        vv = pd.to_numeric(df_feat[col], errors="coerce").to_numpy()
        ok = np.isfinite(vv)
        out[ok] = (vv[ok] >= 0).astype(np.int16)
        return out

    raise ValueError("regime_mode 仅支持: none / vol_quantile / trend_sign / funding_sign")


def _align_row_labels_to_samples(
    row_labels: np.ndarray,
    n_rows: int,
    seq_length: int,
    align: str,
) -> np.ndarray:
    """把 row-level 标签对齐到 sample-level（每个 sample 取 window 最后一行的标签）。"""
    align = (align or "next").strip().lower()
    if align not in ("next", "current"):
        raise ValueError("align 仅支持: next / current")
    n_rows = int(n_rows)
    seq_length = int(seq_length)
    n_samples = n_rows - seq_length + (1 if align == "current" else 0)
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")
    end_rows = np.arange(seq_length - 1, seq_length - 1 + n_samples)
    return np.asarray(row_labels, dtype=np.int16)[end_rows]


def _regime_cv_separate_train(
    trainer: ModelTrainer,
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    regime_s: np.ndarray,
    cv_cfg: dict,
    model_params: dict | None,
    fit_params: dict | None,
    min_train: int,
    min_val: int,
    min_test: int,
) -> dict:
    """按 regime 分别训练（每折内、每个regime单独训练/评估），返回加权平均与各regime统计。"""
    splitter = TimeSeriesSplitter(
        n_splits=int(cv_cfg["n_splits"]),
        gap=int(cv_cfg["gap"]),
        val_ratio=float(cv_cfg["val_ratio"]),
        test_ratio=float(cv_cfg["test_ratio"]),
    )
    splits = splitter.split(len(X))
    regimes = sorted([int(r) for r in np.unique(regime_s) if int(r) >= 0])

    per_r = {r: {"val_auc": [], "test_auc": [], "test_acc": [], "test_n": []} for r in regimes}

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        for r in regimes:
            m_tr = regime_s[train_idx] == r
            m_va = regime_s[val_idx] == r
            m_te = regime_s[test_idx] == r
            tr = train_idx[m_tr]
            va = val_idx[m_va]
            te = test_idx[m_te]
            if len(tr) < int(min_train) or len(va) < int(min_val) or len(te) < int(min_test):
                continue

            model, metrics = trainer.train_model(
                model_type,
                X[tr], y[tr],
                X[va], y[va],
                model_params=model_params,
                fit_params=fit_params,
            )
            probs = model.predict_proba(X[te])
            y_te = y[te]
            test_auc = roc_auc_score(y_te, probs) if len(set(y_te)) > 1 else 0.5
            test_acc = accuracy_score(y_te, (probs >= 0.5).astype(int))

            per_r[r]["val_auc"].append(float(metrics.get("val_auc", 0.0)))
            per_r[r]["test_auc"].append(float(test_auc))
            per_r[r]["test_acc"].append(float(test_acc))
            per_r[r]["test_n"].append(int(len(te)))

    out = {"regimes": regimes, "per_regime": {}}
    # 用每个 fold 的 test_n 做全局加权：sum(auc * n) / sum(n)
    total_n = 0
    weighted_auc_sum = 0.0
    weighted_acc_sum = 0.0
    for r in regimes:
        ns = per_r[r]["test_n"]
        aucs = per_r[r]["test_auc"]
        accs = per_r[r]["test_acc"]
        n_sum = int(np.sum(ns)) if ns else 0
        if n_sum > 0 and aucs and accs:
            w = np.asarray(ns, dtype=np.float64)
            auc_arr = np.asarray(aucs, dtype=np.float64)
            acc_arr = np.asarray(accs, dtype=np.float64)
            weighted_auc_sum += float(np.sum(w * auc_arr))
            weighted_acc_sum += float(np.sum(w * acc_arr))
            total_n += int(np.sum(w))

        out["per_regime"][f"r{r}"] = {
            "folds_used": int(len(aucs)),
            "test_n": int(n_sum),
            "val_auc_mean": float(np.mean(per_r[r]["val_auc"])) if per_r[r]["val_auc"] else 0.0,
            "val_auc_std": float(np.std(per_r[r]["val_auc"])) if per_r[r]["val_auc"] else 0.0,
            "test_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
            "test_auc_std": float(np.std(aucs)) if aucs else 0.0,
            "test_acc_mean": float(np.mean(accs)) if accs else 0.0,
        }

    out["weighted_test_auc"] = float(weighted_auc_sum / max(total_n, 1))
    out["weighted_test_acc"] = float(weighted_acc_sum / max(total_n, 1))
    out["total_test_n"] = int(total_n)
    return out


def _trade_stats_from_probs_and_future_return(
    probs: np.ndarray,
    future_return: np.ndarray,
    mode: str,
    long_thr: float,
    short_thr: float,
    cost_bps: float,
) -> dict:
    """最简交易统计（允许重叠持仓，按每个样本独立计算）。"""
    probs = np.asarray(probs, dtype=np.float32)
    fr = np.asarray(future_return, dtype=np.float32)
    mode = (mode or "long").strip().lower()
    if mode not in ("long", "long_short"):
        raise ValueError("trade_mode 仅支持: long / long_short")
    cost = float(cost_bps) / 10000.0

    long_m = probs >= float(long_thr)
    short_m = (probs <= float(short_thr)) if mode == "long_short" else np.zeros_like(long_m, dtype=bool)
    trade_m = long_m | short_m
    if not np.any(trade_m):
        return {"trade_n": 0, "pnl_sum": 0.0, "pnl_mean": 0.0, "pnl_std": 0.0, "hit_sum": 0, "hit_rate": 0.0}

    pnl = np.zeros_like(fr, dtype=np.float32)
    pnl[long_m] = fr[long_m]
    pnl[short_m] = -fr[short_m]
    pnl = pnl[trade_m] - cost
    return {
        "trade_n": int(len(pnl)),
        "pnl_sum": float(np.sum(pnl)),
        "pnl_mean": float(np.mean(pnl)),
        "pnl_std": float(np.std(pnl)),
        "hit_sum": int(np.sum(pnl > 0)),
        "hit_rate": float(np.mean(pnl > 0)),
    }


def _regime_cv_single_train_report(
    trainer: ModelTrainer,
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    regime_s: np.ndarray,
    future_return_s: np.ndarray,
    cv_cfg: dict,
    model_params: dict | None,
    fit_params: dict | None,
    trade_eval: bool = False,
    trade_mode: str = "long",
    trade_long_thr: float = 0.55,
    trade_short_thr: float = 0.45,
    trade_cost_bps: float = 0.0,
    trade_step: int = 1,
) -> dict:
    """单模型训练，但按 regime 输出每折 test AUC 与覆盖率（coverage）。"""
    splitter = TimeSeriesSplitter(
        n_splits=int(cv_cfg["n_splits"]),
        gap=int(cv_cfg["gap"]),
        val_ratio=float(cv_cfg["val_ratio"]),
        test_ratio=float(cv_cfg["test_ratio"]),
    )
    splits = splitter.split(len(X))
    # 只统计有效 regime（>=0）。缺失/不可用为 -1，会从 regime 覆盖率统计中剔除。
    regimes = sorted([int(r) for r in np.unique(regime_s) if int(r) >= 0])

    trade_step = max(int(trade_step), 1)
    overall = {"val_auc": [], "test_auc": [], "test_acc": [], "test_n": [], "valid_test_n": []}
    per_r = {r: {"test_auc": [], "test_acc": [], "test_n": [], "pos_sum": 0} for r in regimes}
    per_r_trade = (
        {r: {"sample_n": 0, "trade_n": 0, "pnl_sum": 0.0, "hit_sum": 0} for r in regimes}
        if bool(trade_eval)
        else None
    )

    for _, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        model, metrics = trainer.train_model(
            model_type,
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            model_params=model_params,
            fit_params=fit_params,
        )

        probs_all = model.predict_proba(X[test_idx])
        y_all = y[test_idx]
        fr_all = np.asarray(future_return_s, dtype=np.float32)[test_idx]
        test_auc_all = roc_auc_score(y_all, probs_all) if len(set(y_all)) > 1 else 0.5
        test_acc_all = accuracy_score(y_all, (probs_all >= 0.5).astype(int))

        overall["val_auc"].append(float(metrics.get("val_auc", 0.0)))
        overall["test_auc"].append(float(test_auc_all))
        overall["test_acc"].append(float(test_acc_all))
        overall["test_n"].append(int(len(test_idx)))

        r_test = regime_s[test_idx]
        valid_mask = r_test >= 0
        overall["valid_test_n"].append(int(np.sum(valid_mask)))

        if trade_eval:
            # 交易评估：可通过 trade_step 做“下采样”来减少重叠持仓带来的虚高
            probs_trade = probs_all[::trade_step]
            fr_trade = fr_all[::trade_step]
            r_trade = r_test[::trade_step]

        for r in regimes:
            m = r_test == r
            if not np.any(m):
                continue
            probs_r = probs_all[m]
            y_r = y_all[m]
            auc_r = roc_auc_score(y_r, probs_r) if len(set(y_r)) > 1 else 0.5
            acc_r = accuracy_score(y_r, (probs_r >= 0.5).astype(int))
            per_r[r]["test_auc"].append(float(auc_r))
            per_r[r]["test_acc"].append(float(acc_r))
            per_r[r]["test_n"].append(int(np.sum(m)))
            per_r[r]["pos_sum"] += int(np.sum(y_r))

            if trade_eval:
                mt = r_trade == r
                if np.any(mt):
                    per_r_trade[r]["sample_n"] += int(np.sum(mt))
                    st = _trade_stats_from_probs_and_future_return(
                        probs_trade[mt],
                        fr_trade[mt],
                        mode=str(trade_mode),
                        long_thr=float(trade_long_thr),
                        short_thr=float(trade_short_thr),
                        cost_bps=float(trade_cost_bps),
                    )
                    per_r_trade[r]["trade_n"] += int(st["trade_n"])
                    per_r_trade[r]["pnl_sum"] += float(st["pnl_sum"])
                    per_r_trade[r]["hit_sum"] += int(st["hit_sum"])

    out = {
        "folds": int(len(splits)),
        "regimes": regimes,
        "overall": {
            "val_auc_mean": float(np.mean(overall["val_auc"])) if overall["val_auc"] else 0.0,
            "val_auc_std": float(np.std(overall["val_auc"])) if overall["val_auc"] else 0.0,
            "test_auc_mean": float(np.mean(overall["test_auc"])) if overall["test_auc"] else 0.0,
            "test_auc_std": float(np.std(overall["test_auc"])) if overall["test_auc"] else 0.0,
            "test_acc_mean": float(np.mean(overall["test_acc"])) if overall["test_acc"] else 0.0,
            "total_test_n": int(np.sum(overall["test_n"])) if overall["test_n"] else 0,
            "total_valid_test_n": int(np.sum(overall["valid_test_n"])) if overall["valid_test_n"] else 0,
            "dropped_test_n": int(np.sum(overall["test_n"])) - int(np.sum(overall["valid_test_n"])) if overall["test_n"] else 0,
        },
        "per_regime": {},
    }

    total_valid = int(out["overall"]["total_valid_test_n"])
    for r in regimes:
        ns = per_r[r]["test_n"]
        aucs = per_r[r]["test_auc"]
        accs = per_r[r]["test_acc"]
        n_sum = int(np.sum(ns)) if ns else 0
        pos_sum = int(per_r[r]["pos_sum"])
        out["per_regime"][f"r{r}"] = {
            "folds_used": int(len(aucs)),
            "test_n": int(n_sum),
            "coverage": float(n_sum / max(total_valid, 1)),
            "test_up_ratio": float(pos_sum / max(n_sum, 1)),
            "test_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
            "test_auc_std": float(np.std(aucs)) if aucs else 0.0,
            "test_acc_mean": float(np.mean(accs)) if accs else 0.0,
        }
        if trade_eval:
            sn = int(per_r_trade[r]["sample_n"])
            tn = int(per_r_trade[r]["trade_n"])
            out["per_regime"][f"r{r}"]["trade_sample_n"] = sn
            out["per_regime"][f"r{r}"]["trade_sample_coverage"] = float(sn / max(n_sum, 1))
            out["per_regime"][f"r{r}"]["trade_n"] = tn
            out["per_regime"][f"r{r}"]["trade_coverage"] = float(tn / max(sn, 1))
            out["per_regime"][f"r{r}"]["trade_pnl_mean"] = float(per_r_trade[r]["pnl_sum"] / max(tn, 1))
            out["per_regime"][f"r{r}"]["trade_hit_rate"] = float(per_r_trade[r]["hit_sum"] / max(tn, 1))

    return out


def _regime_cv_single_train_report_two(
    trainer: ModelTrainer,
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    regime1_s: np.ndarray,
    regime2_s: np.ndarray | None,
    future_return_s: np.ndarray,
    cv_cfg: dict,
    model_params: dict | None,
    fit_params: dict | None,
    trade_eval: bool = False,
    trade_mode: str = "long",
    trade_long_thr: float = 0.55,
    trade_short_thr: float = 0.45,
    trade_cost_bps: float = 0.0,
    trade_step: int = 1,
) -> dict:
    """单模型训练；同时按两个 regime 输出 test AUC/覆盖率，并输出交叉(AND)分桶结果。"""
    # 如果没提供第二个regime，退化为单regime报表
    if regime2_s is None:
        out1 = _regime_cv_single_train_report(
            trainer,
            model_type=model_type,
            X=X,
            y=y,
            regime_s=regime1_s,
            future_return_s=future_return_s,
            cv_cfg=cv_cfg,
            model_params=model_params,
            fit_params=fit_params,
            trade_eval=trade_eval,
            trade_mode=trade_mode,
            trade_long_thr=trade_long_thr,
            trade_short_thr=trade_short_thr,
            trade_cost_bps=trade_cost_bps,
            trade_step=trade_step,
        )
        return {"regime1": out1, "regime2": None, "intersection": None}

    trade_step = max(int(trade_step), 1)

    splitter = TimeSeriesSplitter(
        n_splits=int(cv_cfg["n_splits"]),
        gap=int(cv_cfg["gap"]),
        val_ratio=float(cv_cfg["val_ratio"]),
        test_ratio=float(cv_cfg["test_ratio"]),
    )
    splits = splitter.split(len(X))

    r1_vals = sorted([int(r) for r in np.unique(regime1_s) if int(r) >= 0])
    r2_vals = sorted([int(r) for r in np.unique(regime2_s) if int(r) >= 0])
    pairs = [(a, b) for a in r1_vals for b in r2_vals]

    overall = {"val_auc": [], "test_auc": [], "test_acc": [], "test_n": []}
    valid1_n, valid2_n, valid12_n = [], [], []

    per_r1 = {a: {"test_auc": [], "test_acc": [], "test_n": [], "pos_sum": 0} for a in r1_vals}
    per_r2 = {b: {"test_auc": [], "test_acc": [], "test_n": [], "pos_sum": 0} for b in r2_vals}
    per_pair = {(a, b): {"test_auc": [], "test_acc": [], "test_n": [], "pos_sum": 0} for (a, b) in pairs}

    per_r1_trade = {a: {"sample_n": 0, "trade_n": 0, "pnl_sum": 0.0, "hit_sum": 0} for a in r1_vals} if trade_eval else None
    per_r2_trade = {b: {"sample_n": 0, "trade_n": 0, "pnl_sum": 0.0, "hit_sum": 0} for b in r2_vals} if trade_eval else None
    per_pair_trade = {(a, b): {"sample_n": 0, "trade_n": 0, "pnl_sum": 0.0, "hit_sum": 0} for (a, b) in pairs} if trade_eval else None

    for _, (train_idx, val_idx, test_idx) in enumerate(splits):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)

        model, metrics = trainer.train_model(
            model_type,
            X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            model_params=model_params,
            fit_params=fit_params,
        )

        probs_all = model.predict_proba(X[test_idx])
        y_all = y[test_idx]
        fr_all = np.asarray(future_return_s, dtype=np.float32)[test_idx]
        r1_test = regime1_s[test_idx]
        r2_test = regime2_s[test_idx]

        overall["val_auc"].append(float(metrics.get("val_auc", 0.0)))
        overall["test_auc"].append(float(roc_auc_score(y_all, probs_all) if len(set(y_all)) > 1 else 0.5))
        overall["test_acc"].append(float(accuracy_score(y_all, (probs_all >= 0.5).astype(int))))
        overall["test_n"].append(int(len(test_idx)))

        valid1_n.append(int(np.sum(r1_test >= 0)))
        valid2_n.append(int(np.sum(r2_test >= 0)))
        valid12_n.append(int(np.sum((r1_test >= 0) & (r2_test >= 0))))

        if trade_eval:
            probs_tr = probs_all[::trade_step]
            fr_tr = fr_all[::trade_step]
            r1_tr = r1_test[::trade_step]
            r2_tr = r2_test[::trade_step]

        for a in r1_vals:
            m = r1_test == a
            if np.any(m):
                y_m = y_all[m]
                p_m = probs_all[m]
                per_r1[a]["test_auc"].append(float(roc_auc_score(y_m, p_m) if len(set(y_m)) > 1 else 0.5))
                per_r1[a]["test_acc"].append(float(accuracy_score(y_m, (p_m >= 0.5).astype(int))))
                per_r1[a]["test_n"].append(int(np.sum(m)))
                per_r1[a]["pos_sum"] += int(np.sum(y_m))
            if trade_eval:
                mt = r1_tr == a
                if np.any(mt):
                    per_r1_trade[a]["sample_n"] += int(np.sum(mt))
                    st = _trade_stats_from_probs_and_future_return(
                        probs_tr[mt],
                        fr_tr[mt],
                        mode=str(trade_mode),
                        long_thr=float(trade_long_thr),
                        short_thr=float(trade_short_thr),
                        cost_bps=float(trade_cost_bps),
                    )
                    per_r1_trade[a]["trade_n"] += int(st["trade_n"])
                    per_r1_trade[a]["pnl_sum"] += float(st["pnl_sum"])
                    per_r1_trade[a]["hit_sum"] += int(st["hit_sum"])

        for b in r2_vals:
            m = r2_test == b
            if np.any(m):
                y_m = y_all[m]
                p_m = probs_all[m]
                per_r2[b]["test_auc"].append(float(roc_auc_score(y_m, p_m) if len(set(y_m)) > 1 else 0.5))
                per_r2[b]["test_acc"].append(float(accuracy_score(y_m, (p_m >= 0.5).astype(int))))
                per_r2[b]["test_n"].append(int(np.sum(m)))
                per_r2[b]["pos_sum"] += int(np.sum(y_m))
            if trade_eval:
                mt = r2_tr == b
                if np.any(mt):
                    per_r2_trade[b]["sample_n"] += int(np.sum(mt))
                    st = _trade_stats_from_probs_and_future_return(
                        probs_tr[mt],
                        fr_tr[mt],
                        mode=str(trade_mode),
                        long_thr=float(trade_long_thr),
                        short_thr=float(trade_short_thr),
                        cost_bps=float(trade_cost_bps),
                    )
                    per_r2_trade[b]["trade_n"] += int(st["trade_n"])
                    per_r2_trade[b]["pnl_sum"] += float(st["pnl_sum"])
                    per_r2_trade[b]["hit_sum"] += int(st["hit_sum"])

        for (a, b) in pairs:
            m = (r1_test == a) & (r2_test == b)
            if np.any(m):
                y_m = y_all[m]
                p_m = probs_all[m]
                per_pair[(a, b)]["test_auc"].append(float(roc_auc_score(y_m, p_m) if len(set(y_m)) > 1 else 0.5))
                per_pair[(a, b)]["test_acc"].append(float(accuracy_score(y_m, (p_m >= 0.5).astype(int))))
                per_pair[(a, b)]["test_n"].append(int(np.sum(m)))
                per_pair[(a, b)]["pos_sum"] += int(np.sum(y_m))
            if trade_eval:
                mt = (r1_tr == a) & (r2_tr == b)
                if np.any(mt):
                    per_pair_trade[(a, b)]["sample_n"] += int(np.sum(mt))
                    st = _trade_stats_from_probs_and_future_return(
                        probs_tr[mt],
                        fr_tr[mt],
                        mode=str(trade_mode),
                        long_thr=float(trade_long_thr),
                        short_thr=float(trade_short_thr),
                        cost_bps=float(trade_cost_bps),
                    )
                    per_pair_trade[(a, b)]["trade_n"] += int(st["trade_n"])
                    per_pair_trade[(a, b)]["pnl_sum"] += float(st["pnl_sum"])
                    per_pair_trade[(a, b)]["hit_sum"] += int(st["hit_sum"])

    overall_summary = {
        "val_auc_mean": float(np.mean(overall["val_auc"])) if overall["val_auc"] else 0.0,
        "val_auc_std": float(np.std(overall["val_auc"])) if overall["val_auc"] else 0.0,
        "test_auc_mean": float(np.mean(overall["test_auc"])) if overall["test_auc"] else 0.0,
        "test_auc_std": float(np.std(overall["test_auc"])) if overall["test_auc"] else 0.0,
        "test_acc_mean": float(np.mean(overall["test_acc"])) if overall["test_acc"] else 0.0,
        "total_test_n": int(np.sum(overall["test_n"])) if overall["test_n"] else 0,
    }

    out1 = {"folds": int(len(splits)), "regimes": r1_vals, "overall": overall_summary, "per_regime": {}}
    out2 = {"folds": int(len(splits)), "regimes": r2_vals, "overall": overall_summary, "per_regime": {}}
    inter = {"pairs": [f"r{a}&r{b}" for (a, b) in pairs], "per_pair": {}}

    total_valid1 = int(np.sum(valid1_n)) if valid1_n else 0
    total_valid2 = int(np.sum(valid2_n)) if valid2_n else 0
    total_valid12 = int(np.sum(valid12_n)) if valid12_n else 0

    for a in r1_vals:
        ns = per_r1[a]["test_n"]
        aucs = per_r1[a]["test_auc"]
        accs = per_r1[a]["test_acc"]
        n_sum = int(np.sum(ns)) if ns else 0
        pos_sum = int(per_r1[a]["pos_sum"])
        rec = {
            "folds_used": int(len(aucs)),
            "test_n": int(n_sum),
            "coverage": float(n_sum / max(total_valid1, 1)),
            "test_up_ratio": float(pos_sum / max(n_sum, 1)),
            "test_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
            "test_auc_std": float(np.std(aucs)) if aucs else 0.0,
            "test_acc_mean": float(np.mean(accs)) if accs else 0.0,
        }
        if trade_eval:
            sn = int(per_r1_trade[a]["sample_n"])
            tn = int(per_r1_trade[a]["trade_n"])
            rec["trade_sample_n"] = sn
            rec["trade_sample_coverage"] = float(sn / max(n_sum, 1))
            rec["trade_n"] = tn
            rec["trade_coverage"] = float(tn / max(sn, 1))
            rec["trade_pnl_mean"] = float(per_r1_trade[a]["pnl_sum"] / max(tn, 1))
            rec["trade_hit_rate"] = float(per_r1_trade[a]["hit_sum"] / max(tn, 1))
        out1["per_regime"][f"r{a}"] = rec

    for b in r2_vals:
        ns = per_r2[b]["test_n"]
        aucs = per_r2[b]["test_auc"]
        accs = per_r2[b]["test_acc"]
        n_sum = int(np.sum(ns)) if ns else 0
        pos_sum = int(per_r2[b]["pos_sum"])
        rec = {
            "folds_used": int(len(aucs)),
            "test_n": int(n_sum),
            "coverage": float(n_sum / max(total_valid2, 1)),
            "test_up_ratio": float(pos_sum / max(n_sum, 1)),
            "test_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
            "test_auc_std": float(np.std(aucs)) if aucs else 0.0,
            "test_acc_mean": float(np.mean(accs)) if accs else 0.0,
        }
        if trade_eval:
            sn = int(per_r2_trade[b]["sample_n"])
            tn = int(per_r2_trade[b]["trade_n"])
            rec["trade_sample_n"] = sn
            rec["trade_sample_coverage"] = float(sn / max(n_sum, 1))
            rec["trade_n"] = tn
            rec["trade_coverage"] = float(tn / max(sn, 1))
            rec["trade_pnl_mean"] = float(per_r2_trade[b]["pnl_sum"] / max(tn, 1))
            rec["trade_hit_rate"] = float(per_r2_trade[b]["hit_sum"] / max(tn, 1))
        out2["per_regime"][f"r{b}"] = rec

    for (a, b) in pairs:
        rec0 = per_pair[(a, b)]
        ns = rec0["test_n"]
        aucs = rec0["test_auc"]
        accs = rec0["test_acc"]
        n_sum = int(np.sum(ns)) if ns else 0
        pos_sum = int(rec0["pos_sum"])
        key = f"r{a}&r{b}"
        rec = {
            "folds_used": int(len(aucs)),
            "test_n": int(n_sum),
            "coverage": float(n_sum / max(total_valid12, 1)),
            "test_up_ratio": float(pos_sum / max(n_sum, 1)),
            "test_auc_mean": float(np.mean(aucs)) if aucs else 0.0,
            "test_auc_std": float(np.std(aucs)) if aucs else 0.0,
            "test_acc_mean": float(np.mean(accs)) if accs else 0.0,
        }
        if trade_eval:
            sn = int(per_pair_trade[(a, b)]["sample_n"])
            tn = int(per_pair_trade[(a, b)]["trade_n"])
            rec["trade_sample_n"] = sn
            rec["trade_sample_coverage"] = float(sn / max(n_sum, 1))
            rec["trade_n"] = tn
            rec["trade_coverage"] = float(tn / max(sn, 1))
            rec["trade_pnl_mean"] = float(per_pair_trade[(a, b)]["pnl_sum"] / max(tn, 1))
            rec["trade_hit_rate"] = float(per_pair_trade[(a, b)]["hit_sum"] / max(tn, 1))
        inter["per_pair"][key] = rec

    return {"regime1": out1, "regime2": out2, "intersection": inter}


class RollingWindowDataset(Dataset):
    """滚动窗口序列数据集：window = [s, s+seq_length)，label = y[s+seq_length]。

    对齐方式与本项目 create_sequences 保持一致（用过去 seq_length 根预测“下一根”的 label）。
    """

    def __init__(
        self,
        X_rows: np.ndarray,
        y_rows: np.ndarray,
        seq_length: int,
        start: int,
        end: int,
        align: str = "next",
    ):
        self.X_rows = np.asarray(X_rows, dtype=np.float32)
        self.y_rows = np.asarray(y_rows, dtype=np.float32)
        self.seq_length = int(seq_length)
        self.start = int(start)
        self.end = int(end)
        self.align = (align or "next").strip().lower()
        if self.align not in ("next", "current"):
            raise ValueError("RollingWindowDataset: align 仅支持 next/current")

        n_rows = len(self.X_rows)
        n_samples = n_rows - self.seq_length + (1 if self.align == "current" else 0)
        if n_samples <= 0:
            raise ValueError("seq_length 太大，无法生成样本")
        if not (0 <= self.start <= self.end <= n_samples):
            raise ValueError("RollingWindowDataset: start/end 超出范围")

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx: int):
        s = self.start + int(idx)
        x = self.X_rows[s:s + self.seq_length]
        label_idx = s + (self.seq_length if self.align == "next" else self.seq_length - 1)
        y = self.y_rows[label_idx]
        return torch.from_numpy(x), torch.tensor(float(y), dtype=torch.float32)


def _select_features_if_needed(
    df_feat: pd.DataFrame,
    feature_cols: list[str],
    no_select: bool,
    select_method: str,
    max_features: int,
) -> list[str]:
    if no_select:
        return feature_cols
    fe = FeatureEngineer()
    fe.feature_names = feature_cols
    train_end = int(len(df_feat) * 0.7)
    selected = fe.select_features(df_feat, method=select_method, max_features=int(max_features), train_end_idx=train_end)
    return selected


def _prune_correlated_features(
    df_feat: pd.DataFrame,
    cols: list[str],
    threshold: float,
    method: str = "pearson",
    train_end_idx: int | None = None,
) -> list[str]:
    """在训练段做共线性剪枝：按 cols 的顺序（通常已按重要性排序）贪心保留。"""
    thr = float(threshold)
    if not (0.0 < thr < 1.0):
        raise ValueError("threshold 需在 (0,1) 内")
    method = (method or "pearson").strip().lower()
    if method not in ("pearson", "spearman"):
        raise ValueError("corr method 仅支持 pearson/spearman")

    if train_end_idx is None:
        train_end_idx = int(len(df_feat) * 0.7)

    sub = df_feat.iloc[:int(train_end_idx)][cols].copy()
    sub = sub.apply(pd.to_numeric, errors="coerce")
    med = sub.median(numeric_only=True)
    sub = sub.fillna(med)

    corr = sub.corr(method=method).abs().to_numpy()
    keep_idx: list[int] = []
    for i in range(len(cols)):
        if not keep_idx:
            keep_idx.append(i)
            continue
        mx = float(np.max(corr[i, keep_idx]))
        if mx < thr:
            keep_idx.append(i)
    return [cols[i] for i in keep_idx]


def _train_deep_with_loaders(
    trainer: ModelTrainer,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_train_samples: np.ndarray,
    input_size: int,
    seq_length: int,
    config: dict | None = None,
):
    """复用 ModelTrainer 的核心训练逻辑，但避免一次性构造完整 3D numpy 数组。"""
    if config is None:
        config = TRAINING_CONFIG

    model = trainer._create_model(model_type, input_size=input_size, seq_length=seq_length)

    pos_weight = trainer._compute_pos_weight(np.asarray(y_train_samples, dtype=np.int64))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config["scheduler_factor"],
        patience=config["scheduler_patience"],
    )
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_min_delta"],
        mode="max",
    )

    best_metrics = {}
    for epoch in range(config["max_epochs"]):
        train_loss, train_auc = trainer.train_epoch(
            model, train_loader, criterion, optimizer, config["gradient_clip_norm"]
        )
        val_loss, val_auc, val_acc, _, _ = trainer.evaluate(model, val_loader, criterion)
        scheduler.step(val_auc)

        early_stopping(val_auc, model)
        if early_stopping.early_stop:
            break

        best_metrics = {
            "val_auc": float(early_stopping.best_score) if early_stopping.best_score is not None else float(val_auc),
            "val_acc": float(val_acc),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "epoch": int(epoch + 1),
        }

    if early_stopping.best_model_state:
        model.load_state_dict(early_stopping.best_model_state)
    return model, best_metrics


def run_experiment(df, horizon, model_types, seq_length=60, label_mode='binary', pt_sl=(1.0, 1.0),
                   feature_set: str = "ta+hf", tabular: bool = False,
                   rolling_dataset: bool = False,
                   add_state_features: bool = False,
                   state_window: int = 48,
                   state_funding_col: str | None = None,
                   add_alpha101: bool = False,
                   alpha101_list: str | None = None,
                   alpha101_rank_window: int = 20,
                   alpha101_adv_window: int = 20,
                   no_select: bool = False,
                   select_method: str = "mutual_info",
                   max_features: int = MAX_FEATURES,
                   batch_size: int | None = None,
                   prune_corr: float | None = None,
                   corr_method: str = "pearson",
                   tabular_agg: str = "last_mean_std",
                   tabular_scale: str = "robust",
                   tabular_clip: float | None = 5.0,
                   align: str = "next",
                   trade_eval: bool = False,
                   trade_mode: str = "long",
                   trade_long_thr: float = 0.55,
                   trade_short_thr: float = 0.45,
                   trade_cost_bps: float = 0.0,
                   trade_step: int | None = None,
                   xgb_params: dict | None = None,
                   lgbm_params: dict | None = None,
                   xgb_fit_params: dict | None = None,
                   lgbm_fit_params: dict | None = None,
                   split_gap: int = 0,
                   cv: bool = False,
                   cv_splits: int | None = None,
                   cv_gap: int | None = None,
                   cv_val_ratio: float | None = None,
                   cv_test_ratio: float | None = None,
                   regime_mode: str = "none",
                   regime_window: int = 48,
                   regime_bins: int = 2,
                   regime_col: str | None = None,
                   regime2_mode: str = "none",
                   regime2_window: int = 48,
                   regime2_bins: int = 2,
                   regime2_col: str | None = None,
                   regime_train: str = "single",
                   regime_min_train: int = 2000,
                   regime_min_val: int = 500,
                   regime_min_test: int = 500):
    """Run single horizon experiment with multiple models."""
    df_feat, feature_cols = build_features(
        df,
        horizon,
        label_mode=label_mode,
        pt_sl=pt_sl,
        feature_set=feature_set,
        add_state_features=bool(add_state_features),
        state_window=int(state_window),
        state_funding_col=state_funding_col,
        add_alpha101=bool(add_alpha101),
        alpha101_list=alpha101_list,
        alpha101_rank_window=int(alpha101_rank_window),
        alpha101_adv_window=int(alpha101_adv_window),
    )
    if len(df_feat) < 500:
        print(f"  Insufficient data ({len(df_feat)} rows), skipping")
        return None

    # 重要提示：label 用的是未来 horizon 的收益（shift(-horizon)），如果 split/cv_gap 太小，
    # 训练集靠近边界的样本 label 会引用到验证/测试区间的价格，产生“轻微前视泄露”。
    # 更严格的做法是 purge+embargo（gap >= horizon）。这里先给出提醒，不强制改默认行为。
    eff_split_gap = int(split_gap) if split_gap is not None else 0
    eff_cv_gap = int(cv_gap) if cv_gap is not None else int(CV_CONFIG.get("gap", 0))
    if eff_split_gap < int(horizon) and not cv:
        print(f"  警告：--split-gap={eff_split_gap} < horizon={horizon}，边界附近样本可能存在未来信息泄露。建议设为 >= {horizon}。")
    if cv and eff_cv_gap < int(horizon):
        print(f"  警告：--cv-gap={eff_cv_gap} < horizon={horizon}，CV 边界附近样本可能存在未来信息泄露。建议设为 >= {horizon}。")

    if trade_step is None:
        trade_step = int(horizon)
    trade_step = max(int(trade_step), 1)

    used_cols = _select_features_if_needed(
        df_feat, feature_cols,
        no_select=bool(no_select),
        select_method=str(select_method),
        max_features=int(max_features),
    )
    if prune_corr is not None:
        before = len(used_cols)
        used_cols = _prune_correlated_features(
            df_feat,
            used_cols,
            threshold=float(prune_corr),
            method=str(corr_method),
            train_end_idx=int(len(df_feat) * 0.7),
        )
        print(f"  共线性剪枝: {before} -> {len(used_cols)} (thr={float(prune_corr):.3f}, method={corr_method})")
    print(f"  Features: {len(used_cols)} (raw={len(feature_cols)}), Samples: {len(df_feat)}")

    gbdt_only = all(mt in ("lgbm", "xgboost") for mt in model_types)
    use_tabular = bool(tabular or gbdt_only)
    if use_tabular and not gbdt_only:
        raise ValueError("tabular 模式仅支持 lgbm/xgboost")

    trainer = ModelTrainer()
    align = (align or "next").strip().lower()
    if align not in ("next", "current"):
        raise ValueError("align 仅支持: next / current")

    # ============ 1) GBDT（默认 tabular） ============
    if use_tabular:
        X, y = create_tabular(
            df_feat,
            used_cols,
            seq_length=seq_length,
            tabular_agg=tabular_agg,
            tabular_scale=tabular_scale,
            tabular_clip=tabular_clip,
            align=align,
        )
        gap = int(split_gap) if split_gap is not None else 0
        results = {
            'horizon': horizon,
            'n_features': len(used_cols),
            'n_train': int(len(X) * 0.7),
            'n_test': int(len(X) - int(len(X) * 0.85)),
            'test_up_ratio': float(np.mean(y[int(len(y) * 0.85):])) if len(y) else 0.0,
        }

        for mt in model_types:
            t0 = time.time()
            try:
                model_params = None
                fit_params = None
                if mt == "xgboost":
                    model_params = xgb_params
                    fit_params = xgb_fit_params
                elif mt == "lgbm":
                    model_params = lgbm_params
                    fit_params = lgbm_fit_params

                if cv:
                    cfg = {
                        "n_splits": int(cv_splits) if cv_splits is not None else None,
                        "gap": int(cv_gap) if cv_gap is not None else None,
                        "val_ratio": float(cv_val_ratio) if cv_val_ratio is not None else None,
                        "test_ratio": float(cv_test_ratio) if cv_test_ratio is not None else None,
                    }
                    cv_cfg = {k: v for k, v in cfg.items() if v is not None}
                    cv_cfg_full = {**CV_CONFIG, **(cv_cfg if cv_cfg else {})}

                    regime_mode_n = (regime_mode or "none").strip().lower()
                    regime2_mode_n = (regime2_mode or "none").strip().lower()
                    regime_train_n = (regime_train or "single").strip().lower()
                    if regime_mode_n != "none" and regime_train_n == "separate":
                        train_end_idx = int(len(df_feat) * 0.7)
                        row_reg = _compute_regime_row_labels(
                            df_feat,
                            mode=regime_mode_n,
                            window=int(regime_window),
                            bins=int(regime_bins),
                            train_end_idx=train_end_idx,
                            regime_col=regime_col,
                        )
                        reg_s = _align_row_labels_to_samples(row_reg, n_rows=len(df_feat), seq_length=int(seq_length), align=align)
                        # 过滤缺失 regime 的样本（-1）
                        good = reg_s >= 0
                        Xg, yg, rg = X[good], y[good], reg_s[good]

                        cv_out = _regime_cv_separate_train(
                            trainer,
                            model_type=mt,
                            X=Xg, y=yg,
                            regime_s=rg,
                            cv_cfg=cv_cfg_full,
                            model_params=model_params,
                            fit_params=fit_params,
                            min_train=int(regime_min_train),
                            min_val=int(regime_min_val),
                            min_test=int(regime_min_test),
                        )
                        results["regime_mode"] = regime_mode_n
                        results["regime_window"] = int(regime_window)
                        results["regime_bins"] = int(regime_bins)
                        results["regime_train"] = "separate"

                        results[f"{mt}_test_auc"] = float(cv_out.get("weighted_test_auc", 0.0))
                        results[f"{mt}_test_acc"] = float(cv_out.get("weighted_test_acc", 0.0))
                        results[f"{mt}_val_auc"] = 0.0  # regime separate 下 val/test 结构不同，先不输出整体 val
                        results[f"{mt}_time"] = round(time.time() - t0, 1)
                        results[f"{mt}_cv_folds"] = int(cv_cfg_full["n_splits"])
                        results[f"{mt}_test_auc_std"] = 0.0
                        results[f"{mt}_val_auc_std"] = 0.0

                        print(f"  {mt} (CV, regime=separate): weighted_test_auc={results[f'{mt}_test_auc']:.4f}, total_test_n={cv_out.get('total_test_n', 0)}, time={results[f'{mt}_time']:.1f}s")
                        for k, v in (cv_out.get("per_regime", {}) or {}).items():
                            results[f"{mt}_{k}_test_auc"] = float(v.get("test_auc_mean", 0.0))
                            results[f"{mt}_{k}_test_auc_std"] = float(v.get("test_auc_std", 0.0))
                            results[f"{mt}_{k}_test_n"] = int(v.get("test_n", 0))
                            results[f"{mt}_{k}_folds"] = int(v.get("folds_used", 0))
                            print(f"    {k}: test_auc={v.get('test_auc_mean',0.0):.4f}±{v.get('test_auc_std',0.0):.4f}, test_n={v.get('test_n',0)}, folds={v.get('folds_used',0)}")
                    elif regime_mode_n != "none" and regime_train_n == "single":
                        train_end_idx = int(len(df_feat) * 0.7)
                        row_reg = _compute_regime_row_labels(
                            df_feat,
                            mode=regime_mode_n,
                            window=int(regime_window),
                            bins=int(regime_bins),
                            train_end_idx=train_end_idx,
                            regime_col=regime_col,
                        )
                        reg_s = _align_row_labels_to_samples(row_reg, n_rows=len(df_feat), seq_length=int(seq_length), align=align)

                        reg2_s = None
                        if regime2_mode_n != "none":
                            row_reg2 = _compute_regime_row_labels(
                                df_feat,
                                mode=regime2_mode_n,
                                window=int(regime2_window),
                                bins=int(regime2_bins),
                                train_end_idx=train_end_idx,
                                regime_col=regime2_col,
                            )
                            reg2_s = _align_row_labels_to_samples(row_reg2, n_rows=len(df_feat), seq_length=int(seq_length), align=align)

                        fr_rows = pd.to_numeric(df_feat["future_return"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
                        future_return_s = fr_rows[int(seq_length):] if align == "next" else fr_rows[int(seq_length) - 1:]
                        if len(future_return_s) != len(y):
                            raise RuntimeError(
                                f"future_return_s 与样本长度不一致：len(future_return_s)={len(future_return_s)} vs len(y)={len(y)}。"
                                f"（align={align}, seq_length={seq_length}, n_rows={len(df_feat)}）"
                            )

                        cv_out2 = _regime_cv_single_train_report_two(
                            trainer,
                            model_type=mt,
                            X=X,
                            y=y,
                            regime1_s=reg_s,
                            regime2_s=reg2_s,
                            future_return_s=future_return_s,
                            cv_cfg=cv_cfg_full,
                            model_params=model_params,
                            fit_params=fit_params,
                            trade_eval=bool(trade_eval),
                            trade_mode=str(trade_mode),
                            trade_long_thr=float(trade_long_thr),
                            trade_short_thr=float(trade_short_thr),
                            trade_cost_bps=float(trade_cost_bps),
                            trade_step=int(trade_step),
                        )

                        results["regime_mode"] = regime_mode_n
                        results["regime_window"] = int(regime_window)
                        results["regime_bins"] = int(regime_bins)
                        if regime2_mode_n != "none":
                            results["regime2_mode"] = regime2_mode_n
                            results["regime2_window"] = int(regime2_window)
                            results["regime2_bins"] = int(regime2_bins)
                        results["regime_train"] = "single"

                        ov = (cv_out2.get("regime1", {}) or {}).get("overall", {}) or {}
                        results[f"{mt}_val_auc"] = float(ov.get("val_auc_mean", 0.0))
                        results[f"{mt}_val_auc_std"] = float(ov.get("val_auc_std", 0.0))
                        results[f"{mt}_test_auc"] = float(ov.get("test_auc_mean", 0.0))
                        results[f"{mt}_test_auc_std"] = float(ov.get("test_auc_std", 0.0))
                        results[f"{mt}_test_acc"] = float(ov.get("test_acc_mean", 0.0))
                        results[f"{mt}_time"] = round(time.time() - t0, 1)
                        results[f"{mt}_cv_folds"] = int((cv_out2.get("regime1", {}) or {}).get("folds", cv_cfg_full["n_splits"]))

                        print(f"  {mt} (CV, regime=report): avg_test_auc={results[f'{mt}_test_auc']:.4f}±{results[f'{mt}_test_auc_std']:.4f}, avg_val_auc={results[f'{mt}_val_auc']:.4f}±{results[f'{mt}_val_auc_std']:.4f}, time={results[f'{mt}_time']:.1f}s")
                        for k, v in ((cv_out2.get("regime1", {}) or {}).get("per_regime", {}) or {}).items():
                            results[f"{mt}_{k}_test_auc"] = float(v.get("test_auc_mean", 0.0))
                            results[f"{mt}_{k}_test_auc_std"] = float(v.get("test_auc_std", 0.0))
                            results[f"{mt}_{k}_coverage"] = float(v.get("coverage", 0.0))
                            results[f"{mt}_{k}_test_n"] = int(v.get("test_n", 0))
                            results[f"{mt}_{k}_folds"] = int(v.get("folds_used", 0))
                            results[f"{mt}_{k}_test_up_ratio"] = float(v.get("test_up_ratio", 0.0))
                            if trade_eval:
                                results[f"{mt}_{k}_trade_pnl_mean"] = float(v.get("trade_pnl_mean", 0.0))
                                results[f"{mt}_{k}_trade_hit_rate"] = float(v.get("trade_hit_rate", 0.0))
                                results[f"{mt}_{k}_trade_coverage"] = float(v.get("trade_coverage", 0.0))
                            print(
                                f"    {k}: test_auc={v.get('test_auc_mean',0.0):.4f}±{v.get('test_auc_std',0.0):.4f}, "
                                f"cov={v.get('coverage',0.0)*100:.1f}%, up={v.get('test_up_ratio',0.0)*100:.1f}%, "
                                f"test_n={v.get('test_n',0)}, folds={v.get('folds_used',0)}"
                                + (
                                    f", trade_pnl_mean={v.get('trade_pnl_mean',0.0):.6f}, trade_hit={v.get('trade_hit_rate',0.0)*100:.1f}%, trade_cov={v.get('trade_coverage',0.0)*100:.1f}%"
                                    if trade_eval
                                    else ""
                                )
                            )
                        if regime2_mode_n != "none":
                            print(f"  {mt} (regime2={regime2_mode_n}):")
                            for k, v in ((cv_out2.get("regime2", {}) or {}).get("per_regime", {}) or {}).items():
                                results[f"{mt}_regime2_{k}_test_auc"] = float(v.get("test_auc_mean", 0.0))
                                results[f"{mt}_regime2_{k}_test_auc_std"] = float(v.get("test_auc_std", 0.0))
                                results[f"{mt}_regime2_{k}_coverage"] = float(v.get("coverage", 0.0))
                                results[f"{mt}_regime2_{k}_test_n"] = int(v.get("test_n", 0))
                                results[f"{mt}_regime2_{k}_folds"] = int(v.get("folds_used", 0))
                                results[f"{mt}_regime2_{k}_test_up_ratio"] = float(v.get("test_up_ratio", 0.0))
                                if trade_eval:
                                    results[f"{mt}_regime2_{k}_trade_pnl_mean"] = float(v.get("trade_pnl_mean", 0.0))
                                    results[f"{mt}_regime2_{k}_trade_hit_rate"] = float(v.get("trade_hit_rate", 0.0))
                                    results[f"{mt}_regime2_{k}_trade_coverage"] = float(v.get("trade_coverage", 0.0))
                                print(
                                    f"    {k}: test_auc={v.get('test_auc_mean',0.0):.4f}±{v.get('test_auc_std',0.0):.4f}, "
                                    f"cov={v.get('coverage',0.0)*100:.1f}%, up={v.get('test_up_ratio',0.0)*100:.1f}%, "
                                    f"test_n={v.get('test_n',0)}, folds={v.get('folds_used',0)}"
                                    + (
                                        f", trade_pnl_mean={v.get('trade_pnl_mean',0.0):.6f}, trade_hit={v.get('trade_hit_rate',0.0)*100:.1f}%, trade_cov={v.get('trade_coverage',0.0)*100:.1f}%"
                                        if trade_eval
                                        else ""
                                    )
                                )
                            inter = cv_out2.get("intersection", {}) or {}
                            per_pair = inter.get("per_pair", {}) or {}
                            if per_pair:
                                print(f"  {mt} (intersection {regime_mode_n} & {regime2_mode_n}):")
                                # 按 test_auc_mean 降序打印前几个组合
                                items = list(per_pair.items())
                                items.sort(key=lambda kv: float(kv[1].get("test_auc_mean", 0.0)), reverse=True)
                                for key, v in items[:6]:
                                    results[f"{mt}_inter_{key}_test_auc"] = float(v.get("test_auc_mean", 0.0))
                                    results[f"{mt}_inter_{key}_test_auc_std"] = float(v.get("test_auc_std", 0.0))
                                    results[f"{mt}_inter_{key}_coverage"] = float(v.get("coverage", 0.0))
                                    results[f"{mt}_inter_{key}_test_n"] = int(v.get("test_n", 0))
                                    results[f"{mt}_inter_{key}_folds"] = int(v.get("folds_used", 0))
                                    results[f"{mt}_inter_{key}_test_up_ratio"] = float(v.get("test_up_ratio", 0.0))
                                    if trade_eval:
                                        results[f"{mt}_inter_{key}_trade_pnl_mean"] = float(v.get("trade_pnl_mean", 0.0))
                                        results[f"{mt}_inter_{key}_trade_hit_rate"] = float(v.get("trade_hit_rate", 0.0))
                                        results[f"{mt}_inter_{key}_trade_coverage"] = float(v.get("trade_coverage", 0.0))
                                    print(
                                        f"    {key}: test_auc={v.get('test_auc_mean',0.0):.4f}±{v.get('test_auc_std',0.0):.4f}, "
                                        f"cov={v.get('coverage',0.0)*100:.1f}%, up={v.get('test_up_ratio',0.0)*100:.1f}%, "
                                        f"test_n={v.get('test_n',0)}, folds={v.get('folds_used',0)}"
                                        + (
                                            f", trade_pnl_mean={v.get('trade_pnl_mean',0.0):.6f}, trade_hit={v.get('trade_hit_rate',0.0)*100:.1f}%, trade_cov={v.get('trade_coverage',0.0)*100:.1f}%"
                                            if trade_eval
                                            else ""
                                        )
                                    )
                    else:
                        cv_out = trainer.cross_validate(
                            mt, X, y,
                            cv_config=cv_cfg if cv_cfg else None,
                            model_params=model_params,
                            fit_params=fit_params,
                        )
                        fold_results = cv_out.get("fold_results", [])
                        fold_val_aucs = [float(r.get("val_auc", 0.0)) for r in fold_results]
                        fold_test_aucs = [float(r.get("test_auc", 0.0)) for r in fold_results]
                        avg_val_auc = float(np.mean([r.get("val_auc", 0.0) for r in fold_results])) if fold_results else 0.0
                        results[f'{mt}_val_auc'] = avg_val_auc
                        results[f'{mt}_test_auc'] = float(cv_out.get("avg_test_auc", 0.0))
                        results[f'{mt}_test_acc'] = float(cv_out.get("avg_test_acc", 0.0))
                        results[f'{mt}_time'] = round(time.time() - t0, 1)
                        results[f'{mt}_cv_folds'] = int(len(fold_results))
                        results[f'{mt}_val_auc_std'] = float(np.std(fold_val_aucs)) if fold_val_aucs else 0.0
                        results[f'{mt}_test_auc_std'] = float(np.std(fold_test_aucs)) if fold_test_aucs else 0.0
                        print(f"  {mt} (CV): folds={len(fold_results)}, avg_val_auc={avg_val_auc:.4f}, avg_test_auc={results[f'{mt}_test_auc']:.4f}, time={results[f'{mt}_time']:.1f}s")
                else:
                    n = len(X)
                    train_end = int(n * 0.7)
                    val_size = int(n * 0.15)
                    val_start = train_end + gap
                    val_end = val_start + val_size
                    test_start = val_end + gap
                    if val_end >= n or test_start >= n:
                        raise ValueError(f"split_gap={gap} 过大，导致 val/test 为空（n={n}）。请减小 --split-gap。")
                    X_train, y_train = X[:train_end], y[:train_end]
                    X_val, y_val = X[val_start:val_end], y[val_start:val_end]
                    X_test, y_test = X[test_start:], y[test_start:]
                    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
                    print(f"  Label dist: up={y_test.mean()*100:.1f}%")

                    model, metrics = trainer.train_model(
                        mt, X_train, y_train, X_val, y_val,
                        model_params=model_params,
                        fit_params=fit_params,
                    )
                    test_probs = model.predict_proba(X_test)
                    test_auc = roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else 0.5
                    test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))
                    elapsed = time.time() - t0

                    results[f'{mt}_val_auc'] = metrics.get('val_auc', 0)
                    results[f'{mt}_test_auc'] = float(test_auc)
                    results[f'{mt}_test_acc'] = float(test_acc)
                    results[f'{mt}_time'] = round(elapsed, 1)
                    print(f"  {mt}: val_auc={metrics.get('val_auc', 0):.4f}, test_auc={test_auc:.4f}, time={elapsed:.1f}s")
            except Exception as e:
                print(f"  {mt} FAILED: {e}")
                results[f'{mt}_val_auc'] = 0
                results[f'{mt}_test_auc'] = 0
                results[f'{mt}_test_acc'] = 0
                results[f'{mt}_time'] = 0
        results["n_train"] = int(len(X_train)) if "X_train" in locals() else results["n_train"]
        results["n_test"] = int(len(X_test)) if "X_test" in locals() else results["n_test"]
        results["test_up_ratio"] = float(np.mean(y_test)) if "y_test" in locals() and len(y_test) else results["test_up_ratio"]
        return results

    # ============ 2) 深度模型滚动 Dataset（省内存） ============
    if rolling_dataset:
        if any(mt in ("lgbm", "xgboost") for mt in model_types):
            raise ValueError("rolling-dataset 仅用于深度模型；lgbm/xgboost 请加 --tabular")

        X_rows = df_feat[used_cols].values
        y_rows = df_feat["label"].values.astype(np.int64)
        X_rows = np.nan_to_num(X_rows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        n_rows = len(X_rows)
        n_samples = n_rows - int(seq_length) + (1 if align == "current" else 0)  # sample start s in [0, n_samples)
        if n_samples < 500:
            print(f"  Insufficient sequence samples ({n_samples}), skipping")
            return None

        gap = int(split_gap) if split_gap is not None else 0
        train_end_s = int(n_samples * 0.7)
        val_size_s = int(n_samples * 0.15)
        val_start_s = train_end_s + gap
        val_end_s = val_start_s + val_size_s
        test_start_s = val_end_s + gap
        if val_end_s >= n_samples or test_start_s >= n_samples:
            raise ValueError(f"split_gap={gap} 过大，导致 val/test 为空（n_samples={n_samples}）。请减小 --split-gap。")

        # scaler fit 到训练样本覆盖到的最后一行，避免泄露
        fit_end_row = max(train_end_s + int(seq_length) - 2, 0)
        scaler = RobustScaler()
        scaler.fit(X_rows[:fit_end_row + 1])
        X_rows = scaler.transform(X_rows).astype(np.float32)
        X_rows = np.clip(X_rows, -5, 5)

        bs = int(batch_size) if batch_size else int(TRAINING_CONFIG["batch_size"])
        train_ds = RollingWindowDataset(X_rows, y_rows, seq_length=int(seq_length), start=0, end=train_end_s, align=align)
        val_ds = RollingWindowDataset(X_rows, y_rows, seq_length=int(seq_length), start=val_start_s, end=val_end_s, align=align)
        test_ds = RollingWindowDataset(X_rows, y_rows, seq_length=int(seq_length), start=test_start_s, end=n_samples, align=align)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)

        label_offset = int(seq_length) if align == "next" else int(seq_length) - 1
        y_test_samples = y_rows[label_offset + test_start_s:label_offset + n_samples]
        print(f"  Split(samples): train={train_end_s}, val={val_end_s-val_start_s}, test={n_samples-test_start_s}")
        print(f"  Label dist(test): up={y_test_samples.mean()*100:.1f}%")

        results = {
            'horizon': horizon,
            'n_features': len(used_cols),
            'n_train': int(train_end_s),
            'n_test': int(n_samples - test_start_s),
            'test_up_ratio': float(y_test_samples.mean()) if len(y_test_samples) else 0.0,
        }

        for mt in model_types:
            t0 = time.time()
            try:
                y_train_samples = y_rows[label_offset:label_offset + train_end_s]
                model, metrics = _train_deep_with_loaders(
                    trainer,
                    model_type=mt,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    y_train_samples=y_train_samples,
                    input_size=int(X_rows.shape[1]),
                    seq_length=int(seq_length),
                )
                criterion = torch.nn.BCEWithLogitsLoss()
                _, test_auc, test_acc, _, _ = trainer.evaluate(model, test_loader, criterion)
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

    # ============ 3) 深度模型（构造完整 3D 序列，耗内存） ============
    X, y = create_sequences(df_feat, used_cols, seq_length, align=align)
    n = len(X)
    gap = int(split_gap) if split_gap is not None else 0
    train_end = int(n * 0.7)
    val_size = int(n * 0.15)
    val_start = train_end + gap
    val_end = val_start + val_size
    test_start = val_end + gap
    if val_end >= n or test_start >= n:
        raise ValueError(f"split_gap={gap} 过大，导致 val/test 为空（n={n}）。请减小 --split-gap。")
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[val_start:val_end], y[val_start:val_end]
    X_test, y_test = X[test_start:], y[test_start:]

    print(f"  Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Label dist: up={y_test.mean()*100:.1f}%")

    results = {
        'horizon': horizon,
        'n_features': len(used_cols),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_up_ratio': y_test.mean(),
    }

    for mt in model_types:
        t0 = time.time()
        try:
            model, metrics = trainer.train_model(mt, X_train, y_train, X_val, y_val)

            if mt in ('lgbm', 'xgboost'):
                test_probs = model.predict_proba(X_test)
            else:
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
    parser.add_argument('--hf-csv', default=None, help='使用预先计算的高频特征 CSV（替代本地 pkl 数据源）')
    parser.add_argument('--feature-set', default='ta+hf', choices=['ta', 'hf', 'ta+hf'],
                        help='特征集选择：ta=仅TA；hf=仅CSV自带特征；ta+hf=合并')
    parser.add_argument('--data-mode', default='ohlcv', choices=['ohlcv', 'orderflow'],
                        help='Data source: ohlcv (default) or orderflow (Binance klines taker-buy proxy)')
    parser.add_argument('--horizons', default=None, help='Comma-separated horizons (e.g., 1,7,14)')
    parser.add_argument('--models', default='lgbm,xgboost',
                        help='Comma-separated model types')
    parser.add_argument('--seq-length', type=int, default=60)
    parser.add_argument('--tabular', action='store_true',
                        help='对 lgbm/xgboost 使用 tabular(last/mean/std) 表示，避免构造 3D 大矩阵（更省内存）')
    parser.add_argument(
        '--tabular-agg',
        default='last_mean_std',
        choices=[
            'last',
            'last_mean',
            'last_mean_std',
            'last_mean_std_slope',
            'last_mean_std_z',
            'last_mean_std_slope_z',
        ],
        help='tabular 聚合方式（用于 --tabular）：last / last_mean / last_mean_std / last_mean_std_slope / last_mean_std_z / last_mean_std_slope_z',
    )
    parser.add_argument('--tabular-scale', default='robust', choices=['robust', 'none'],
                        help='tabular 缩放方式（用于 --tabular）：robust / none')
    parser.add_argument('--tabular-clip', type=float, default=5.0,
                        help='tabular 缩放后裁剪阈值；设为 0 表示不裁剪（用于 --tabular）')
    parser.add_argument('--align', default='next', choices=['next', 'current'],
                        help='样本对齐口径：next=用到t-1预测t；current=用到t预测t（更常见的bar-close口径）')
    parser.add_argument('--split-gap', type=int, default=0,
                        help='train/val/test 之间的样本间隔（embargo，防止相邻窗口强相关）。默认0；建议与 --cv-gap 同量级（例如10）。')
    parser.add_argument('--rolling-dataset', action='store_true',
                        help='深度模型使用滚动窗口 Dataset 动态取样（显著降低内存峰值；推荐 transformer/lstm 在大样本上用）')
    parser.add_argument('--add-state-features', action='store_true',
                        help='添加连续“状态变量”特征（trend/vol/funding 等），不做 regime 分桶，让模型自动学习分段/交互')
    parser.add_argument('--state-window', type=int, default=48,
                        help='连续状态特征窗口（用于 trend/vol），默认48')
    parser.add_argument('--state-funding-col', type=str, default=None,
                        help='资金费率类连续状态特征列名（可选；不填则自动在 funding_pressure/funding_rate/funding_annualized 中选）')
    parser.add_argument('--add-alpha101', action='store_true',
                        help='添加 Alpha101（单品种近似版）因子特征（默认 1,3~20；可用 --alpha101-list 自定义）')
    parser.add_argument('--alpha101-list', type=str, default=None,
                        help='Alpha101 因子编号列表（逗号分隔）。例如: 1,3,5,7,10；不填则默认 1,3~20')
    parser.add_argument('--alpha101-rank-window', type=int, default=20,
                        help='Alpha101 中 rank/scale 的时间序列近似窗口（默认20）')
    parser.add_argument('--alpha101-adv-window', type=int, default=20,
                        help='Alpha101 中 adv 的窗口（默认20）')
    parser.add_argument('--no-select', action='store_true', help='关闭特征选择，直接使用全部候选特征')
    parser.add_argument('--select-method', default='mutual_info', choices=['mutual_info', 'f_classif', 'random_forest'],
                        help='特征选择方法')
    parser.add_argument('--max-features', type=int, default=MAX_FEATURES, help='特征选择保留上限')
    parser.add_argument('--batch-size', type=int, default=None, help='覆盖默认 batch_size（不填则用 config.settings）')
    parser.add_argument('--prune-corr', type=float, default=None,
                        help='对选择后的特征做共线性剪枝（训练段计算 abs(corr)；例如 0.98）')
    parser.add_argument('--corr-method', default='pearson', choices=['pearson', 'spearman'],
                        help='相关系数方法（用于 --prune-corr）')
    parser.add_argument('--xgb-params', type=str, default=None,
                        help='覆盖/追加 XGBoost 参数（JSON dict）。例如: \'{"max_depth":4,"min_child_weight":10,"subsample":0.8,"colsample_bytree":0.8,"reg_lambda":3}\'')
    parser.add_argument('--xgb-num-boost-round', type=int, default=None, help='覆盖 XGBoost num_boost_round')
    parser.add_argument('--xgb-early-stopping-rounds', type=int, default=None, help='覆盖 XGBoost early_stopping_rounds')
    parser.add_argument('--lgbm-params', type=str, default=None,
                        help='覆盖/追加 LightGBM 参数（JSON dict）。例如: \'{"num_leaves":31,"min_child_samples":50}\'')
    parser.add_argument('--lgbm-num-boost-round', type=int, default=None, help='覆盖 LightGBM num_boost_round')
    parser.add_argument('--lgbm-early-stopping-rounds', type=int, default=None, help='覆盖 LightGBM early_stopping_rounds')
    parser.add_argument('--cv', action='store_true',
                        help='使用 expanding window CV 评估（更稳，避免单次切分偶然性）')
    parser.add_argument('--cv-splits', type=int, default=None, help='CV 折数（默认用 config.settings）')
    parser.add_argument('--cv-gap', type=int, default=None, help='CV gap（默认用 config.settings）')
    parser.add_argument('--cv-val-ratio', type=float, default=None, help='CV 验证集比例（默认用 config.settings）')
    parser.add_argument('--cv-test-ratio', type=float, default=None, help='CV 测试集比例（默认用 config.settings）')
    parser.add_argument('--regime-mode', default='none',
                        choices=['none', 'vol_quantile', 'trend_sign', 'funding_sign'],
                        help='Regime 划分方式：none / vol_quantile / trend_sign / funding_sign')
    parser.add_argument('--regime-window', type=int, default=48, help='regime rolling 窗口（用于 vol/trend）')
    parser.add_argument('--regime-bins', type=int, default=2, help='regime 分桶数（仅用于 vol_quantile）')
    parser.add_argument('--regime-col', type=str, default=None, help='regime 使用的列名（用于 funding_sign，可不填自动猜）')
    parser.add_argument('--regime2-mode', default='none',
                        choices=['none', 'vol_quantile', 'trend_sign', 'funding_sign'],
                        help='第二个 Regime 划分方式（用于 AND 交叉报表）：none / vol_quantile / trend_sign / funding_sign')
    parser.add_argument('--regime2-window', type=int, default=48, help='regime2 rolling 窗口（用于 vol/trend）')
    parser.add_argument('--regime2-bins', type=int, default=2, help='regime2 分桶数（仅用于 vol_quantile）')
    parser.add_argument('--regime2-col', type=str, default=None, help='regime2 使用的列名（用于 funding_sign，可不填自动猜）')
    parser.add_argument('--regime-train', default='single', choices=['single', 'separate'],
                        help='regime 训练方式：single=全样本一个模型；separate=按regime分别训练')
    parser.add_argument('--regime-min-train', type=int, default=2000, help='separate 训练时每个regime每折最小训练样本数')
    parser.add_argument('--regime-min-val', type=int, default=500, help='separate 训练时每个regime每折最小验证样本数')
    parser.add_argument('--regime-min-test', type=int, default=500, help='separate 训练时每个regime每折最小测试样本数')
    parser.add_argument('--trade-eval', action='store_true',
                        help='在 regime 报表中输出最简交易统计（用 future_return 近似；默认每个样本独立计算）')
    parser.add_argument('--trade-mode', default='long', choices=['long', 'long_short'],
                        help='交易方向：long=仅做多；long_short=多空')
    parser.add_argument('--trade-long-thr', type=float, default=0.55, help='做多阈值：prob>=thr 触发开仓')
    parser.add_argument('--trade-short-thr', type=float, default=0.45, help='做空阈值：prob<=thr 触发开仓（仅 long_short）')
    parser.add_argument('--trade-cost-bps', type=float, default=0.0, help='单笔交易成本（bps），会从每笔样本 PnL 扣减')
    parser.add_argument('--trade-step', type=int, default=None,
                        help='trade_eval 下采样步长（减少 horizon 重叠带来的虚高）。默认=当前 horizon')
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

    if args.hf_csv:
        if args.micro:
            print("警告：hf-csv 模式下忽略 --micro（CSV 已包含你计算的高频/微观结构特征）")
        df = load_hf_csv(args.hf_csv)
    else:
        if args.data_mode == 'orderflow':
            df = load_btc_orderflow(args.freq)
        else:
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

    xgb_params = _parse_json_dict(args.xgb_params, "--xgb-params")
    lgbm_params = _parse_json_dict(args.lgbm_params, "--lgbm-params")
    xgb_fit_params = {}
    if args.xgb_num_boost_round is not None:
        xgb_fit_params["num_boost_round"] = int(args.xgb_num_boost_round)
    if args.xgb_early_stopping_rounds is not None:
        xgb_fit_params["early_stopping_rounds"] = int(args.xgb_early_stopping_rounds)
    if not xgb_fit_params:
        xgb_fit_params = None

    lgbm_fit_params = {}
    if args.lgbm_num_boost_round is not None:
        lgbm_fit_params["num_boost_round"] = int(args.lgbm_num_boost_round)
    if args.lgbm_early_stopping_rounds is not None:
        lgbm_fit_params["early_stopping_rounds"] = int(args.lgbm_early_stopping_rounds)
    if not lgbm_fit_params:
        lgbm_fit_params = None

    all_results = []
    for h in horizons:
        print(f'\n{"="*70}')
        print(f'  Horizon: {h} bars')
        print(f'{"="*70}')
        result = run_experiment(
            df, h, model_types,
            seq_length=args.seq_length,
            label_mode=args.label_mode,
            pt_sl=pt_sl_vals,
            feature_set=args.feature_set,
            tabular=bool(args.tabular),
            rolling_dataset=bool(args.rolling_dataset),
            add_state_features=bool(args.add_state_features),
            state_window=int(args.state_window),
            state_funding_col=args.state_funding_col,
            add_alpha101=bool(args.add_alpha101),
            alpha101_list=args.alpha101_list,
            alpha101_rank_window=int(args.alpha101_rank_window),
            alpha101_adv_window=int(args.alpha101_adv_window),
            no_select=bool(args.no_select),
            select_method=str(args.select_method),
            max_features=int(args.max_features),
            batch_size=args.batch_size,
            prune_corr=args.prune_corr,
            corr_method=str(args.corr_method),
            tabular_agg=str(args.tabular_agg),
            tabular_scale=str(args.tabular_scale),
            tabular_clip=(None if float(args.tabular_clip) <= 0 else float(args.tabular_clip)),
            align=str(args.align),
            trade_eval=bool(args.trade_eval),
            trade_mode=str(args.trade_mode),
            trade_long_thr=float(args.trade_long_thr),
            trade_short_thr=float(args.trade_short_thr),
            trade_cost_bps=float(args.trade_cost_bps),
            trade_step=args.trade_step,
            xgb_params=xgb_params,
            lgbm_params=lgbm_params,
            xgb_fit_params=xgb_fit_params,
            lgbm_fit_params=lgbm_fit_params,
            split_gap=int(args.split_gap),
            cv=bool(args.cv),
            cv_splits=args.cv_splits,
            cv_gap=args.cv_gap,
            cv_val_ratio=args.cv_val_ratio,
            cv_test_ratio=args.cv_test_ratio,
            regime_mode=str(args.regime_mode),
            regime_window=int(args.regime_window),
            regime_bins=int(args.regime_bins),
            regime_col=args.regime_col,
            regime2_mode=str(args.regime2_mode),
            regime2_window=int(args.regime2_window),
            regime2_bins=int(args.regime2_bins),
            regime2_col=args.regime2_col,
            regime_train=str(args.regime_train),
            regime_min_train=int(args.regime_min_train),
            regime_min_val=int(args.regime_min_val),
            regime_min_test=int(args.regime_min_test),
        )
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
        cols.extend([
            f'{mt}_test_auc', f'{mt}_test_auc_std',
            f'{mt}_val_auc', f'{mt}_val_auc_std',
            f'{mt}_time', f'{mt}_cv_folds',
        ])
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
