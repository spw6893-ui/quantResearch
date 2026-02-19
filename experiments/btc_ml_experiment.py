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
from models.trainer import ModelTrainer, EarlyStopping
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
    if agg not in ("last", "last_mean", "last_mean_std", "last_mean_std_slope"):
        raise ValueError("tabular_agg 仅支持: last / last_mean / last_mean_std / last_mean_std_slope")
    mult = {"last": 1, "last_mean": 2, "last_mean_std": 3, "last_mean_std_slope": 4}[agg]

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
        else:
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

def build_features(df, horizon, label_mode='binary', pt_sl=(1.0, 1.0), feature_set: str = "ta+hf"):
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
                   xgb_params: dict | None = None,
                   lgbm_params: dict | None = None,
                   xgb_fit_params: dict | None = None,
                   lgbm_fit_params: dict | None = None,
                   split_gap: int = 0,
                   cv: bool = False,
                   cv_splits: int | None = None,
                   cv_gap: int | None = None,
                   cv_val_ratio: float | None = None,
                   cv_test_ratio: float | None = None):
    """Run single horizon experiment with multiple models."""
    df_feat, feature_cols = build_features(df, horizon, label_mode, pt_sl, feature_set=feature_set)
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
    parser.add_argument('--tabular-agg', default='last_mean_std', choices=['last', 'last_mean', 'last_mean_std', 'last_mean_std_slope'],
                        help='tabular 聚合方式（用于 --tabular）：last / last_mean / last_mean_std / last_mean_std_slope')
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
