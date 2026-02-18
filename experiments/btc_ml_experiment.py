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
    return df


def _seq_to_tabular_fast(X_rows: np.ndarray, seq_length: int) -> np.ndarray:
    """把 (N_rows, F) 快速转为 (N_samples, 3F)：last + mean + std（窗口长度=seq_length）。"""
    X = np.asarray(X_rows, dtype=np.float32)
    n_rows, n_feat = X.shape
    seq_length = int(seq_length)
    n_samples = n_rows - seq_length + 1
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    csum = np.cumsum(X.astype(np.float64), axis=0)
    csum2 = np.cumsum((X.astype(np.float64) ** 2), axis=0)

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
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]
    df = df.dropna(subset=feature_cols + ['label']).reset_index(drop=True)
    df['label'] = df['label'].astype(int)
    return df, feature_cols


def create_sequences(df, feature_cols, seq_length=60):
    """Create 3D sequences for deep learning models."""
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


def create_tabular(df, feature_cols, seq_length=60):
    """Create 2D tabular features for GBDT models: last + mean + std, aligned to window end."""
    X_rows = df[feature_cols].values
    y_rows = df["label"].values.astype(np.int64)
    X_rows = np.nan_to_num(X_rows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    n_rows = len(X_rows)
    seq_length = int(seq_length)
    n_samples = n_rows - seq_length + 1
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    # 划分按样本比例，映射回行做 scaler fit，避免泄露
    train_end_s = int(n_samples * 0.7)
    train_end_row = train_end_s + seq_length - 1
    scaler = RobustScaler()
    scaler.fit(X_rows[:train_end_row + 1])
    X_scaled = scaler.transform(X_rows).astype(np.float32)
    X_scaled = np.clip(X_scaled, -5, 5)

    X_tab = _seq_to_tabular_fast(X_scaled, seq_length=seq_length)
    y = y_rows[seq_length - 1:]
    return X_tab, y


class RollingWindowDataset(Dataset):
    """滚动窗口序列数据集：window = [s, s+seq_length)，label = y[s+seq_length]。

    对齐方式与本项目 create_sequences 保持一致（用过去 seq_length 根预测“下一根”的 label）。
    """

    def __init__(self, X_rows: np.ndarray, y_rows: np.ndarray, seq_length: int, start: int, end: int):
        self.X_rows = np.asarray(X_rows, dtype=np.float32)
        self.y_rows = np.asarray(y_rows, dtype=np.float32)
        self.seq_length = int(seq_length)
        self.start = int(start)
        self.end = int(end)

        n_rows = len(self.X_rows)
        n_samples = n_rows - self.seq_length
        if n_samples <= 0:
            raise ValueError("seq_length 太大，无法生成样本")
        if not (0 <= self.start <= self.end <= n_samples):
            raise ValueError("RollingWindowDataset: start/end 超出范围")

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx: int):
        s = self.start + int(idx)
        x = self.X_rows[s:s + self.seq_length]
        y = self.y_rows[s + self.seq_length]
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
                   batch_size: int | None = None):
    """Run single horizon experiment with multiple models."""
    df_feat, feature_cols = build_features(df, horizon, label_mode, pt_sl, feature_set=feature_set)
    if len(df_feat) < 500:
        print(f"  Insufficient data ({len(df_feat)} rows), skipping")
        return None

    used_cols = _select_features_if_needed(
        df_feat, feature_cols,
        no_select=bool(no_select),
        select_method=str(select_method),
        max_features=int(max_features),
    )
    print(f"  Features: {len(used_cols)} (raw={len(feature_cols)}), Samples: {len(df_feat)}")

    gbdt_only = all(mt in ("lgbm", "xgboost") for mt in model_types)
    use_tabular = bool(tabular or gbdt_only)
    if use_tabular and not gbdt_only:
        raise ValueError("tabular 模式仅支持 lgbm/xgboost")

    trainer = ModelTrainer()

    # ============ 1) GBDT（默认 tabular） ============
    if use_tabular:
        X, y = create_tabular(df_feat, used_cols, seq_length)
        n = len(X)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

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
                test_probs = model.predict_proba(X_test)
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

    # ============ 2) 深度模型滚动 Dataset（省内存） ============
    if rolling_dataset:
        if any(mt in ("lgbm", "xgboost") for mt in model_types):
            raise ValueError("rolling-dataset 仅用于深度模型；lgbm/xgboost 请加 --tabular")

        X_rows = df_feat[used_cols].values
        y_rows = df_feat["label"].values.astype(np.int64)
        X_rows = np.nan_to_num(X_rows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        n_rows = len(X_rows)
        n_samples = n_rows - int(seq_length)  # sample start s in [0, n_samples)
        if n_samples < 500:
            print(f"  Insufficient sequence samples ({n_samples}), skipping")
            return None

        train_end_s = int(n_samples * 0.7)
        val_end_s = int(n_samples * 0.85)

        # scaler fit 到训练样本覆盖到的最后一行，避免泄露
        fit_end_row = train_end_s + int(seq_length)
        scaler = RobustScaler()
        scaler.fit(X_rows[:fit_end_row + 1])
        X_rows = scaler.transform(X_rows).astype(np.float32)
        X_rows = np.clip(X_rows, -5, 5)

        bs = int(batch_size) if batch_size else int(TRAINING_CONFIG["batch_size"])
        train_ds = RollingWindowDataset(X_rows, y_rows, seq_length=int(seq_length), start=0, end=train_end_s)
        val_ds = RollingWindowDataset(X_rows, y_rows, seq_length=int(seq_length), start=train_end_s, end=val_end_s)
        test_ds = RollingWindowDataset(X_rows, y_rows, seq_length=int(seq_length), start=val_end_s, end=n_samples)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False)
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, drop_last=False)

        y_test_samples = y_rows[int(seq_length) + val_end_s:int(seq_length) + n_samples]
        print(f"  Split(samples): train={train_end_s}, val={val_end_s-train_end_s}, test={n_samples-val_end_s}")
        print(f"  Label dist(test): up={y_test_samples.mean()*100:.1f}%")

        results = {
            'horizon': horizon,
            'n_features': len(used_cols),
            'n_train': int(train_end_s),
            'n_test': int(n_samples - val_end_s),
            'test_up_ratio': float(y_test_samples.mean()) if len(y_test_samples) else 0.0,
        }

        for mt in model_types:
            t0 = time.time()
            try:
                y_train_samples = y_rows[int(seq_length):int(seq_length) + train_end_s]
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
    X, y = create_sequences(df_feat, used_cols, seq_length)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

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
    parser.add_argument('--rolling-dataset', action='store_true',
                        help='深度模型使用滚动窗口 Dataset 动态取样（显著降低内存峰值；推荐 transformer/lstm 在大样本上用）')
    parser.add_argument('--no-select', action='store_true', help='关闭特征选择，直接使用全部候选特征')
    parser.add_argument('--select-method', default='mutual_info', choices=['mutual_info', 'f_classif', 'random_forest'],
                        help='特征选择方法')
    parser.add_argument('--max-features', type=int, default=MAX_FEATURES, help='特征选择保留上限')
    parser.add_argument('--batch-size', type=int, default=None, help='覆盖默认 batch_size（不填则用 config.settings）')
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
