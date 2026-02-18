"""
BTC 高频特征适配实验

目标：
- 将 /home/ppw/quantResearch/BTC_USDT:USDT_final.csv 中的高频特征接入当前模型训练流水线
- 对比不同特征集（TA-only / HF-only / TA+HF）在固定切分下的 Test AUC

说明：
- 数据粒度：小时线（该 CSV 的中位间隔为 3600s）
- 标签：future_return = close[t+h]/close[t]-1，label = (future_return > 0)
- 为避免“最后 h 行未知标签被误当作 0”，会显式丢弃 future_return 为 NaN 的样本
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import MAX_FEATURES, RANDOM_SEED
from data.feature_engineering import FeatureEngineer
from experiments.btc_signal_scan import add_ta_indicators
from models.trainer import ModelTrainer
from utils.helpers import set_seed

set_seed(RANDOM_SEED)

DEFAULT_CSV = "/home/ppw/quantResearch/BTC_USDT:USDT_final.csv"


def _to_utc_naive_datetime(s: pd.Series) -> pd.Series:
    """将 datetime 列统一成 UTC 的 naive datetime（便于与其他数据源对齐/排序）。"""
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt.dt.tz_convert(None)


def load_hf_data(csv_path: str) -> pd.DataFrame:
    """加载高频特征 CSV，并按时间排序去重。"""
    df = pd.read_csv(csv_path)
    if "datetime" not in df.columns:
        raise ValueError("CSV 缺少 datetime 列，无法进行时序对齐。")
    df["datetime"] = _to_utc_naive_datetime(df["datetime"])
    df = df.dropna(subset=["datetime"]).drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    return df


def add_label(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """基于 close 构造标签（未来 horizon 根 K 的方向）。"""
    if "close" not in df.columns:
        raise ValueError("缺少 close 列，无法生成标签。")
    future_return = df["close"].shift(-int(horizon)) / df["close"] - 1
    label = (future_return > 0).astype("float32")
    label[future_return.isna()] = np.nan
    out = df.copy()
    out["future_return"] = future_return
    out["label"] = label
    # 丢弃未知标签行（通常是最后 horizon 行）
    out = out.dropna(subset=["label"]).reset_index(drop=True)
    out["label"] = out["label"].astype(int)
    return out


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in ("label", "future_return")
    ]


def build_feature_columns(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    """构建不同特征集对应的 feature_cols。

    feature_set:
      - 'ta'      : 仅使用 add_ta_indicators 生成的 TA 特征
      - 'hf'      : 仅使用 CSV 自带的高频特征（不含 TA 衍生列）
      - 'ta+hf'   : TA + 高频特征并集
    """
    feature_set = feature_set.strip().lower()
    if feature_set not in ("ta", "hf", "ta+hf"):
        raise ValueError("feature_set 仅支持: ta / hf / ta+hf")

    base_cols = ["datetime", "open", "high", "low", "close", "volume"]
    # amount 不是必须列，但很多地方会默认存在
    work = df.copy()
    if "amount" not in work.columns and all(c in work.columns for c in ("close", "volume")):
        work["amount"] = work["close"].astype(float) * work["volume"].astype(float)

    # TA-only 作为严格基线：先裁剪到 OHLCV(+amount)，再生成 TA
    if feature_set == "ta":
        missing = [c for c in base_cols if c not in work.columns]
        if missing:
            raise ValueError(f"缺少 OHLCV 列 {missing}，无法生成 TA 特征。")
        work = work[base_cols + (["amount"] if "amount" in work.columns else [])].copy()
        before_cols = set(work.columns)
        work = add_ta_indicators(work)
        ta_cols = [c for c in _numeric_cols(work) if c not in base_cols and c != "amount"]
        return work, ta_cols

    # hf / ta+hf：保留 CSV 的全部数值列（排除少数明显“非特征/标识”列）
    exclude_non_feature = {
        "datetime",
        "bar_end_time",
        "feature_time",
        "label",
        "future_return",
    }
    hf_cols = [c for c in _numeric_cols(work) if c not in exclude_non_feature and c not in base_cols]

    if feature_set == "hf":
        return work, hf_cols

    # ta+hf
    before_cols = set(work.columns)
    work = add_ta_indicators(work)
    ta_cols = [c for c in _numeric_cols(work) if c not in base_cols and c not in exclude_non_feature]
    # ta_cols 里会包含 hf_cols；这里做并集去重即可
    union_cols = list(dict.fromkeys(hf_cols + ta_cols))
    return work, union_cols


def build_sequences_with_pipeline(
    df_labeled: pd.DataFrame,
    feature_cols: list[str],
    seq_length: int,
    max_features: int,
    do_select: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """复用当前项目的 FeatureEngineer：特征选择 + 标准化 + 序列构造。"""
    fe = FeatureEngineer()
    fe.feature_names = feature_cols

    train_end_idx = int(len(df_labeled) * 0.7)
    if do_select:
        selected = fe.select_features(df_labeled, max_features=int(max_features), train_end_idx=train_end_idx)
        used = selected
    else:
        used = feature_cols
        fe.selected_features = used

    X, y, _ = fe.create_sequences(df_labeled, feature_cols=used, seq_length=int(seq_length), train_end_idx=train_end_idx)
    return X, y, used


def _seq_to_tabular_fast(X_rows: np.ndarray, seq_length: int) -> np.ndarray:
    """把 (N_rows, F) 快速转为 (N_samples, 3F)：last + mean + std（窗口长度=seq_length）。

    这里的样本对齐是“窗口末尾对齐”：
      - window 末尾行 index = seq_length-1 .. N_rows-1
      - 生成的样本数 n_samples = N_rows - seq_length + 1
    """
    X = np.asarray(X_rows, dtype=np.float32)
    n_rows, n_feat = X.shape
    seq_length = int(seq_length)
    n_samples = n_rows - seq_length + 1
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    # 累积和/平方和用 float64 降低数值误差
    csum = np.cumsum(X.astype(np.float64), axis=0)
    csum2 = np.cumsum((X.astype(np.float64) ** 2), axis=0)

    end = np.arange(seq_length - 1, seq_length - 1 + n_samples)  # window 的最后一行 index
    start_prev = end - seq_length  # window 起点的前一行

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
    tab = np.concatenate([last, mean.astype(np.float32), std.astype(np.float32)], axis=1).astype(np.float32)
    return tab


def build_tabular_with_rolling(
    df_labeled: pd.DataFrame,
    feature_cols: list[str],
    seq_length: int,
    max_features: int,
    do_select: bool,
    select_method: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """内存友好的 GBDT 数据构造：不生成 3D 序列，直接构造 last/mean/std 的 2D 特征矩阵。"""
    work = df_labeled.copy()
    fe = FeatureEngineer()
    fe.feature_names = feature_cols

    # 先按“行”做特征选择（与窗口末尾对齐的标签口径一致）
    train_end_row = int(len(work) * 0.7)
    if do_select:
        used = fe.select_features(
            work, method=select_method, max_features=int(max_features), train_end_idx=train_end_row
        )
    else:
        used = feature_cols

    X_rows = work[used].values
    y_rows = work["label"].values.astype(np.int64)
    X_rows = np.nan_to_num(X_rows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 样本数量按窗口定义
    seq_length = int(seq_length)
    n_rows = len(X_rows)
    n_samples = n_rows - seq_length + 1
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    # 划分按“样本”比例，再映射回“行”确定 scaler 的 fit 截止，避免泄露
    train_end_s = int(n_samples * 0.7)
    train_end_row2 = train_end_s + seq_length - 1

    scaler = RobustScaler()
    scaler.fit(X_rows[:train_end_row2 + 1])
    X_scaled = scaler.transform(X_rows).astype(np.float32)
    X_scaled = np.clip(X_scaled, -5, 5)

    X_tab = _seq_to_tabular_fast(X_scaled, seq_length=seq_length)
    y = y_rows[seq_length - 1:]
    return X_tab, y, used


def _predict_probs(trainer: ModelTrainer, model_type: str, model, X: np.ndarray) -> np.ndarray:
    if model_type in ("lgbm", "xgboost"):
        return model.predict_proba(X)
    import torch
    model.eval()
    out = []
    bs = 512
    with torch.no_grad():
        for i in range(0, len(X), bs):
            batch = torch.FloatTensor(X[i:i + bs]).to(trainer.device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            out.append(probs)
    return np.concatenate(out)


def run_experiment(
    csv_path: str,
    feature_sets: list[str],
    horizons: list[int],
    models: list[str],
    seq_length: int = 60,
    max_features: int = MAX_FEATURES,
    no_select: bool = False,
    tabular: bool = False,
    select_method: str = "mutual_info",
):
    """运行不同特征集与不同 horizon 的对照实验。"""
    print(f"加载数据: {csv_path}")
    df0 = load_hf_data(csv_path)
    print(f"数据范围: {len(df0)} rows, {df0['datetime'].iloc[0]} ~ {df0['datetime'].iloc[-1]}")

    results = []
    trainer = ModelTrainer()

    gbdt_only = all(mt in ("lgbm", "xgboost") for mt in models)
    use_tabular = bool(tabular or gbdt_only)
    if use_tabular and not gbdt_only:
        raise ValueError("tabular 模式仅支持 lgbm/xgboost（深度模型需要 3D 序列输入）。")
    if use_tabular:
        print("表示形式: tabular(last/mean/std)（更省内存，适合 CPU/小内存机器）")
    else:
        print("表示形式: 3D 序列（更占内存，且深度模型在无 GPU 时会很慢）")

    for feature_set in feature_sets:
        print(f"\n{'='*90}")
        print(f"特征集: {feature_set}")
        print(f"{'='*90}")
        df_fs, raw_feature_cols = build_feature_columns(df0, feature_set)
        print(f"候选特征数: {len(raw_feature_cols)}")

        for horizon in horizons:
            print(f"\n{'-'*70}")
            print(f"Horizon: {horizon} bars, seq_len={seq_length}, select={'否' if no_select else '是'}")
            print(f"{'-'*70}")

            df_labeled = add_label(df_fs, horizon)
            if len(df_labeled) < 500:
                print(f"样本过少({len(df_labeled)}), 跳过")
                continue

            if use_tabular:
                X, y, used_cols = build_tabular_with_rolling(
                    df_labeled,
                    raw_feature_cols,
                    seq_length=seq_length,
                    max_features=max_features,
                    do_select=(not no_select),
                    select_method=select_method,
                )
            else:
                X, y, used_cols = build_sequences_with_pipeline(
                    df_labeled,
                    raw_feature_cols,
                    seq_length=seq_length,
                    max_features=max_features,
                    do_select=(not no_select),
                )
            n = len(X)
            if n < 500:
                print(f"序列样本过少({n}), 跳过")
                continue

            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            X_test, y_test = X[val_end:], y[val_end:]

            print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
            print(f"Label(up) test={y_test.mean()*100:.1f}%, 使用特征={len(used_cols)}")

            row = {
                "feature_set": feature_set,
                "horizon": int(horizon),
                "seq_length": int(seq_length),
                "n_features": int(len(used_cols)),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
                "test_up_ratio": float(y_test.mean()),
            }

            for mt in models:
                print(f"\n训练模型: {mt}")
                t0 = time.time()
                try:
                    model, metrics = trainer.train_model(mt, X_train, y_train, X_val, y_val)
                    test_probs = _predict_probs(trainer, mt, model, X_test)
                    test_auc = roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else 0.5
                    test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))
                    row[f"{mt}_val_auc"] = float(metrics.get("val_auc", 0))
                    row[f"{mt}_test_auc"] = float(test_auc)
                    row[f"{mt}_test_acc"] = float(test_acc)
                    row[f"{mt}_time_s"] = round(time.time() - t0, 2)
                    print(f"{mt}: val_auc={row[f'{mt}_val_auc']:.4f}, test_auc={test_auc:.4f}, time={row[f'{mt}_time_s']:.2f}s")
                except Exception as e:
                    row[f"{mt}_val_auc"] = 0.0
                    row[f"{mt}_test_auc"] = 0.0
                    row[f"{mt}_test_acc"] = 0.0
                    row[f"{mt}_time_s"] = round(time.time() - t0, 2)
                    print(f"{mt} 失败: {e}")

            results.append(row)

    # Summary
    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "=" * 110)
        print("RESULTS SUMMARY")
        print("=" * 110)
        # 让输出更紧凑可读
        pd.set_option("display.max_columns", 50)
        pd.set_option("display.width", 240)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(df_res.sort_values(["feature_set", "horizon"]).to_string(index=False))

        # Save results
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'results', 'experiments')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "btc_hf_results.csv")
        df_res.to_csv(save_path, index=False)
        print(f"\n结果已保存: {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="BTC 高频特征适配实验")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="高频特征 CSV 路径")
    parser.add_argument("--feature-sets", default="ta,hf,ta+hf", help="逗号分隔: ta,hf,ta+hf")
    parser.add_argument("--horizons", default="1,24,48", help="逗号分隔 horizon（单位=bar）")
    parser.add_argument("--models", default="lgbm,xgboost", help="逗号分隔模型类型")
    parser.add_argument("--seq-length", type=int, default=60, help="序列长度（bar 数）")
    parser.add_argument("--max-features", type=int, default=MAX_FEATURES, help="特征选择保留上限")
    parser.add_argument("--no-select", action="store_true", help="关闭特征选择，直接使用全量候选特征")
    parser.add_argument("--tabular", action="store_true",
                        help="使用 2D tabular(last/mean/std) 表示（推荐在 CPU/小内存机器上跑 lgbm/xgboost）")
    parser.add_argument("--select-method", default="mutual_info",
                        choices=["mutual_info", "f_classif", "random_forest"],
                        help="特征选择方法（只影响 --no-select 未开启时）")
    args = parser.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    feature_sets = [x.strip() for x in args.feature_sets.split(",") if x.strip()]

    run_experiment(
        csv_path=args.csv,
        feature_sets=feature_sets,
        horizons=horizons,
        models=models,
        seq_length=int(args.seq_length),
        max_features=int(args.max_features),
        no_select=bool(args.no_select),
        tabular=bool(args.tabular),
        select_method=str(args.select_method),
    )
