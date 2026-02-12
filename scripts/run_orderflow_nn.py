"""
一键训练脚本：Orderflow(订单流/盘口代理)微观结构特征 + 深度模型

设计目标
  - 不依赖“历史 L2 orderbook”（无法通过公开 API 回补），使用 Binance Klines 扩展字段做 orderflow 代理
  - 支持大样本（如 5min 5y ~52万根K线）：不把全部序列一次性堆进内存
  - 训练过程打印清晰，结束给出可复用的总结（JSON + 文本）

推荐（BTC/USDT 5min, 5y）：
  python3 -u scripts/run_orderflow_nn.py --freq 5min --days 1825 --model orderflow_tcn
"""
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score

from config.settings import RANDOM_SEED, LGBM_CONFIG, XGB_CONFIG
from utils.helpers import set_seed, ensure_dir
from models.trainer import ModelTrainer, EarlyStopping
from models.lgbm_model import LGBMModel
from models.xgb_model import XGBModel
from experiments.btc_data import (
    load_btc_orderflow,
    add_orderflow_microstructure_features,
    triple_barrier_label,
)


class RollingWindowDataset(Dataset):
    """滚动窗口序列数据集：用 start_idx 表示窗口起点，标签在窗口末尾的下一根 bar。"""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int, start: int, end: int):
        self.X = X
        self.y = y
        self.seq_length = int(seq_length)
        self.start = int(start)
        self.end = int(end)
        if self.start < 0 or self.end < 0 or self.end <= self.start:
            raise ValueError("RollingWindowDataset: start/end 非法")
        if self.end > (len(self.X) - self.seq_length):
            raise ValueError("RollingWindowDataset: end 超出可用样本范围")

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        s = self.start + int(idx)
        x = self.X[s:s + self.seq_length]
        label_idx = s + self.seq_length
        y = self.y[label_idx]
        return torch.from_numpy(x), torch.tensor(float(y), dtype=torch.float32)


def _print_kv(title: str, kv: dict):
    print(f"\n[{title}]")
    for k, v in kv.items():
        print(f"  - {k}: {v}")


def _build_labels(
    df: pd.DataFrame,
    horizon: int,
    label_mode: str,
    pt_sl: tuple[float, float],
    tb_zero_mode: str,
):
    """构造标签。

    关键点：若使用 triple barrier，tb_raw==0 表示“到期未触碰上下轨”。
    如果你要“按涨跌来”，更合理的做法是：tb_raw==0 时用到期收益的正负来决定 0/1（tb_zero_mode=sign）。
    """
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    if label_mode == "triple_barrier":
        tb = triple_barrier_label(df, horizon, pt_sl=pt_sl)
        df["tb_raw"] = tb
        if tb_zero_mode == "as_negative":
            df["label"] = (tb == 1).astype(int)
        elif tb_zero_mode == "sign":
            # -1 -> 0, +1 -> 1, 0 -> 按到期收益涨跌
            base = (tb == 1).astype(int)
            zero_mask = (tb == 0)
            base.loc[zero_mask] = (df.loc[zero_mask, "future_return"] > 0).astype(int)
            df["label"] = base.astype(int)
        else:
            raise ValueError(f"未知 tb_zero_mode: {tb_zero_mode}")
    else:
        df["label"] = (df["future_return"] > 0).astype(int)
    return df


def _add_basic_price_features(df: pd.DataFrame):
    out = df.copy()
    out["ret_1"] = out["close"].pct_change()
    out["log_ret_1"] = np.log(out["close"].astype(float).replace(0, np.nan)).diff()
    out["hl_range"] = (out["high"] - out["low"]) / (out["close"] + 1e-10)
    out["body"] = (out["close"] - out["open"]) / (out["open"] + 1e-10)
    out["log_volume"] = np.log(out["volume"].astype(float) + 1.0)
    if "amount" in out.columns:
        out["log_amount"] = np.log(out["amount"].astype(float) + 1.0)
    return out


def _seq_to_tabular_fast(X: np.ndarray, seq_length: int) -> np.ndarray:
    """快速把 (N_rows, F) 转为 (N_samples, 3F)：last + mean + std（窗口长度=seq_length）。

    这里的样本对齐与 create_sequences 保持一致：
      - window: [i, i+seq_length)
      - label : y[i+seq_length]
    """
    X = np.asarray(X, dtype=np.float32)
    n_rows, n_feat = X.shape
    seq_length = int(seq_length)
    n_samples = n_rows - seq_length
    if n_samples <= 0:
        raise ValueError("seq_length 太大，无法生成样本")

    # 累积和/平方和用 float64 降低数值误差
    csum = np.cumsum(X.astype(np.float64), axis=0)
    csum2 = np.cumsum((X.astype(np.float64) ** 2), axis=0)

    end = np.arange(seq_length - 1, seq_length - 1 + n_samples)  # window 的最后一行 index
    start_prev = np.arange(-1, n_samples - 1)                    # i-1

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

    last = X[end]  # (n_samples, n_feat)
    tab = np.concatenate([last, mean.astype(np.float32), std.astype(np.float32)], axis=1).astype(np.float32)
    return tab


def main():
    parser = argparse.ArgumentParser(description="Orderflow microstructure + Deep NN runner")
    parser.add_argument("--freq", default="5min", choices=["5min", "1h", "daily"])
    parser.add_argument("--days", type=int, default=1825, help="回溯天数（默认 1825≈5年）")

    parser.add_argument("--model", default="orderflow_tcn",
                        choices=["orderflow_tcn", "transformer_lstm", "lstm", "cnn", "mlp", "lgbm", "xgboost"])
    parser.add_argument("--models", default=None,
                        help="逗号分隔的模型列表，如 lgbm,xgboost,orderflow_tcn；不填则用 --model")
    parser.add_argument("--seq-length", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=None, help="预测跨度（bar 数）；不填则按 freq 取默认")
    parser.add_argument("--label-mode", default="triple_barrier", choices=["binary", "triple_barrier"])
    parser.add_argument("--pt-sl", default="1.0,1.0", help="triple barrier 的 pt,sl 倍数（相对波动）")
    parser.add_argument("--tb-zero-mode", default="sign", choices=["sign", "as_negative"],
                        help="triple barrier 下 tb_raw==0 的处理：sign=按到期涨跌；as_negative=当作负类")

    parser.add_argument("--vpin-window", type=int, default=50)
    parser.add_argument("--kyle-window", type=int, default=50)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--clip", type=float, default=5.0, help="缩放后特征截断阈值（默认±5）")
    parser.add_argument("--gbdt-rounds", type=int, default=2000)
    parser.add_argument("--gbdt-early-stop", type=int, default=100)

    parser.add_argument("--gap", type=int, default=None,
                        help="时间切分 gap（防止标签重叠泄露，单位=样本数）；不填则自动取 horizon")
    parser.add_argument("--out-dir", default=os.path.join("results", "orderflow_runs"))
    args = parser.parse_args()

    set_seed(RANDOM_SEED)

    if args.horizon is None:
        if args.freq == "daily":
            args.horizon = 3
        elif args.freq == "1h":
            args.horizon = 24
        else:
            args.horizon = 12
    if args.gap is None:
        args.gap = int(args.horizon)

    pt_sl = tuple(float(x) for x in args.pt_sl.split(","))
    ensure_dir(args.out_dir)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.out_dir, f"{args.freq}_{args.model}_{run_id}")
    ensure_dir(run_dir)

    cfg = {
        "freq": args.freq,
        "days": args.days,
        "model": args.model,
        "seq_length": args.seq_length,
        "horizon": args.horizon,
        "label_mode": args.label_mode,
        "pt_sl": pt_sl,
        "tb_zero_mode": args.tb_zero_mode,
        "vpin_window": args.vpin_window,
        "kyle_window": args.kyle_window,
        "train": {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "clip": args.clip,
            "gap": int(args.gap),
        }
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ============ 数据加载/准备 ============
    t0 = time.time()
    try:
        df = load_btc_orderflow(args.freq)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("[ERROR] 没有找到本地 orderflow 数据文件。若你已推送到仓库，请先在服务器执行 `git pull`。")
        print("[ERROR] 若需要重新生成，可用：")
        print("  - daily:  python3 -u experiments/btc_data.py --freq daily --fetch-daily-orderflow --days 1825")
        print("  - 1h:     python3 -u experiments/btc_data.py --freq 1h --fetch-1h-orderflow --days 1825")
        print("  - 5min:   python3 -u experiments/btc_data.py --freq 5min --fetch-5min-orderflow --days 1825")
        raise SystemExit(2)

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_localize(None)
    df = df.sort_values("datetime").reset_index(drop=True)
    if args.days and args.days > 0:
        end_dt = pd.to_datetime(df["datetime"].iloc[-1])
        start_dt = end_dt - pd.Timedelta(days=int(args.days))
        df = df[df["datetime"] >= start_dt].reset_index(drop=True)

    # 保证微观结构特征齐全（可覆盖不同窗口配置）
    if "bar_duration_sec" not in df.columns:
        if args.freq == "daily":
            df["bar_duration_sec"] = 86400.0
        elif args.freq == "1h":
            df["bar_duration_sec"] = 3600.0
        else:
            df["bar_duration_sec"] = 300.0
    df = add_orderflow_microstructure_features(
        df,
        vpin_window=int(args.vpin_window),
        kyle_window=int(args.kyle_window),
    )
    df = _add_basic_price_features(df)
    df = _build_labels(df, int(args.horizon), args.label_mode, pt_sl, args.tb_zero_mode)

    # 特征列：所有数值列，排除标签/未来信息/时间
    exclude = {"datetime", "label", "future_return", "tb_raw"}
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = numeric_cols

    # 清洗
    df_feat = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
    n_rows = len(df_feat)
    if n_rows <= args.seq_length + 50:
        raise ValueError(f"可用样本不足：rows={n_rows}, seq_length={args.seq_length}")

    X = df_feat[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y = df_feat["label"].to_numpy(dtype=np.int8, copy=True)

    n_samples = n_rows - int(args.seq_length)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)
    gap = int(max(int(args.gap), 0))
    val_start = min(val_end, train_end + gap)
    test_start = min(n_samples, val_end + gap)
    if n_samples < 200 or train_end < 100 or (val_end - val_start) < 20 or (n_samples - test_start) < 20:
        print("\n[WARN] 样本较少，训练/验证/测试切分可能不稳定；建议：减小 --seq-length 或增大 --days。")

    _print_kv("数据概览", {
        "freq": args.freq,
        "bars": n_rows,
        "range": f"{df_feat['datetime'].iloc[0]} ~ {df_feat['datetime'].iloc[-1]}",
        "features": len(feature_cols),
        "seq_length": args.seq_length,
        "samples(total)": n_samples,
        "gap": gap,
        "split": f"train=0:{train_end}, val={val_start}:{val_end}, test={test_start}:{n_samples}",
        "label(up%)": f"{y.mean() * 100:.2f}%",
        "label_mode": args.label_mode,
        "horizon": args.horizon,
    })

    # ============ 归一化（只用训练区间的行，避免泄露） ============
    scaler = RobustScaler()
    fit_end_row = int(args.seq_length) + int(train_end)
    scaler.fit(X[:fit_end_row])
    X = scaler.transform(X)
    if args.clip and args.clip > 0:
        X = np.clip(X, -float(args.clip), float(args.clip))

    trainer = ModelTrainer()
    pin_memory = (trainer.device.type == "cuda")
    model_list = [args.model] if not args.models else [m.strip() for m in args.models.split(",") if m.strip()]

    need_gbdt = any(m in ("lgbm", "xgboost") for m in model_list)
    X_tab_all = None
    y_samp = y[int(args.seq_length):int(args.seq_length) + n_samples].astype(np.int8, copy=False)
    if need_gbdt:
        print("\n[构造GBDT表格特征] last+mean+std (避免 3D 序列占用内存)")
        X_tab_all = _seq_to_tabular_fast(X, int(args.seq_length))

    all_summaries = []
    results_rows = []

    for mt in model_list:
        set_seed(RANDOM_SEED)
        mt_dir = os.path.join(run_dir, "models", mt)
        ensure_dir(mt_dir)
        mt_t0 = time.time()

        if mt in ("lgbm", "xgboost"):
            X_samp = X_tab_all
            X_train = X_samp[:train_end]
            y_train = y_samp[:train_end]
            X_val = X_samp[val_start:val_end]
            y_val = y_samp[val_start:val_end]
            X_test = X_samp[test_start:n_samples]
            y_test = y_samp[test_start:n_samples]

            if mt == "lgbm":
                model = LGBMModel(**LGBM_CONFIG)
                model.fit(X_train, y_train, X_val, y_val,
                          num_boost_round=int(args.gbdt_rounds),
                          early_stopping_rounds=int(args.gbdt_early_stop))
                val_probs = model.predict_proba(X_val)
                test_probs = model.predict_proba(X_test)
                import lightgbm as lgb
                model_path = os.path.join(mt_dir, "lgbm.txt")
                model.model.save_model(model_path)
            else:
                model = XGBModel(**XGB_CONFIG)
                model.fit(X_train, y_train, X_val, y_val,
                          num_boost_round=int(args.gbdt_rounds),
                          early_stopping_rounds=int(args.gbdt_early_stop))
                val_probs = model.predict_proba(X_val)
                test_probs = model.predict_proba(X_test)
                model_path = os.path.join(mt_dir, "xgb.json")
                model.model.save_model(model_path)

            val_auc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.5
            val_acc = accuracy_score(y_val, (np.array(val_probs) >= 0.5).astype(int))
            test_auc = roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else 0.5
            test_acc = accuracy_score(y_test, (np.array(test_probs) >= 0.5).astype(int))

            mt_elapsed = round(time.time() - mt_t0, 1)
            summary = {
                "model": mt,
                "val_auc": float(val_auc),
                "val_acc": float(val_acc),
                "test_auc": float(test_auc),
                "test_acc": float(test_acc),
                "elapsed_sec": mt_elapsed,
                "artifact": model_path,
            }
            all_summaries.append(summary)
            results_rows.append(summary)
            print(f"\n[{mt}] val_auc={val_auc:.4f} val_acc={val_acc:.4f} test_auc={test_auc:.4f} test_acc={test_acc:.4f} time={mt_elapsed:.1f}s")
            continue

        # ============ 深度模型：DataLoader（不构造全量 3D 张量） ============
        train_ds = RollingWindowDataset(X, y, args.seq_length, 0, train_end)
        val_ds = RollingWindowDataset(X, y, args.seq_length, val_start, val_end)
        test_ds = RollingWindowDataset(X, y, args.seq_length, test_start, n_samples)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True, pin_memory=pin_memory)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, drop_last=False, pin_memory=pin_memory)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, drop_last=False, pin_memory=pin_memory)

        model = trainer._create_model(mt, input_size=X.shape[1], seq_length=args.seq_length)

        pos_weight = trainer._compute_pos_weight(y[:fit_end_row])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        early = EarlyStopping(patience=args.patience, min_delta=1e-4, mode="max")

        history = []
        best_val_auc = 0.0
        best_epoch = 0
        print(f"\n[训练开始] device={trainer.device}, model={mt}, batch={args.batch_size}, epochs={args.max_epochs}")

        for epoch in range(1, int(args.max_epochs) + 1):
            ep0 = time.time()
            train_loss, train_auc = trainer.train_epoch(model, train_loader, criterion, optimizer, clip_norm=1.0)
            val_loss, val_auc, val_acc, _, _ = trainer.evaluate(model, val_loader, criterion)
            scheduler.step(val_auc)

            history.append({
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_auc": float(train_auc),
                "val_loss": float(val_loss),
                "val_auc": float(val_auc),
                "val_acc": float(val_acc),
                "sec": round(time.time() - ep0, 2),
            })

            if val_auc > best_val_auc:
                best_val_auc = float(val_auc)
                best_epoch = epoch

            if epoch == 1 or epoch % 5 == 0:
                print(f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
                      f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} val_acc={val_acc:.4f} "
                      f"time={history[-1]['sec']:.2f}s")

            early(val_auc, model)
            if early.early_stop:
                print(f"[早停] epoch={epoch}, best_val_auc={early.best_score:.4f}")
                break

        if early.best_model_state:
            model.load_state_dict(early.best_model_state)

        test_loss, test_auc, test_acc, _, _ = trainer.evaluate(model, test_loader, criterion)
        mt_elapsed = round(time.time() - mt_t0, 1)

        # artifacts
        torch.save(model.state_dict(), os.path.join(mt_dir, "model_state.pt"))
        pd.DataFrame(history).to_csv(os.path.join(mt_dir, "history.csv"), index=False)

        summary = {
            "model": mt,
            "val_best_auc": float(best_val_auc),
            "val_best_epoch": int(best_epoch),
            "test_auc": float(test_auc),
            "test_acc": float(test_acc),
            "test_loss": float(test_loss),
            "elapsed_sec": mt_elapsed,
            "artifact": f"{mt_dir}/model_state.pt",
        }
        with open(os.path.join(mt_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        all_summaries.append(summary)
        results_rows.append(summary)

        print(f"[{mt}] val_best_auc={best_val_auc:.4f} (epoch={best_epoch}) "
              f"test_auc={test_auc:.4f} test_acc={test_acc:.4f} time={mt_elapsed:.1f}s")

    elapsed = round(time.time() - t0, 1)
    pd.DataFrame(results_rows).to_csv(os.path.join(run_dir, "results.csv"), index=False)
    with open(os.path.join(run_dir, "summary_all.json"), "w", encoding="utf-8") as f:
        json.dump({
            "run_dir": run_dir,
            "elapsed_sec": elapsed,
            "models": model_list,
            "bars": int(n_rows),
            "samples": int(n_samples),
            "features": int(len(feature_cols)),
            "seq_length": int(args.seq_length),
            "horizon": int(args.horizon),
            "label_mode": args.label_mode,
            "results": all_summaries,
        }, f, ensure_ascii=False, indent=2)

    _print_kv("总体总结", {
        "models": ",".join(model_list),
        "elapsed_sec": elapsed,
        "artifacts": f"{run_dir}/(config.json, results.csv, summary_all.json, models/...)",
    })


if __name__ == "__main__":
    main()
