"""
预测周期扫描实验
测试不同PREDICT_HORIZON对模型AUC的影响
只用LightGBM (最快) + transformer_lstm
"""
import os
import sys
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.lgbm_model import LGBMModel
from models.trainer import ModelTrainer
from config.settings import LGBM_CONFIG, SEQUENCE_LENGTH, MAX_FEATURES
from utils.helpers import set_seed
from sklearn.metrics import roc_auc_score, accuracy_score

set_seed(42)

# 实验参数
HORIZONS = [3, 6, 12, 24, 48, 96]  # 15min, 30min, 1hr, 2hr, 1day, 2day
MODEL_TYPES = ["lgbm", "transformer_lstm"]


def run_experiment(df_raw, horizon, model_types):
    """对单个horizon运行实验"""
    import config.settings as cfg
    orig_horizon = cfg.PREDICT_HORIZON
    cfg.PREDICT_HORIZON = horizon

    fe = FeatureEngineer()

    # 构建特征 (用修改后的horizon生成label)
    df = df_raw.copy()
    df = fe._add_ma(df)
    df = fe._add_ema(df)
    df = fe._add_rsi(df)
    df = fe._add_macd(df)
    df = fe._add_bollinger(df)
    df = fe._add_atr(df)
    df = fe._add_kdj(df)
    df = fe._add_cci(df)
    df = fe._add_williams(df)
    df = fe._add_volume_indicators(df)
    df = fe._add_price_features(df)
    df = fe._add_time_features(df)
    df = fe._add_multiscale_features(df)

    # 标签
    df['future_return'] = df['close'].shift(-horizon) / df['close'] - 1
    df['label'] = (df['future_return'] > 0).astype(int)

    fe.feature_names = [c for c in df.columns if c not in
                       ['datetime', 'date', 'time', 'open', 'high', 'low', 'close',
                        'volume', 'amount', 'future_return', 'label',
                        'day_of_week', 'hour', 'minute', 'vwap']]

    df = df.dropna().reset_index(drop=True)

    # 收益率统计
    ret = df['future_return']
    ret_stats = {
        'median_abs_ret': ret.abs().median() * 100,
        'pct_gt_05': (ret.abs() > 0.005).mean() * 100,
        'pct_gt_10': (ret.abs() > 0.01).mean() * 100,
        'ret_std': ret.std() * 100,
        'label_ratio': df['label'].mean(),
        'n_samples': len(df),
    }

    # 特征选择 (只用训练集)
    train_end = int(len(df) * 0.7)
    fe.select_features(df, max_features=MAX_FEATURES, train_end_idx=train_end)
    X, y, ts = fe.create_sequences(df, train_end_idx=train_end)

    if len(X) < 500:
        cfg.PREDICT_HORIZON = orig_horizon
        return None

    # 70/15/15 split
    n = len(X)
    train_end_seq = int(n * 0.7)
    val_end_seq = int(n * 0.85)

    X_train, y_train = X[:train_end_seq], y[:train_end_seq]
    X_val, y_val = X[train_end_seq:val_end_seq], y[train_end_seq:val_end_seq]
    X_test, y_test = X[val_end_seq:], y[val_end_seq:]

    results = {'horizon': horizon, 'horizon_min': horizon * 5, **ret_stats}

    trainer = ModelTrainer()

    for mt in model_types:
        t0 = time.time()
        try:
            model, metrics = trainer.train_model(mt, X_train, y_train, X_val, y_val)

            # 测试集评估
            if mt == "lgbm":
                test_probs = model.predict_proba(X_test)
            else:
                import torch
                model.eval()
                with torch.no_grad():
                    test_probs = torch.sigmoid(
                        model(torch.FloatTensor(X_test).to(trainer.device))
                    ).cpu().numpy()

            test_auc = roc_auc_score(y_test, test_probs) if len(set(y_test)) > 1 else 0.5
            test_acc = accuracy_score(y_test, (test_probs >= 0.5).astype(int))
            elapsed = time.time() - t0

            results[f'{mt}_val_auc'] = metrics.get('val_auc', 0)
            results[f'{mt}_test_auc'] = test_auc
            results[f'{mt}_test_acc'] = test_acc
            results[f'{mt}_time_s'] = round(elapsed, 1)
        except Exception as e:
            print(f"  {mt} failed: {e}")
            results[f'{mt}_val_auc'] = 0
            results[f'{mt}_test_auc'] = 0
            results[f'{mt}_test_acc'] = 0
            results[f'{mt}_time_s'] = 0

    cfg.PREDICT_HORIZON = orig_horizon
    return results


def main():
    print("=" * 70)
    print("预测周期扫描实验")
    print("=" * 70)

    loader = DataLoader()
    df_raw = loader.prepare_data()
    print(f"数据量: {len(df_raw)}, 时间: {df_raw['datetime'].iloc[0]} ~ {df_raw['datetime'].iloc[-1]}")

    all_results = []
    for h in HORIZONS:
        print(f"\n{'='*70}")
        print(f"  PREDICT_HORIZON = {h} ({h*5}分钟)")
        print(f"{'='*70}")

        result = run_experiment(df_raw, h, MODEL_TYPES)
        if result:
            all_results.append(result)
            print(f"  收益率中位绝对值: {result['median_abs_ret']:.3f}%")
            print(f"  |ret|>0.5%占比: {result['pct_gt_05']:.1f}%")
            print(f"  |ret|>1.0%占比: {result['pct_gt_10']:.1f}%")
            for mt in MODEL_TYPES:
                k = f'{mt}_test_auc'
                if k in result:
                    print(f"  {mt}: test_auc={result[k]:.4f}, "
                          f"test_acc={result[f'{mt}_test_acc']:.4f}, "
                          f"time={result[f'{mt}_time_s']}s")
        else:
            print(f"  样本不足，跳过")

    # 汇总
    if not all_results:
        print("\n没有有效结果")
        return

    df_results = pd.DataFrame(all_results)

    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)

    summary_cols = ['horizon', 'horizon_min', 'n_samples',
                    'median_abs_ret', 'pct_gt_05', 'pct_gt_10', 'ret_std', 'label_ratio']
    for mt in MODEL_TYPES:
        summary_cols.extend([f'{mt}_test_auc', f'{mt}_test_acc', f'{mt}_time_s'])

    df_display = df_results[[c for c in summary_cols if c in df_results.columns]].copy()
    df_display = df_display.rename(columns={
        'horizon': 'H(bars)', 'horizon_min': 'H(min)', 'n_samples': 'Samples',
        'median_abs_ret': 'Med|Ret|%', 'pct_gt_05': '>0.5%', 'pct_gt_10': '>1.0%',
        'ret_std': 'Ret_Std%', 'label_ratio': 'Up_Ratio',
    })
    for mt in MODEL_TYPES:
        df_display = df_display.rename(columns={
            f'{mt}_test_auc': f'{mt}_AUC',
            f'{mt}_test_acc': f'{mt}_Acc',
            f'{mt}_time_s': f'{mt}_Time',
        })

    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df_display.to_string(index=False))

    # 找最佳horizon
    best_col = f'{MODEL_TYPES[0]}_test_auc'
    best_idx = df_results[best_col].idxmax()
    best = df_results.iloc[best_idx]
    print(f"\n最佳预测周期: {int(best['horizon'])} bars ({int(best['horizon_min'])}分钟)")
    print(f"  LightGBM Test AUC: {best[best_col]:.4f}")
    if f'{MODEL_TYPES[1]}_test_auc' in best:
        print(f"  {MODEL_TYPES[1]} Test AUC: {best[f'{MODEL_TYPES[1]}_test_auc']:.4f}")

    # 保存结果
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'experiments')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'horizon_sweep.csv')
    df_results.to_csv(save_path, index=False)
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    main()
