"""
序列长度扫描实验
测试不同SEQUENCE_LENGTH对模型AUC的影响
固定PREDICT_HORIZON=12 (1小时), 只用LightGBM + transformer_lstm
"""
import os
import sys
import numpy as np
import pandas as pd
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.trainer import ModelTrainer
from config.settings import MAX_FEATURES
from utils.helpers import set_seed
from sklearn.metrics import roc_auc_score, accuracy_score

set_seed(42)

# 实验参数
SEQ_LENGTHS = [15, 30, 60, 120, 240]  # 1.25hr, 2.5hr, 5hr, 10hr, 20hr
MODEL_TYPES = ["lgbm", "transformer_lstm"]


def run_experiment(df_featured, fe, seq_len, model_types):
    """对单个seq_len运行实验"""
    import config.settings as cfg
    orig_seq = cfg.SEQUENCE_LENGTH
    cfg.SEQUENCE_LENGTH = seq_len

    # 特征选择 (只用训练集)
    train_end = int(len(df_featured) * 0.7)
    fe_copy = FeatureEngineer()
    fe_copy.feature_names = fe.feature_names.copy()
    fe_copy.select_features(df_featured, max_features=MAX_FEATURES, train_end_idx=train_end)
    X, y, ts = fe_copy.create_sequences(df_featured, train_end_idx=train_end)

    cfg.SEQUENCE_LENGTH = orig_seq

    if len(X) < 500:
        return None

    # 70/15/15 split
    n = len(X)
    train_end_seq = int(n * 0.7)
    val_end_seq = int(n * 0.85)

    X_train, y_train = X[:train_end_seq], y[:train_end_seq]
    X_val, y_val = X[train_end_seq:val_end_seq], y[train_end_seq:val_end_seq]
    X_test, y_test = X[val_end_seq:], y[val_end_seq:]

    results = {
        'seq_len': seq_len,
        'seq_hours': seq_len * 5 / 60,
        'n_samples': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X.shape[2],
    }

    trainer = ModelTrainer()

    for mt in model_types:
        t0 = time.time()
        try:
            model, metrics = trainer.train_model(mt, X_train, y_train, X_val, y_val)

            # 测试集评估
            if mt in ("lgbm", "xgboost"):
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

    return results


def main():
    print("=" * 70)
    print("序列长度扫描实验 (PREDICT_HORIZON=12, 1小时)")
    print("=" * 70)

    loader = DataLoader()
    df_raw = loader.prepare_data()
    print(f"数据量: {len(df_raw)}")

    # 构建特征一次 (特征不依赖seq_len)
    fe = FeatureEngineer()
    df_featured = fe.build_features(df_raw)
    print(f"特征数量: {len(fe.feature_names)}")

    all_results = []
    for sl in SEQ_LENGTHS:
        print(f"\n{'='*70}")
        print(f"  SEQUENCE_LENGTH = {sl} ({sl*5/60:.1f} 小时)")
        print(f"{'='*70}")

        result = run_experiment(df_featured, fe, sl, MODEL_TYPES)
        if result:
            all_results.append(result)
            print(f"  样本量: {result['n_samples']}, 特征: {result['n_features']}")
            for mt in MODEL_TYPES:
                k = f'{mt}_test_auc'
                if k in result and result[k] > 0:
                    print(f"  {mt}: test_auc={result[k]:.4f}, "
                          f"val_auc={result[f'{mt}_val_auc']:.4f}, "
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

    summary_cols = ['seq_len', 'seq_hours', 'n_samples']
    for mt in MODEL_TYPES:
        summary_cols.extend([f'{mt}_test_auc', f'{mt}_val_auc', f'{mt}_time_s'])

    df_display = df_results[[c for c in summary_cols if c in df_results.columns]].copy()
    df_display = df_display.rename(columns={
        'seq_len': 'SeqLen', 'seq_hours': 'Hours', 'n_samples': 'Samples',
    })
    for mt in MODEL_TYPES:
        df_display = df_display.rename(columns={
            f'{mt}_test_auc': f'{mt}_TestAUC',
            f'{mt}_val_auc': f'{mt}_ValAUC',
            f'{mt}_time_s': f'{mt}_Time',
        })

    pd.set_option('display.max_columns', 30)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(df_display.to_string(index=False))

    # 找最佳
    for mt in MODEL_TYPES:
        col = f'{mt}_test_auc'
        if col in df_results.columns:
            best_idx = df_results[col].idxmax()
            best = df_results.iloc[best_idx]
            print(f"\n{mt} 最佳序列长度: {int(best['seq_len'])} ({best['seq_hours']:.1f}小时), Test AUC={best[col]:.4f}")

    # 保存结果
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'experiments')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'seqlen_sweep.csv')
    df_results.to_csv(save_path, index=False)
    print(f"\n结果已保存: {save_path}")


if __name__ == "__main__":
    main()
