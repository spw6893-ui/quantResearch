"""
滚动窗口回测系统
模拟真实交易环境: 定期重新训练模型 + 滚动预测
每个窗口重新检测动态周期、重新选特征、重新标准化
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    BACKTEST_CONFIG, TRAINING_CONFIG, SEQUENCE_LENGTH, T0_CONFIG,
    TRANSFORMER_LSTM_CONFIG, LSTM_CONFIG, CNN_CONFIG, MLP_CONFIG,
    ENSEMBLE_WEIGHTS, DEVICE, RANDOM_SEED, RESULTS_DIR, MODEL_DIR,
    MAX_FEATURES, DYNAMIC_PERIODS_ENABLED
)

T0_CONFIG_THRESHOLD = T0_CONFIG['signal_threshold']
from models.transformer_lstm import TransformerLSTM
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from models.ensemble import EnsembleModel
from models.trainer import ModelTrainer
from data.feature_engineering import FeatureEngineer
from backtesting.backtest_engine import BacktestEngine
from utils.logger import get_logger
from utils.helpers import set_seed, ensure_dir

logger = get_logger(__name__, "rolling_backtest.log")


class RollingBacktest:
    """
    滚动窗口回测:
    1. 使用前N天数据训练模型 (每窗口重新检测周期)
    2. 在后续M天上进行预测和交易
    3. 滚动前进，重新训练
    """

    def __init__(self, train_window_days: int = None,
                 retrain_interval_days: int = None):
        self.train_window = train_window_days or BACKTEST_CONFIG['rolling_window_days']
        self.retrain_interval = retrain_interval_days or BACKTEST_CONFIG['retrain_interval_days']
        self.results_dir = os.path.join(RESULTS_DIR, "rolling_backtest")
        ensure_dir(self.results_dir)
        set_seed(RANDOM_SEED)

    def _train_ensemble(self, trainer, X_tr, y_tr, X_val, y_val, quick_config):
        """训练全部4个模型并返回集成"""
        models = {}

        for mt in ["transformer_lstm", "lstm", "cnn", "mlp"]:
            try:
                model, metrics = trainer.train_model(
                    mt, X_tr, y_tr, X_val, y_val, config=quick_config
                )
                models[mt] = model
                logger.info(f"    {mt}: val_auc={metrics.get('val_auc', 0):.4f}")
            except Exception as e:
                logger.warning(f"    {mt} 训练失败: {e}")

        if not models:
            return None

        ensemble = EnsembleModel(models, ENSEMBLE_WEIGHTS, DEVICE)
        return ensemble

    def run(self, df_featured: pd.DataFrame, prices_df: pd.DataFrame,
            fe: FeatureEngineer = None) -> dict:
        """
        执行滚动回测 (每个窗口重新检测动态周期)

        Args:
            df_featured: 已构建静态特征的DataFrame (来自build_features, 含label列)
            prices_df: 价格DataFrame (含close列)
            fe: FeatureEngineer实例 (用于重建动态特征/特征选择/序列构建)
        """
        if fe is None:
            fe = FeatureEngineer()
            fe.feature_names = [c for c in df_featured.columns if c not in
                               ['datetime', 'date', 'time', 'open', 'high', 'low', 'close',
                                'volume', 'amount', 'future_return', 'label',
                                'day_of_week', 'hour', 'minute', 'vwap']]

        logger.info(f"滚动回测: 训练窗口={self.train_window}天, "
                    f"重训间隔={self.retrain_interval}天")

        bars_per_day = 48
        train_size = self.train_window * bars_per_day
        retrain_size = self.retrain_interval * bars_per_day
        n_total = len(df_featured)

        if train_size >= n_total:
            logger.error(f"训练窗口({train_size})大于总数据量({n_total})")
            return {}

        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_timestamps = []
        window_results = []

        trainer = ModelTrainer()
        window_idx = 0
        start = 0

        from sklearn.metrics import roc_auc_score, accuracy_score

        while start + train_size + retrain_size <= n_total:
            train_end = start + train_size
            test_end = min(train_end + retrain_size, n_total)

            logger.info(f"\n--- 滚动窗口 {window_idx + 1} ---")
            logger.info(f"  数据范围: [{start}:{test_end}], 训练截止: {train_end}")

            # 取当前窗口的df切片 (train + test)
            df_window = df_featured.iloc[start:test_end].copy().reset_index(drop=True)
            window_train_end = train_end - start  # 窗口内的训练截止索引

            # 每个窗口重新检测动态周期 (只用训练数据)
            if DYNAMIC_PERIODS_ENABLED:
                df_window = fe.rebuild_dynamic_features(df_window, train_end_idx=window_train_end)
                detected = getattr(fe, '_detected_periods', [])
                if detected:
                    logger.info(f"  窗口检测到周期: {detected}")

            # 特征选择 (只用训练数据)
            fe.select_features(df_window, max_features=MAX_FEATURES,
                              train_end_idx=window_train_end)

            # 创建序列 (scaler只在训练数据上fit)
            X_win, y_win, ts_win = fe.create_sequences(
                df_window, train_end_idx=window_train_end
            )

            if len(X_win) == 0:
                logger.warning(f"  窗口 {window_idx+1} 序列为空，跳过")
                start += retrain_size
                window_idx += 1
                continue

            # 把序列分成训练和测试部分
            seq_len = X_win.shape[1]
            n_train_seq = max(window_train_end - seq_len, 0)
            if n_train_seq >= len(X_win):
                n_train_seq = len(X_win) - 1

            X_train_seq = X_win[:n_train_seq]
            y_train_seq = y_win[:n_train_seq]
            X_test_seq = X_win[n_train_seq:]
            y_test_seq = y_win[n_train_seq:]
            ts_test_seq = ts_win[n_train_seq:]

            if len(X_train_seq) < 100 or len(X_test_seq) < 10:
                logger.warning(f"  窗口 {window_idx+1} 训练/测试样本不足，跳过")
                start += retrain_size
                window_idx += 1
                continue

            # 训练 (80/20 分割为 train/val)
            split = int(len(X_train_seq) * 0.8)
            X_tr = X_train_seq[:split]
            y_tr = y_train_seq[:split]
            X_val = X_train_seq[split:]
            y_val = y_train_seq[split:]

            # 快速训练配置
            quick_config = {
                **TRAINING_CONFIG,
                'max_epochs': 30,
                'early_stopping_patience': 8,
            }

            # 训练集成模型
            ensemble = self._train_ensemble(trainer, X_tr, y_tr, X_val, y_val, quick_config)

            if ensemble is None:
                logger.warning(f"  窗口 {window_idx+1} 无可用模型，跳过")
                start += retrain_size
                window_idx += 1
                continue

            # 在测试集上预测
            x_tensor = torch.FloatTensor(X_test_seq)
            probs = ensemble.predict_proba(x_tensor)
            preds = (probs >= T0_CONFIG_THRESHOLD).astype(int)

            all_predictions.extend(preds)
            all_probabilities.extend(probs)
            all_labels.extend(y_test_seq)
            all_timestamps.extend(ts_test_seq)

            test_auc = roc_auc_score(y_test_seq, probs) if len(set(y_test_seq)) > 1 else 0.5
            test_acc = accuracy_score(y_test_seq, preds)

            window_results.append({
                'window': window_idx + 1,
                'train_range': f"[{start}:{train_end}]",
                'test_range': f"[{train_end}:{test_end}]",
                'test_auc': test_auc,
                'test_acc': test_acc,
                'n_train': len(X_train_seq),
                'n_test': len(X_test_seq),
                'detected_periods': str(getattr(fe, '_detected_periods', [])),
            })

            logger.info(f"  窗口结果: AUC={test_auc:.4f}, Acc={test_acc:.4f}")

            start += retrain_size
            window_idx += 1

        if len(all_predictions) == 0:
            logger.error("没有任何有效窗口")
            return {}

        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_labels = np.array(all_labels)
        all_timestamps = np.array(all_timestamps)

        # 使用BacktestEngine进行交易模拟
        test_prices = prices_df.iloc[-len(all_predictions):].reset_index(drop=True)
        if len(test_prices) < len(all_predictions):
            all_predictions = all_predictions[:len(test_prices)]
            all_probabilities = all_probabilities[:len(test_prices)]
            all_timestamps = all_timestamps[:len(test_prices)]

        engine = BacktestEngine()
        bt_result = engine.run_backtest(
            predictions=all_predictions,
            probabilities=all_probabilities,
            prices=test_prices,
            timestamps=all_timestamps
        )

        self._save_results(window_results, bt_result)

        return {
            'window_results': window_results,
            'backtest_result': bt_result,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
        }

    def _save_results(self, window_results: list, bt_result: dict):
        """保存结果"""
        pd.DataFrame(window_results).to_csv(
            os.path.join(self.results_dir, 'window_results.csv'), index=False
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        windows = [r['window'] for r in window_results]
        aucs = [r['test_auc'] for r in window_results]
        ax.plot(windows, aucs, 'o-', color='steelblue')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Window')
        ax.set_ylabel('Test AUC')
        ax.set_title('Rolling Window Model Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'rolling_auc.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"滚动回测结果已保存: {self.results_dir}")
