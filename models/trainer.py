"""
模型训练模块
实施时间序列交叉验证、Optuna超参数优化、早停机制
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
import optuna
from copy import deepcopy
import pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    TRAINING_CONFIG, CV_CONFIG, OPTUNA_CONFIG, MODEL_DIR,
    TRANSFORMER_LSTM_CONFIG, LSTM_CONFIG, CNN_CONFIG, MLP_CONFIG,
    SEQUENCE_LENGTH, DEVICE, RANDOM_SEED
)
from models.transformer_lstm import TransformerLSTM
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from utils.logger import get_logger
from utils.helpers import set_seed, ensure_dir

logger = get_logger(__name__, "training.log")


class TimeSeriesSplitter:
    """时间序列交叉验证分割器 (expanding window)"""

    def __init__(self, n_splits=5, gap=10, val_ratio=0.15, test_ratio=0.15):
        self.n_splits = n_splits
        self.gap = gap
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, n_samples):
        """生成expanding window交叉验证索引"""
        splits = []
        val_size = int(n_samples * self.val_ratio)
        test_size = int(n_samples * self.test_ratio)
        block_size = val_size + test_size + 2 * self.gap
        usable = n_samples - block_size
        step = max(usable // self.n_splits, 1)

        for i in range(self.n_splits):
            train_end = step * (i + 1)
            if train_end < 100:
                continue

            val_start = train_end + self.gap
            val_end = val_start + val_size
            test_start = val_end + self.gap
            test_end = min(test_start + test_size, n_samples)

            if test_end > n_samples or val_end > n_samples:
                continue

            train_idx = list(range(0, train_end))
            val_idx = list(range(val_start, val_end))
            test_idx = list(range(test_start, test_end))

            if len(train_idx) > 100 and len(val_idx) > 10 and len(test_idx) > 10:
                splits.append((train_idx, val_idx, test_idx))

        return splits


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=15, min_delta=1e-4, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_model_state = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            return

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_model_state = deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ModelTrainer:
    """模型训练器"""

    def __init__(self, device: str = DEVICE):
        self.device = torch.device(device)
        ensure_dir(MODEL_DIR)
        set_seed(RANDOM_SEED)

    def _create_model(self, model_type: str, input_size: int,
                      seq_length: int = SEQUENCE_LENGTH, **kwargs) -> nn.Module:
        """创建指定类型的模型"""
        if model_type == "transformer_lstm":
            config = {**TRANSFORMER_LSTM_CONFIG, **kwargs}
            model = TransformerLSTM(
                input_size=input_size,
                d_model=config['d_model'],
                nhead=config['nhead'],
                num_transformer_layers=config['num_transformer_layers'],
                lstm_hidden_size=config['lstm_hidden_size'],
                lstm_num_layers=config['lstm_num_layers'],
                dropout=config['dropout'],
                fc_hidden_size=config['fc_hidden_size'],
            )
        elif model_type == "lstm":
            config = {**LSTM_CONFIG, **kwargs}
            model = LSTMModel(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                fc_hidden_size=config['fc_hidden_size'],
            )
        elif model_type == "cnn":
            config = {**CNN_CONFIG, **kwargs}
            model = CNNModel(
                input_size=input_size,
                seq_length=seq_length,
                num_filters=config['num_filters'],
                kernel_sizes=config['kernel_sizes'],
                dropout=config['dropout'],
                fc_hidden_size=config['fc_hidden_size'],
            )
        elif model_type == "mlp":
            config = {**MLP_CONFIG, **kwargs}
            model = MLPModel(
                input_size=input_size,
                seq_length=seq_length,
                hidden_sizes=config['hidden_sizes'],
                dropout=config['dropout'],
            )
        else:
            raise ValueError(f"未知模型类型: {model_type}")

        return model.to(self.device)

    def _compute_pos_weight(self, y):
        """计算正样本权重用于不平衡标签校正"""
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        if n_pos == 0 or n_neg == 0:
            return None
        return torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)

    def train_epoch(self, model, train_loader, criterion, optimizer, clip_norm=1.0):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            optimizer.step()
            total_loss += loss.item()

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        return avg_loss, auc

    @torch.no_grad()
    def evaluate(self, model, data_loader, criterion):
        """评估模型"""
        model.eval()
        total_loss = 0
        all_preds, all_labels = [], []

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        acc = accuracy_score(all_labels, (np.array(all_preds) >= 0.5).astype(int))
        return avg_loss, auc, acc, np.array(all_preds), np.array(all_labels)

    def train_model(self, model_type: str, X_train, y_train, X_val, y_val,
                    config: dict = None) -> tuple:
        """训练单个模型"""
        if config is None:
            config = TRAINING_CONFIG

        input_size = X_train.shape[2]
        seq_length = X_train.shape[1]
        model = self._create_model(model_type, input_size, seq_length)

        logger.info(f"训练 {model_type} 模型: input_size={input_size}, seq_length={seq_length}")
        logger.info(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 数据加载器 (只有训练集用drop_last)
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train.astype(np.float32))
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val.astype(np.float32))
        )
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], drop_last=False)

        # 损失函数 (带pos_weight校正标签不平衡)
        pos_weight = self._compute_pos_weight(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=config['scheduler_factor'],
            patience=config['scheduler_patience']
        )
        early_stopping = EarlyStopping(
            patience=config['early_stopping_patience'],
            min_delta=config['early_stopping_min_delta'],
            mode='max'
        )

        # 训练循环
        best_metrics = {}
        for epoch in range(config['max_epochs']):
            train_loss, train_auc = self.train_epoch(
                model, train_loader, criterion, optimizer, config['gradient_clip_norm']
            )
            val_loss, val_auc, val_acc, _, _ = self.evaluate(model, val_loader, criterion)
            scheduler.step(val_auc)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"  Epoch {epoch+1}/{config['max_epochs']}: "
                    f"train_loss={train_loss:.4f}, train_auc={train_auc:.4f}, "
                    f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, val_acc={val_acc:.4f}"
                )

            early_stopping(val_auc, model)
            if early_stopping.early_stop:
                logger.info(f"  早停于 epoch {epoch+1}, 最佳 val_auc={early_stopping.best_score:.4f}")
                break

            best_metrics = {
                'val_auc': early_stopping.best_score,
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch + 1
            }

        # 恢复最佳模型
        if early_stopping.best_model_state:
            model.load_state_dict(early_stopping.best_model_state)

        return model, best_metrics

    def cross_validate(self, model_type: str, X, y, cv_config: dict = None) -> dict:
        """时间序列交叉验证"""
        if cv_config is None:
            cv_config = CV_CONFIG

        splitter = TimeSeriesSplitter(
            n_splits=cv_config['n_splits'],
            gap=cv_config['gap'],
            val_ratio=cv_config['val_ratio'],
            test_ratio=cv_config['test_ratio'],
        )
        splits = splitter.split(len(X))

        logger.info(f"\n时间序列交叉验证 ({model_type}): {len(splits)} folds")

        fold_results = []
        best_model = None
        best_auc = 0

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits):
            logger.info(f"\n--- Fold {fold_idx+1}/{len(splits)} ---")
            logger.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model, metrics = self.train_model(
                model_type, X_train, y_train, X_val, y_val
            )

            # 在测试集上评估 (不用drop_last)
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test), torch.FloatTensor(y_test.astype(np.float32))
            )
            test_loader = DataLoader(test_dataset, batch_size=TRAINING_CONFIG['batch_size'], drop_last=False)
            criterion = nn.BCEWithLogitsLoss()
            _, test_auc, test_acc, test_preds, test_labels = self.evaluate(
                model, test_loader, criterion
            )

            fold_result = {
                'fold': fold_idx + 1,
                'val_auc': metrics.get('val_auc', 0),
                'test_auc': test_auc,
                'test_acc': test_acc,
            }
            fold_results.append(fold_result)

            logger.info(f"  Test AUC: {test_auc:.4f}, Test Acc: {test_acc:.4f}")

            if test_auc > best_auc:
                best_auc = test_auc
                best_model = deepcopy(model)

        # 汇总结果
        avg_test_auc = np.mean([r['test_auc'] for r in fold_results])
        avg_test_acc = np.mean([r['test_acc'] for r in fold_results])
        logger.info(f"\n{model_type} CV结果: 平均Test AUC={avg_test_auc:.4f}, 平均Test Acc={avg_test_acc:.4f}")

        return {
            'model_type': model_type,
            'best_model': best_model,
            'fold_results': fold_results,
            'avg_test_auc': avg_test_auc,
            'avg_test_acc': avg_test_acc
        }

    def optimize_hyperparams(self, model_type: str, X, y,
                             n_trials: int = None) -> dict:
        """使用Optuna进行超参数优化"""
        if n_trials is None:
            n_trials = OPTUNA_CONFIG['n_trials']

        split_idx = int(len(X) * 0.7)
        gap = CV_CONFIG['gap']
        val_end = int(len(X) * 0.85)

        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx+gap:val_end], y[split_idx+gap:val_end]

        logger.info(f"\nOptuna超参数优化 ({model_type}): {n_trials} trials")

        input_size = X.shape[2]
        seq_length = X.shape[1]

        def objective(trial):
            config = {
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                'max_epochs': 50,
                'early_stopping_patience': 10,
                'early_stopping_min_delta': 1e-4,
                'scheduler_factor': 0.5,
                'scheduler_patience': 5,
                'gradient_clip_norm': 1.0,
            }

            dropout = trial.suggest_float('dropout', 0.1, 0.5)

            if model_type == "transformer_lstm":
                d_model_choices = [32, 64, 128]
                nhead_choices = [2, 4, 8]
                d_model = trial.suggest_categorical('d_model', d_model_choices)
                valid_nheads = [n for n in nhead_choices if d_model % n == 0]
                nhead = trial.suggest_categorical('nhead', valid_nheads)
                model_kwargs = {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_transformer_layers': trial.suggest_int('num_transformer_layers', 1, 4),
                    'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [64, 128, 256]),
                    'lstm_num_layers': trial.suggest_int('lstm_num_layers', 1, 3),
                    'dropout': dropout,
                    'fc_hidden_size': trial.suggest_categorical('fc_hidden_size', [32, 64, 128]),
                }
            elif model_type == "lstm":
                model_kwargs = {
                    'hidden_size': trial.suggest_categorical('hidden_size', [64, 128, 256]),
                    'num_layers': trial.suggest_int('num_layers', 1, 3),
                    'dropout': dropout,
                    'fc_hidden_size': trial.suggest_categorical('fc_hidden_size', [32, 64, 128]),
                }
            elif model_type == "cnn":
                n_conv_layers = trial.suggest_int('n_conv_layers', 2, 4)
                num_filters = [trial.suggest_categorical(f'filters_{i}', [32, 64, 128])
                              for i in range(n_conv_layers)]
                model_kwargs = {
                    'num_filters': num_filters,
                    'kernel_sizes': [3] * n_conv_layers,
                    'dropout': dropout,
                    'fc_hidden_size': trial.suggest_categorical('fc_hidden_size', [32, 64, 128]),
                }
            elif model_type == "mlp":
                n_layers = trial.suggest_int('n_layers', 2, 4)
                hidden_sizes = [trial.suggest_categorical(f'hidden_{i}', [64, 128, 256, 512])
                               for i in range(n_layers)]
                model_kwargs = {
                    'hidden_sizes': hidden_sizes,
                    'dropout': dropout,
                }
            else:
                model_kwargs = {'dropout': dropout}

            try:
                model = self._create_model(
                    model_type, input_size, seq_length, **model_kwargs
                )
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train), torch.FloatTensor(y_train.astype(np.float32))
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val), torch.FloatTensor(y_val.astype(np.float32))
                )
                train_loader = DataLoader(
                    train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=config['batch_size'], drop_last=False
                )

                pos_weight = self._compute_pos_weight(y_train)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=config['weight_decay']
                )
                early_stopping = EarlyStopping(
                    patience=config['early_stopping_patience'],
                    min_delta=config['early_stopping_min_delta'],
                    mode='max'
                )

                for epoch in range(config['max_epochs']):
                    self.train_epoch(
                        model, train_loader, criterion, optimizer,
                        config['gradient_clip_norm']
                    )
                    _, val_auc, _, _, _ = self.evaluate(
                        model, val_loader, criterion
                    )
                    early_stopping(val_auc, model)
                    if early_stopping.early_stop:
                        break

                    trial.report(val_auc, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return early_stopping.best_score or 0.5
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.5

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        study.optimize(objective, n_trials=n_trials,
                      timeout=OPTUNA_CONFIG.get('timeout', 3600))

        logger.info(f"最佳超参数: {study.best_params}")
        logger.info(f"最佳 Val AUC: {study.best_value:.4f}")

        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study
        }

    @staticmethod
    def _extract_arch_params(model_type, params):
        """从Optuna best_params中提取架构参数(去除训练参数)"""
        training_keys = {'batch_size', 'learning_rate', 'weight_decay'}
        arch = {k: v for k, v in params.items() if k not in training_keys}
        if model_type == 'cnn' and 'n_conv_layers' in arch:
            n = arch.pop('n_conv_layers')
            arch['num_filters'] = [arch.pop(f'filters_{i}') for i in range(n)]
            arch['kernel_sizes'] = [3] * n
        if model_type == 'mlp' and 'n_layers' in arch:
            n = arch.pop('n_layers')
            arch['hidden_sizes'] = [arch.pop(f'hidden_{i}') for i in range(n)]
        return arch

    def train_all_models(self, X, y, optuna_params: dict = None) -> dict:
        """训练所有模型并返回结果

        Args:
            optuna_params: {model_type: best_params} 从Optuna优化得到的参数
        """
        # 将Optuna最佳参数写入全局配置
        if optuna_params:
            import config.settings as cfg
            for mt, params in optuna_params.items():
                arch = self._extract_arch_params(mt, params)
                if mt == 'transformer_lstm':
                    cfg.TRANSFORMER_LSTM_CONFIG.update(arch)
                elif mt == 'lstm':
                    cfg.LSTM_CONFIG.update(arch)
                elif mt == 'cnn':
                    cfg.CNN_CONFIG.update(arch)
                elif mt == 'mlp':
                    cfg.MLP_CONFIG.update(arch)
                logger.info(f"  {mt} 应用Optuna架构参数: {arch}")

        model_types = ["transformer_lstm", "lstm", "cnn", "mlp"]
        results = {}

        for mt in model_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"训练模型: {mt}")
            logger.info(f"{'='*60}")
            cv_result = self.cross_validate(mt, X, y)
            results[mt] = cv_result

            # 保存模型
            model_path = os.path.join(MODEL_DIR, f"{mt}_best.pt")
            torch.save(cv_result['best_model'].state_dict(), model_path)
            logger.info(f"模型已保存: {model_path}")

        return results

    def save_model(self, model, model_type: str, extra_info: dict = None):
        """保存模型"""
        model_path = os.path.join(MODEL_DIR, f"{model_type}_best.pt")
        torch.save(model.state_dict(), model_path)
        if extra_info:
            info_path = os.path.join(MODEL_DIR, f"{model_type}_info.json")
            with open(info_path, 'w') as f:
                json.dump(extra_info, f, indent=2, default=str)
        logger.info(f"模型已保存: {model_path}")


if __name__ == "__main__":
    from data.data_loader import DataLoader as DL
    from data.feature_engineering import FeatureEngineer

    loader = DL()
    df = loader.prepare_data()

    fe = FeatureEngineer()
    df = fe.build_features(df)
    selected = fe.select_features(df)
    X, y, ts = fe.create_sequences(df)

    trainer = ModelTrainer()
    results = trainer.train_all_models(X, y)

    print("\n" + "="*60)
    print("所有模型训练结果:")
    for mt, res in results.items():
        print(f"  {mt}: avg_test_auc={res['avg_test_auc']:.4f}, avg_test_acc={res['avg_test_acc']:.4f}")
