"""
LightGBM基线模型
将时序序列转为表格特征后用GBDT分类
"""
import numpy as np
import lightgbm as lgb


class LGBMModel:
    """LightGBM分类器，将序列展平为表格特征"""

    def __init__(self, **params):
        self.model_name = "lgbm"
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'seed': 42,
            'n_jobs': -1,
            **params
        }
        self.model = None

    @staticmethod
    def seq_to_tabular(X):
        """将 (N, seq_len, feat) 转为 (N, feat*3) 表格特征

        取最后时步 + 序列均值 + 序列标准差
        """
        last = X[:, -1, :]                           # (N, feat)
        mean = X.mean(axis=1)                         # (N, feat)
        std = X.std(axis=1)                           # (N, feat)
        return np.concatenate([last, mean, std], axis=1).astype(np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            num_boost_round=1000, early_stopping_rounds=50):
        """训练LightGBM"""
        X_tr = self.seq_to_tabular(X_train) if (isinstance(X_train, np.ndarray) and X_train.ndim == 3) else X_train
        dtrain = lgb.Dataset(X_tr, label=y_train)

        valid_sets = [dtrain]
        valid_names = ['train']
        if X_val is not None:
            X_v = self.seq_to_tabular(X_val) if (isinstance(X_val, np.ndarray) and X_val.ndim == 3) else X_val
            dval = lgb.Dataset(X_v, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append('val')

        callbacks = [lgb.log_evaluation(period=50)]
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        self.model = lgb.train(
            self.params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X):
        """预测概率 (输入可以是3D序列或2D表格)"""
        if isinstance(X, np.ndarray) and X.ndim == 3:
            X = self.seq_to_tabular(X)
        return self.model.predict(X)

    def eval(self):
        """兼容PyTorch接口 (no-op)"""
        return self

    def parameters(self):
        """兼容PyTorch接口"""
        return iter([])

    def to(self, device):
        """兼容PyTorch接口 (no-op)"""
        return self

    def state_dict(self):
        """兼容PyTorch接口"""
        return {'model_str': self.model.model_to_string() if self.model else None}

    def load_state_dict(self, state):
        """兼容PyTorch接口"""
        if state.get('model_str'):
            self.model = lgb.Booster(model_str=state['model_str'])
