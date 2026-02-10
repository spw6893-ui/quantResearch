"""
XGBoost基线模型
将时序序列转为表格特征后用GBDT分类
"""
import numpy as np
import xgboost as xgb


class XGBModel:
    """XGBoost分类器，将序列展平为表格特征"""

    def __init__(self, **params):
        self.model_name = "xgboost"
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 0,
            'seed': 42,
            'nthread': -1,
            **params
        }
        self.model = None

    @staticmethod
    def seq_to_tabular(X):
        """将 (N, seq_len, feat) 转为 (N, feat*3) 表格特征"""
        last = X[:, -1, :]
        mean = X.mean(axis=1)
        std = X.std(axis=1)
        return np.concatenate([last, mean, std], axis=1).astype(np.float32)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            num_boost_round=1000, early_stopping_rounds=50):
        """训练XGBoost"""
        X_tr = self.seq_to_tabular(X_train)
        dtrain = xgb.DMatrix(X_tr, label=y_train)

        evals = [(dtrain, 'train')]
        if X_val is not None:
            X_v = self.seq_to_tabular(X_val)
            dval = xgb.DMatrix(X_v, label=y_val)
            evals.append((dval, 'val'))

        self.model = xgb.train(
            self.params, dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=50,
        )
        return self

    def predict_proba(self, X):
        """预测概率 (输入可以是3D序列或2D表格)"""
        if isinstance(X, np.ndarray) and X.ndim == 3:
            X = self.seq_to_tabular(X)
        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)

    def eval(self):
        return self

    def parameters(self):
        return iter([])

    def to(self, device):
        return self

    def state_dict(self):
        return {'model_json': self.model.save_raw(raw_format='json').decode() if self.model else None}

    def load_state_dict(self, state):
        if state.get('model_json'):
            self.model = xgb.Booster()
            self.model.load_model(bytearray(state['model_json'].encode()))
