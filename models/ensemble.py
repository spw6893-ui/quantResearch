"""
多模型集成系统
"""
import torch
import torch.nn as nn
import numpy as np


class EnsembleModel:
    """多模型加权集成"""

    def __init__(self, models: dict, weights: dict = None, device: str = "cpu"):
        """
        Args:
            models: {name: model} 字典
            weights: {name: weight} 权重字典
        """
        self.models = models
        self.device = device

        if weights is None:
            n = len(models)
            self.weights = {name: 1.0/n for name in models}
        else:
            self.weights = weights

        # 归一化权重
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """集成预测概率 (返回1D数组, 每个元素为上涨概率)"""
        all_probs = []
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                x_input = x.to(self.device)
                logits = model(x_input)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs * self.weights[name])

        ensemble_probs = np.sum(all_probs, axis=0)
        return ensemble_probs

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """集成预测类别"""
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(int)

    def optimize_weights(self, val_loader, metric_fn):
        """使用验证集优化集成权重"""
        from scipy.optimize import minimize

        # 收集所有模型的预测
        model_probs = {}
        all_labels = []

        for name, model in self.models.items():
            model.eval()
            probs_list = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    probs_list.append(probs)
                    if name == list(self.models.keys())[0]:
                        all_labels.append(batch_y.numpy())
            model_probs[name] = np.concatenate(probs_list, axis=0)

        all_labels = np.concatenate(all_labels)
        model_names = list(self.models.keys())

        def objective(w):
            w = np.abs(w) / np.sum(np.abs(w))  # 归一化
            ensemble = sum(model_probs[name] * w[i] for i, name in enumerate(model_names))
            score = metric_fn(all_labels, ensemble)
            return -score  # 最大化

        n = len(model_names)
        result = minimize(objective, x0=np.ones(n)/n, method='Nelder-Mead')
        opt_weights = np.abs(result.x) / np.sum(np.abs(result.x))

        self.weights = {name: float(opt_weights[i]) for i, name in enumerate(model_names)}
        return self.weights
