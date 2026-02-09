"""
模型可解释性分析模块
SHAP分析、梯度重要性分析、排列重要性评估
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import RESULTS_DIR
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__, "analysis.log")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ModelInterpreter:
    """模型可解释性分析器"""

    def __init__(self, model, feature_names: list, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.feature_names = feature_names
        self.results_dir = os.path.join(RESULTS_DIR, "interpretability")
        ensure_dir(self.results_dir)

    def shap_analysis(self, X_background, X_explain, max_samples: int = 100):
        """SHAP分析: 可视化特征对预测结果的影响"""
        import shap
        logger.info("开始SHAP分析...")

        self.model.eval()

        # 包装模型为函数
        def model_predict(x):
            with torch.no_grad():
                tensor_x = torch.FloatTensor(x).to(self.device)
                logits = self.model(tensor_x)
                probs = torch.sigmoid(logits).cpu().numpy()
            return probs

        # 使用KernelExplainer (适用于所有模型)
        bg_samples = min(50, len(X_background))
        bg_idx = np.random.choice(len(X_background), bg_samples, replace=False)
        background = X_background[bg_idx]

        exp_samples = min(max_samples, len(X_explain))
        exp_idx = np.random.choice(len(X_explain), exp_samples, replace=False)
        explain = X_explain[exp_idx]

        # 展平序列数据用于SHAP
        bg_flat = background.reshape(bg_samples, -1)
        exp_flat = explain.reshape(exp_samples, -1)

        def flat_predict(x):
            seq_len = background.shape[1]
            n_features = background.shape[2]
            x_3d = x.reshape(-1, seq_len, n_features)
            return model_predict(x_3d)

        explainer = shap.KernelExplainer(flat_predict, bg_flat)
        shap_values = explainer.shap_values(exp_flat, nsamples=100)

        # 聚合到特征级别 (对序列维度取均值)
        seq_len = background.shape[1]
        n_features = background.shape[2]
        shap_by_feature = shap_values.reshape(exp_samples, seq_len, n_features)
        shap_importance = np.abs(shap_by_feature).mean(axis=(0, 1))  # 对样本和时间步取均值

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 特征重要性条形图
        top_k = min(20, len(self.feature_names))
        sorted_idx = np.argsort(shap_importance)[::-1][:top_k]
        axes[0].barh(
            [self.feature_names[i] for i in sorted_idx][::-1],
            shap_importance[sorted_idx][::-1]
        )
        axes[0].set_xlabel('Mean |SHAP value|')
        axes[0].set_title('SHAP Feature Importance (Top 20)')

        # 热力图: 时间步 x 特征
        shap_heatmap = np.abs(shap_by_feature).mean(axis=0)  # (seq_len, n_features)
        top_feat_idx = sorted_idx[:10]
        sns.heatmap(
            shap_heatmap[:, top_feat_idx].T,
            ax=axes[1],
            xticklabels=range(seq_len),
            yticklabels=[self.feature_names[i] for i in top_feat_idx],
            cmap='YlOrRd'
        )
        axes[1].set_xlabel('Time Step')
        axes[1].set_title('SHAP Values Heatmap (Top 10 Features)')

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'shap_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP分析图已保存: {save_path}")

        return {
            'shap_importance': dict(zip(self.feature_names, shap_importance)),
            'top_features': [(self.feature_names[i], shap_importance[i]) for i in sorted_idx]
        }

    def gradient_importance(self, X_sample, n_samples: int = 200):
        """梯度重要性分析: 识别关键预测因子"""
        logger.info("开始梯度重要性分析...")
        self.model.eval()

        idx = np.random.choice(len(X_sample), min(n_samples, len(X_sample)), replace=False)
        x = torch.FloatTensor(X_sample[idx]).to(self.device)
        x.requires_grad = True

        logits = self.model(x)
        probs = torch.sigmoid(logits)
        probs.sum().backward()

        gradients = x.grad.cpu().numpy()  # (n_samples, seq_len, n_features)
        grad_importance = np.abs(gradients).mean(axis=(0, 1))  # 对样本和时间步取均值

        # 可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        top_k = min(20, len(self.feature_names))
        sorted_idx = np.argsort(grad_importance)[::-1][:top_k]
        ax.barh(
            [self.feature_names[i] for i in sorted_idx][::-1],
            grad_importance[sorted_idx][::-1],
            color='steelblue'
        )
        ax.set_xlabel('Mean |Gradient|')
        ax.set_title('Gradient-based Feature Importance (Top 20)')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'gradient_importance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"梯度重要性图已保存: {save_path}")

        return {
            'gradient_importance': dict(zip(self.feature_names, grad_importance)),
            'top_features': [(self.feature_names[i], grad_importance[i]) for i in sorted_idx]
        }

    def permutation_importance_analysis(self, X, y, n_repeats: int = 10):
        """排列重要性评估特征稳定性"""
        logger.info("开始排列重要性分析...")
        self.model.eval()

        # 基准AUC
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            logits = self.model(x_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        base_auc = roc_auc_score(y, probs)

        n_features = X.shape[2]
        importance_scores = np.zeros((n_repeats, n_features))

        for repeat in range(n_repeats):
            for feat_idx in range(n_features):
                X_perm = X.copy()
                # 打乱该特征在所有时间步上的值
                perm_idx = np.random.permutation(len(X_perm))
                X_perm[:, :, feat_idx] = X_perm[perm_idx, :, feat_idx]

                with torch.no_grad():
                    x_tensor = torch.FloatTensor(X_perm).to(self.device)
                    logits = self.model(x_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()
                perm_auc = roc_auc_score(y, probs)
                importance_scores[repeat, feat_idx] = base_auc - perm_auc

        mean_importance = importance_scores.mean(axis=0)
        std_importance = importance_scores.std(axis=0)

        # 可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        top_k = min(20, len(self.feature_names))
        sorted_idx = np.argsort(mean_importance)[::-1][:top_k]

        ax.barh(
            [self.feature_names[i] for i in sorted_idx][::-1],
            mean_importance[sorted_idx][::-1],
            xerr=std_importance[sorted_idx][::-1],
            color='coral', capsize=3
        )
        ax.set_xlabel('AUC Decrease')
        ax.set_title('Permutation Feature Importance (Top 20)')
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'permutation_importance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"排列重要性图已保存: {save_path}")

        return {
            'mean_importance': dict(zip(self.feature_names, mean_importance)),
            'std_importance': dict(zip(self.feature_names, std_importance)),
            'top_features': [(self.feature_names[i], mean_importance[i], std_importance[i])
                           for i in sorted_idx]
        }

    def full_analysis(self, X_train, X_test, y_test):
        """完整的可解释性分析"""
        results = {}

        # 梯度重要性
        results['gradient'] = self.gradient_importance(X_test)

        # 排列重要性 (使用测试集子集)
        max_samples = min(500, len(X_test))
        idx = np.random.choice(len(X_test), max_samples, replace=False)
        results['permutation'] = self.permutation_importance_analysis(
            X_test[idx], y_test[idx], n_repeats=5
        )

        # SHAP分析 (较慢，使用较少样本)
        try:
            results['shap'] = self.shap_analysis(
                X_train[:100], X_test[:50], max_samples=50
            )
        except Exception as e:
            logger.warning(f"SHAP分析失败: {e}")
            results['shap'] = None

        # 综合排名
        self._create_summary(results)
        return results

    def _create_summary(self, results: dict):
        """创建综合分析摘要"""
        fig, ax = plt.subplots(figsize=(12, 8))

        methods = []
        all_scores = {}

        for method_name, result in results.items():
            if result is None:
                continue
            if 'top_features' in result:
                methods.append(method_name)
                for item in result['top_features'][:15]:
                    fname = item[0]
                    score = item[1]
                    if fname not in all_scores:
                        all_scores[fname] = {}
                    all_scores[fname][method_name] = score

        if not all_scores:
            return

        # 计算综合排名
        for fname in all_scores:
            for method in methods:
                if method not in all_scores[fname]:
                    all_scores[fname][method] = 0

        df_scores = pd.DataFrame(all_scores).T
        # 每个方法归一化到0-1
        for col in df_scores.columns:
            max_val = df_scores[col].max()
            if max_val > 0:
                df_scores[col] = df_scores[col] / max_val

        df_scores['avg_rank'] = df_scores.mean(axis=1)
        df_scores = df_scores.sort_values('avg_rank', ascending=False).head(15)

        df_scores[methods].plot(kind='barh', ax=ax, width=0.8)
        ax.set_xlabel('Normalized Importance Score')
        ax.set_title('Feature Importance Summary (Multi-method)')
        ax.legend(title='Method')
        plt.tight_layout()

        save_path = os.path.join(self.results_dir, 'importance_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"综合分析摘要已保存: {save_path}")
