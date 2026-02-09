"""
通用工具函数
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
