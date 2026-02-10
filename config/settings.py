"""
A股T+0量化交易系统 - 全局配置
中芯国际(688981) 5分钟级别预测与交易
"""
import os

# ============ 基本设置 ============
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
FEATURES_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ============ 股票设置 ============
SYMBOL = "688981"  # 中芯国际
SYMBOL_NAME = "中芯国际"
EXCHANGE = "SSE"  # 上海证券交易所
BAR_INTERVAL = "5min"  # 5分钟K线

# ============ 数据设置 ============
# Tushare token (需要用户自行设置)
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")
TS_SYMBOL = f"{SYMBOL}.SH"  # Tushare格式: 688981.SH

# 数据时间范围
DATA_START_DATE = "20220101"
DATA_END_DATE = "20250207"

# ============ 特征工程设置 ============
# 技术指标参数 (5min bars: 12=1hr, 48=1day, 240=1week)
MA_PERIODS = [5, 10, 20, 48, 96, 240]        # 25min, 50min, 1.7hr, 1day, 2day, 1week
EMA_PERIODS = [5, 12, 24, 48, 96]             # 25min, 1hr, 2hr, 1day, 2day
RSI_PERIODS = [6, 12, 24, 48]                 # 30min, 1hr, 2hr, 1day
MACD_PARAMS = (12, 26, 9)                     # fast, slow, signal
BOLL_PERIOD = 20
BOLL_STD = 2
ATR_PERIOD = 14
KDJ_PARAMS = (9, 3, 3)
CCI_PERIOD = 14
WILLIAMS_PERIOD = 14
MFI_PERIOD = 14
OBV_ENABLED = True
VWAP_ENABLED = True

# 动态周期检测
DYNAMIC_PERIODS_ENABLED = True       # 启用自动周期检测
DYNAMIC_PERIODS_MAX = 5              # 最多检测5个主导周期
DYNAMIC_PERIODS_RANGE = (6, 480)     # 搜索范围: 30min ~ 10天
DYNAMIC_PERIODS_MIN_STRENGTH = 0.02  # 最小自相关强度

# 时序特征
LOOKBACK_WINDOW = 120  # 历史回看窗口(120个5分钟K线 = 10小时)
PREDICT_HORIZON = 6    # 预测未来6个周期(30分钟)
SEQUENCE_LENGTH = 60   # 模型输入序列长度 (扩大以容纳更长周期)

# 特征选择
FEATURE_SELECTION_METHOD = "mutual_info"  # mutual_info, f_classif, random_forest
MAX_FEATURES = 80  # 最大特征数 (扩大以容纳多尺度特征)

# ============ 模型设置 ============
# 通用设置
RANDOM_SEED = 42
import torch as _torch
DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
# Transformer-LSTM混合模型
TRANSFORMER_LSTM_CONFIG = {
    "d_model": 64,
    "nhead": 4,
    "num_transformer_layers": 2,
    "lstm_hidden_size": 128,
    "lstm_num_layers": 2,
    "dropout": 0.3,
    "fc_hidden_size": 64,
}

# 普通LSTM模型
LSTM_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "fc_hidden_size": 64,
}

# CNN模型
CNN_CONFIG = {
    "num_filters": [32, 64, 128],
    "kernel_sizes": [3, 3, 3],
    "dropout": 0.3,
    "fc_hidden_size": 64,
}

# MLP模型
MLP_CONFIG = {
    "hidden_sizes": [256, 128, 64],
    "dropout": 0.3,
}

# 集成模型权重 (初始值，可通过训练调整)
ENSEMBLE_WEIGHTS = {
    "transformer_lstm": 0.4,
    "lstm": 0.25,
    "cnn": 0.2,
    "mlp": 0.15,
}

# ============ 训练设置 ============
TRAINING_CONFIG = {
    "batch_size": 64,
    "max_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "early_stopping_patience": 15,
    "early_stopping_min_delta": 1e-4,
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "gradient_clip_norm": 1.0,
}

# 时间序列交叉验证
CV_CONFIG = {
    "n_splits": 5,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "gap": 10,  # 训练集和验证集之间的间隔(防止数据泄露)
}

# Optuna超参数优化
OPTUNA_CONFIG = {
    "n_trials": 50,
    "timeout": 3600,  # 1小时超时
    "metric": "val_auc",
    "direction": "maximize",
}

# ============ 回测设置 ============
BACKTEST_CONFIG = {
    "initial_capital": 1000000,  # 初始资金100万
    "commission_rate": 0.001,   # 佣金率0.1%
    "slippage": 0.0005,         # 滑点0.05%
    "stamp_tax": 0.001,         # 印花税0.1%(卖出时)
    "max_position_ratio": 0.3,  # 最大仓位比例30%
    "stop_loss_pct": 0.02,      # 止损比例2%
    "take_profit_pct": 0.03,    # 止盈比例3%
    "rolling_window_days": 60,  # 滚动窗口60个交易日
    "retrain_interval_days": 20,  # 每20个交易日重新训练
}

# T+0交易设置
T0_CONFIG = {
    "max_daily_trades": 3,      # 每日最大交易次数
    "min_trade_interval": 15,   # 最小交易间隔(分钟)
    "signal_threshold": 0.6,    # 信号阈值
    "position_sizes": [0.3, 0.5, 0.7],  # 仓位梯度
}

# ============ 日志设置 ============
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
