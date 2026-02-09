#!/bin/bash
# A股T+0量化交易系统 - 实验脚本
# 设计了5组对比实验，覆盖完整的建模流程
#
# 实验设计:
#   实验1: 基线 - 使用默认参数训练全部4个模型 (无Optuna)
#   实验2: Optuna超参数优化 - 对Transformer-LSTM和LSTM进行50次trial搜索
#   实验3: 滚动窗口回测 - 模拟真实交易环境(60天训练窗口, 20天滚动)
#   实验4: 全流程 + Optuna + 滚动回测
#   实验5: 消融实验 - 单模型 vs 集成对比
#
# 用法:
#   bash scripts/run_experiments.sh           # 运行全部实验
#   bash scripts/run_experiments.sh 1         # 仅运行实验1
#   bash scripts/run_experiments.sh 1 2 3     # 运行实验1,2,3
#   TUSHARE_TOKEN=xxx bash scripts/run_experiments.sh  # 使用真实数据

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_DIR/logs/experiments_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  A股T+0量化交易系统 - 实验套件"
echo "  时间: $(date)"
echo "  日志: $LOG_DIR"
echo "============================================================"

# 检查依赖
python3 -c "import torch, numpy, pandas, sklearn, optuna, shap" || {
    echo "ERROR: 缺少依赖。pip install torch numpy pandas scikit-learn optuna matplotlib seaborn shap"
    exit 1
}

# 确定要运行哪些实验
if [ $# -eq 0 ]; then
    EXPERIMENTS=(1 2 3 4 5)
else
    EXPERIMENTS=("$@")
fi

# ============================================================
# 实验1: 基线训练 (默认参数, 无Optuna)
# 目的: 建立性能基准
# 预期输出: 4个模型的5-fold CV AUC, 回测指标
# ============================================================
run_exp1() {
    echo ""
    echo "========== 实验1: 基线训练 =========="
    echo "  描述: 默认参数训练4个模型 (Transformer-LSTM, LSTM, CNN, MLP)"
    echo "  方法: 5-fold时间序列交叉验证"
    echo "  指标: AUC, Accuracy, 夏普比率, 最大回撤"
    echo "======================================"
    python3 main.py --step all 2>&1 | tee "$LOG_DIR/exp1_baseline.log"

    # 备份结果
    if [ -d results ]; then
        cp -r results "$LOG_DIR/exp1_results"
    fi
    echo "实验1完成。结果: $LOG_DIR/exp1_results/"
}

# ============================================================
# 实验2: Optuna超参数优化
# 目的: 搜索最优模型架构和训练参数
# 搜索空间:
#   - learning_rate: [1e-5, 1e-2] (对数)
#   - batch_size: {32, 64, 128}
#   - dropout: [0.1, 0.5]
#   - d_model: {32, 64, 128}
#   - nhead: {2, 4, 8}
#   - lstm_hidden_size: {64, 128, 256}
#   - lstm_num_layers: {1, 2, 3}
#   - fc_hidden_size: {32, 64, 128}
# ============================================================
run_exp2() {
    echo ""
    echo "========== 实验2: Optuna超参数优化 =========="
    echo "  描述: 对Transformer-LSTM和LSTM进行50次trial搜索"
    echo "  搜索空间: 学习率/LSTM单元数/Dropout/d_model/nhead等"
    echo "  剪枝: MedianPruner (5 warmup steps)"
    echo "================================================"
    python3 main.py --step all --optuna 2>&1 | tee "$LOG_DIR/exp2_optuna.log"

    if [ -d results ]; then
        cp -r results "$LOG_DIR/exp2_results"
    fi
    echo "实验2完成。结果: $LOG_DIR/exp2_results/"
}

# ============================================================
# 实验3: 滚动窗口回测
# 目的: 模拟真实交易环境，评估模型在非平稳市场中的适应性
# 参数:
#   - 训练窗口: 60个交易日 (~2880根5分钟K线)
#   - 滚动间隔: 20个交易日 (~960根5分钟K线)
#   - 每次滚动重新训练Transformer-LSTM模型
#   - 交易成本: 佣金0.1% + 印花税0.1% + 滑点0.05%
# ============================================================
run_exp3() {
    echo ""
    echo "========== 实验3: 滚动窗口回测 =========="
    echo "  描述: 60天训练窗口, 20天滚动重训"
    echo "  交易成本: 佣金0.1% + 印花税0.1% + 滑点0.05%"
    echo "  风控: 止损2%, 止盈3%, 动态仓位, 最大3笔/日"
    echo "==========================================="

    # 先准备数据和特征
    python3 main.py --step data 2>&1 | tee "$LOG_DIR/exp3_data.log"
    python3 main.py --step feature 2>&1 | tee -a "$LOG_DIR/exp3_data.log"
    python3 main.py --step rolling 2>&1 | tee "$LOG_DIR/exp3_rolling.log"

    if [ -d results ]; then
        cp -r results "$LOG_DIR/exp3_results"
    fi
    echo "实验3完成。结果: $LOG_DIR/exp3_results/"
}

# ============================================================
# 实验4: 全流程 + Optuna + 滚动回测
# 目的: 完整pipeline，先Optuna搜索最优参数，再滚动回测
# ============================================================
run_exp4() {
    echo ""
    echo "========== 实验4: 全流程(Optuna+滚动回测) =========="
    echo "  描述: Optuna超参数优化 -> 滚动窗口回测"
    echo "====================================================="

    python3 main.py --step all --optuna 2>&1 | tee "$LOG_DIR/exp4_full.log"
    python3 main.py --step rolling 2>&1 | tee "$LOG_DIR/exp4_rolling.log"

    if [ -d results ]; then
        cp -r results "$LOG_DIR/exp4_results"
    fi
    echo "实验4完成。结果: $LOG_DIR/exp4_results/"
}

# ============================================================
# 实验5: 消融实验 - 单模型 vs 集成对比
# 目的: 评估集成学习的增益
# 方法: 分别训练4个单模型并独立回测，与集成结果对比
# ============================================================
run_exp5() {
    echo ""
    echo "========== 实验5: 消融实验(单模型 vs 集成) =========="
    echo "  描述: 对比各单模型与集成模型的性能差异"
    echo "====================================================="

    python3 << 'PYEOF' 2>&1 | tee "$LOG_DIR/exp5_ablation.log"
import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
import pickle

from config.settings import *
from utils.helpers import set_seed, ensure_dir
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.trainer import ModelTrainer
from models.ensemble import EnsembleModel
from models.transformer_lstm import TransformerLSTM
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from backtesting.backtest_engine import BacktestEngine

set_seed(RANDOM_SEED)

# 数据准备
loader = DataLoader()
try:
    df = loader.prepare_data()
except:
    df = loader.generate_synthetic_data(n_days=500)
    df = loader.prepare_data()

fe = FeatureEngineer()
df = fe.build_features(df)
fe.select_features(df)
X, y, ts = fe.create_sequences(df)

# 训练所有模型
trainer = ModelTrainer()
results = trainer.train_all_models(X, y)

# 分别对每个单模型和集成模型进行回测
split = int(len(X) * 0.8)
X_test, y_test, ts_test = X[split:], y[split:], ts[split:]
test_prices = pd.DataFrame({'close': df['close'].values[-len(X_test):]}).reset_index(drop=True)

ablation_results = []
model_configs = {
    'transformer_lstm': (TransformerLSTM, {**TRANSFORMER_LSTM_CONFIG, 'input_size': X.shape[2]}),
    'lstm': (LSTMModel, {**LSTM_CONFIG, 'input_size': X.shape[2]}),
    'cnn': (CNNModel, {**CNN_CONFIG, 'input_size': X.shape[2], 'seq_length': X.shape[1]}),
    'mlp': (MLPModel, {**MLP_CONFIG, 'input_size': X.shape[2], 'seq_length': X.shape[1]}),
}

for name in model_configs:
    if name in results and results[name]['best_model'] is not None:
        model = results[name]['best_model']
        model.eval()
        with torch.no_grad():
            logits = model(torch.FloatTensor(X_test))
            probs = torch.sigmoid(logits).numpy()
        preds = (probs >= T0_CONFIG['signal_threshold']).astype(int)

        engine = BacktestEngine()
        bt = engine.run_backtest(preds, probs, test_prices, ts_test)
        ablation_results.append({
            'model': name,
            'cv_auc': results[name]['avg_test_auc'],
            **bt['metrics']
        })
        print(f"{name}: AUC={results[name]['avg_test_auc']:.4f}, "
              f"Sharpe={bt['metrics']['sharpe_ratio']:.4f}, "
              f"MaxDD={bt['metrics']['max_drawdown']*100:.2f}%")

# 集成模型
models = {n: results[n]['best_model'] for n in results if results[n]['best_model'] is not None}
if len(models) > 1:
    ens = EnsembleModel(models, ENSEMBLE_WEIGHTS, DEVICE)
    x_t = torch.FloatTensor(X_test)
    probs = ens.predict_proba(x_t)
    preds = (probs >= T0_CONFIG['signal_threshold']).astype(int)

    engine = BacktestEngine()
    bt = engine.run_backtest(preds, probs, test_prices, ts_test)
    ablation_results.append({
        'model': 'ensemble',
        'cv_auc': 0,
        **bt['metrics']
    })
    print(f"ensemble: Sharpe={bt['metrics']['sharpe_ratio']:.4f}, "
          f"MaxDD={bt['metrics']['max_drawdown']*100:.2f}%")

# 保存对比结果
ensure_dir('results/ablation')
pd.DataFrame(ablation_results).to_csv('results/ablation/ablation_results.csv', index=False)
print("\n消融实验完成。结果: results/ablation/ablation_results.csv")
PYEOF

    if [ -d results/ablation ]; then
        cp -r results/ablation "$LOG_DIR/exp5_results"
    fi
    echo "实验5完成。结果: $LOG_DIR/exp5_results/"
}

# 运行选定的实验
for exp in "${EXPERIMENTS[@]}"; do
    case $exp in
        1) run_exp1 ;;
        2) run_exp2 ;;
        3) run_exp3 ;;
        4) run_exp4 ;;
        5) run_exp5 ;;
        *) echo "未知实验: $exp (可选: 1-5)" ;;
    esac
done

echo ""
echo "============================================================"
echo "  全部实验完成!"
echo "  日志目录: $LOG_DIR"
echo "  结果目录: $PROJECT_DIR/results/"
echo "============================================================"
