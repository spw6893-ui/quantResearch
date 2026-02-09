#!/bin/bash
# A股T+0量化交易系统 - 训练脚本
# 中芯国际(688981) 5分钟级别预测与交易
#
# 用法:
#   bash scripts/run_train.sh                    # 完整流程 (数据+特征+训练+分析+回测)
#   bash scripts/run_train.sh --optuna           # 启用Optuna超参数优化
#   bash scripts/run_train.sh --step train       # 仅训练
#   bash scripts/run_train.sh --step rolling     # 滚动窗口回测
#   bash scripts/run_train.sh --step analysis    # 仅可解释性分析
#   TUSHARE_TOKEN=xxx bash scripts/run_train.sh  # 使用Tushare真实数据

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "  A股T+0量化交易系统 - 中芯国际(688981)"
echo "  项目目录: $PROJECT_DIR"
echo "  Python: $(python3 --version 2>&1)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "  CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  TUSHARE_TOKEN: $([ -n "$TUSHARE_TOKEN" ] && echo 'SET' || echo 'NOT SET (将使用模拟数据)')"
echo "============================================================"

# 检查依赖
python3 -c "
import torch, numpy, pandas, sklearn, optuna
print('依赖检查通过')
" || {
    echo "ERROR: 缺少必要依赖。请运行:"
    echo "  pip install torch numpy pandas scikit-learn optuna matplotlib seaborn shap"
    exit 1
}

# 运行
echo ""
echo "启动训练..."
echo "参数: $@"
echo ""

python3 main.py "$@"

echo ""
echo "============================================================"
echo "  训练完成!"
echo "  结果目录: $PROJECT_DIR/results/"
echo "  模型目录: $PROJECT_DIR/models/"
echo "  日志目录: $PROJECT_DIR/logs/"
echo "============================================================"
