"""
A股T+0量化交易系统 - 主程序
中芯国际(688981) 5分钟级别预测与交易

使用方法:
    python main.py                    # 完整流程(模拟数据)
    python main.py --step data        # 仅数据准备
    python main.py --step feature     # 仅特征工程
    python main.py --step train       # 仅模型训练
    python main.py --step analysis    # 仅可解释性分析
    python main.py --step backtest    # 仅回测
    python main.py --optuna           # 启用超参数优化
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import *
from utils.helpers import set_seed, ensure_dir
from utils.logger import get_logger
from data.data_loader import DataLoader
from data.feature_engineering import FeatureEngineer
from models.trainer import ModelTrainer
from models.ensemble import EnsembleModel
from models.transformer_lstm import TransformerLSTM
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from analysis.interpretability import ModelInterpreter
from backtesting.backtest_engine import BacktestEngine
from backtesting.rolling_backtest import RollingBacktest

logger = get_logger("main", "main.log")


def step_data():
    """步骤1: 数据准备"""
    logger.info("=" * 60)
    logger.info("步骤1: 数据准备")
    logger.info("=" * 60)

    loader = DataLoader()
    if TUSHARE_TOKEN:
        df = loader.fetch_from_tushare()
    else:
        logger.info("未设置TUSHARE_TOKEN，使用模拟数据")
        df = loader.generate_synthetic_data(n_days=500)

    df = loader.prepare_data()
    logger.info(f"数据量: {len(df)}, 列: {list(df.columns)}")
    return df


def step_feature(df=None):
    """步骤2: 特征工程"""
    logger.info("=" * 60)
    logger.info("步骤2: 特征工程")
    logger.info("=" * 60)

    if df is None:
        loader = DataLoader()
        df = loader.prepare_data()

    fe = FeatureEngineer()
    df = fe.build_features(df)
    selected = fe.select_features(df)
    X, y, ts = fe.create_sequences(df)

    # 保存
    ensure_dir(PROCESSED_DATA_DIR)
    save_data = {
        'X': X, 'y': y, 'timestamps': ts,
        'feature_names': fe.selected_features,
        'scaler': fe.scaler,
        'all_feature_names': fe.feature_names
    }
    save_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    logger.info(f"处理后的数据已保存: {save_path}")

    # 同时保存原始带特征的df用于回测
    df_path = os.path.join(PROCESSED_DATA_DIR, "featured_df.pkl")
    df.to_pickle(df_path)

    return X, y, ts, fe


def step_train(X=None, y=None, use_optuna=False):
    """步骤3: 模型训练"""
    logger.info("=" * 60)
    logger.info("步骤3: 模型训练")
    logger.info("=" * 60)

    if X is None:
        data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, y = data['X'], data['y']

    trainer = ModelTrainer()

    if use_optuna:
        logger.info("使用Optuna进行超参数优化...")
        for mt in ["transformer_lstm", "lstm"]:
            opt_result = trainer.optimize_hyperparams(mt, X, y, n_trials=20)
            logger.info(f"{mt} 最佳参数: {opt_result['best_params']}")

    results = trainer.train_all_models(X, y)

    # 打印汇总
    logger.info("\n" + "=" * 60)
    logger.info("模型训练结果汇总:")
    logger.info("=" * 60)
    for mt, res in results.items():
        logger.info(f"  {mt}: avg_test_auc={res['avg_test_auc']:.4f}, "
                    f"avg_test_acc={res['avg_test_acc']:.4f}")

    return results


def step_analysis(results=None, X=None, y=None, feature_names=None):
    """步骤4: 模型可解释性分析"""
    logger.info("=" * 60)
    logger.info("步骤4: 模型可解释性分析")
    logger.info("=" * 60)

    if X is None:
        data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, y = data['X'], data['y']
        feature_names = data['feature_names']

    # 加载最佳模型(Transformer-LSTM)
    if results and 'transformer_lstm' in results:
        model = results['transformer_lstm']['best_model']
    else:
        input_size = X.shape[2]
        model = TransformerLSTM(input_size=input_size, **TRANSFORMER_LSTM_CONFIG)
        model_path = os.path.join(MODEL_DIR, "transformer_lstm_best.pt")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            logger.warning("未找到训练好的模型，使用随机初始化模型进行分析")

    # 划分训练/测试集
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    interpreter = ModelInterpreter(model, feature_names)
    analysis_results = interpreter.full_analysis(X_train, X_test, y_test)

    logger.info("可解释性分析完成")
    return analysis_results


def step_rolling_backtest(X=None, y=None, ts=None, feature_names=None):
    """步骤5b: 滚动窗口回测"""
    logger.info("=" * 60)
    logger.info("步骤5b: 滚动窗口回测")
    logger.info("=" * 60)

    if X is None:
        data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, y, ts = data['X'], data['y'], data['timestamps']
        feature_names = data['feature_names']

    df_path = os.path.join(PROCESSED_DATA_DIR, "featured_df.pkl")
    if os.path.exists(df_path):
        df_full = pd.read_pickle(df_path)
    else:
        loader = DataLoader()
        df_full = loader.prepare_data()

    prices_df = pd.DataFrame({'close': df_full['close'].values})

    rb = RollingBacktest()
    result = rb.run(
        X=X, y=y, timestamps=ts,
        prices_df=prices_df, feature_names=feature_names
    )
    return result


def step_backtest(results=None, X=None, y=None, ts=None, feature_names=None):
    """步骤5: 回测"""
    logger.info("=" * 60)
    logger.info("步骤5: 回测和风险管理")
    logger.info("=" * 60)

    if X is None:
        data_path = os.path.join(PROCESSED_DATA_DIR, "processed_data.pkl")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        X, y, ts = data['X'], data['y'], data['timestamps']
        feature_names = data['feature_names']

    # 加载原始价格数据
    df_path = os.path.join(PROCESSED_DATA_DIR, "featured_df.pkl")
    if os.path.exists(df_path):
        df_full = pd.read_pickle(df_path)
    else:
        loader = DataLoader()
        df_full = loader.prepare_data()

    # 使用测试集部分进行回测
    split = int(len(X) * 0.8)
    X_test = X[split:]
    y_test = y[split:]
    ts_test = ts[split:]

    # 构建集成模型
    input_size = X.shape[2]
    seq_length = X.shape[1]
    device = DEVICE

    models = {}
    model_configs = {
        'transformer_lstm': (TransformerLSTM, {**TRANSFORMER_LSTM_CONFIG, 'input_size': input_size}),
        'lstm': (LSTMModel, {**LSTM_CONFIG, 'input_size': input_size}),
        'cnn': (CNNModel, {**CNN_CONFIG, 'input_size': input_size, 'seq_length': seq_length}),
        'mlp': (MLPModel, {**MLP_CONFIG, 'input_size': input_size, 'seq_length': seq_length}),
    }

    for name, (model_cls, config) in model_configs.items():
        model_path = os.path.join(MODEL_DIR, f"{name}_best.pt")
        if results and name in results and results[name]['best_model'] is not None:
            models[name] = results[name]['best_model']
        elif os.path.exists(model_path):
            model = model_cls(**config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            models[name] = model
            logger.info(f"加载模型: {name}")

    if not models:
        logger.warning("没有可用的模型，使用随机预测进行回测演示")
        np.random.seed(42)
        predictions = np.random.randint(0, 2, len(X_test))
        probabilities = np.random.rand(len(X_test))
    else:
        ensemble = EnsembleModel(models, ENSEMBLE_WEIGHTS, device)
        x_tensor = torch.FloatTensor(X_test)
        probabilities = ensemble.predict_proba(x_tensor)
        predictions = (probabilities >= T0_CONFIG['signal_threshold']).astype(int)
        logger.info(f"集成模型预测完成: {len(predictions)} 条")

    # 匹配价格数据
    test_prices = pd.DataFrame({
        'close': df_full['close'].values[-len(X_test):]
    }).reset_index(drop=True)

    engine = BacktestEngine()
    bt_result = engine.run_backtest(
        predictions=predictions,
        probabilities=probabilities,
        prices=test_prices,
        timestamps=ts_test
    )

    return bt_result


def main():
    parser = argparse.ArgumentParser(description="A股T+0量化交易系统")
    parser.add_argument('--step', type=str, default='all',
                       choices=['all', 'data', 'feature', 'train', 'analysis', 'backtest', 'rolling'],
                       help='执行步骤')
    parser.add_argument('--optuna', action='store_true', help='启用Optuna超参数优化')
    parser.add_argument('--rolling', action='store_true', help='使用滚动窗口回测')
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    ensure_dir(RESULTS_DIR)

    logger.info("=" * 60)
    logger.info("A股T+0量化交易系统 - 中芯国际(688981)")
    logger.info("=" * 60)

    if args.step == 'data':
        step_data()
    elif args.step == 'feature':
        step_feature()
    elif args.step == 'train':
        step_train(use_optuna=args.optuna)
    elif args.step == 'analysis':
        step_analysis()
    elif args.step == 'backtest':
        step_backtest()
    elif args.step == 'rolling':
        step_rolling_backtest()
    elif args.step == 'all':
        # 完整流程
        df = step_data()
        X, y, ts, fe = step_feature(df)
        results = step_train(X, y, use_optuna=args.optuna)
        step_analysis(results, X, y, fe.selected_features)
        step_backtest(results, X, y, ts, fe.selected_features)

        logger.info("\n" + "=" * 60)
        logger.info("全部流程执行完毕!")
        logger.info(f"结果目录: {RESULTS_DIR}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
