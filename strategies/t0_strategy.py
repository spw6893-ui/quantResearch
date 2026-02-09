"""
T+0日内交易策略 - 基于VeighNa框架
集成深度学习模型进行5分钟级别预测与交易
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import deque
from datetime import datetime, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    T0_CONFIG, BACKTEST_CONFIG, SEQUENCE_LENGTH, MODEL_DIR,
    TRANSFORMER_LSTM_CONFIG, LSTM_CONFIG, CNN_CONFIG, MLP_CONFIG,
    ENSEMBLE_WEIGHTS, DEVICE
)
from models.transformer_lstm import TransformerLSTM
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from models.ensemble import EnsembleModel
from utils.logger import get_logger

logger = get_logger(__name__, "strategy.log")


try:
    from vnpy.trader.constant import Direction, Offset, Status
    from vnpy.trader.object import BarData, OrderData, TradeData
    VNPY_AVAILABLE = True
except ImportError:
    VNPY_AVAILABLE = False
    logger.warning("VeighNa未安装，策略将以独立模式运行")


class T0SignalEngine:
    """
    T+0信号引擎 (与VeighNa解耦的核心逻辑)
    负责特征计算、模型推理、信号生成
    """

    def __init__(self, scaler=None, feature_names: list = None):
        self.device = DEVICE
        self.seq_length = SEQUENCE_LENGTH
        self.signal_threshold = T0_CONFIG['signal_threshold']
        self.max_daily_trades = T0_CONFIG['max_daily_trades']
        self.min_trade_interval = T0_CONFIG['min_trade_interval']  # 分钟

        self.scaler = scaler
        self.feature_names = feature_names

        # 状态
        self.bar_buffer = deque(maxlen=200)  # 保留最近200根K线用于指标计算
        self.feature_buffer = deque(maxlen=self.seq_length + 10)
        self.daily_trade_count = 0
        self.last_trade_time = None
        self.current_date = None
        self.position = 0
        self.entry_price = 0.0

        # 模型
        self.ensemble = None
        self._load_models()

    def _load_models(self):
        """加载训练好的模型并构建集成"""
        models = {}
        # 需要从已保存的数据中获取 input_size
        # 先尝试从特征名推断
        input_size = len(self.feature_names) if self.feature_names else 50

        model_configs = {
            'transformer_lstm': (
                TransformerLSTM,
                {**TRANSFORMER_LSTM_CONFIG, 'input_size': input_size}
            ),
            'lstm': (
                LSTMModel,
                {**LSTM_CONFIG, 'input_size': input_size}
            ),
            'cnn': (
                CNNModel,
                {**CNN_CONFIG, 'input_size': input_size, 'seq_length': self.seq_length}
            ),
            'mlp': (
                MLPModel,
                {**MLP_CONFIG, 'input_size': input_size, 'seq_length': self.seq_length}
            ),
        }

        for name, (cls, config) in model_configs.items():
            model_path = os.path.join(MODEL_DIR, f"{name}_best.pt")
            if os.path.exists(model_path):
                try:
                    model = cls(**config)
                    model.load_state_dict(
                        torch.load(model_path, map_location=self.device)
                    )
                    model.eval()
                    models[name] = model
                    logger.info(f"加载模型: {name}")
                except Exception as e:
                    logger.warning(f"加载模型 {name} 失败: {e}")

        if models:
            self.ensemble = EnsembleModel(models, ENSEMBLE_WEIGHTS, self.device)
            logger.info(f"集成模型已构建: {list(models.keys())}")
        else:
            logger.warning("未加载任何模型，信号引擎将无法生成预测")

    def on_bar(self, bar_dict: dict) -> dict:
        """
        处理新的K线数据并生成交易信号

        Args:
            bar_dict: {'datetime', 'open', 'high', 'low', 'close', 'volume', 'amount'}

        Returns:
            signal: {'action': 'buy'|'sell'|'hold', 'probability': float,
                     'strength': float, 'reason': str}
        """
        self.bar_buffer.append(bar_dict)
        bar_dt = bar_dict['datetime']

        # 日期切换，重置日内计数
        if isinstance(bar_dt, str):
            bar_dt = pd.Timestamp(bar_dt)
        today = bar_dt.date()
        if self.current_date != today:
            self.current_date = today
            self.daily_trade_count = 0

        # 数据不足
        if len(self.bar_buffer) < 65:  # 至少需要60根K线计算指标
            return {'action': 'hold', 'probability': 0.5,
                    'strength': 0.0, 'reason': 'insufficient_data'}

        # 计算特征
        features = self._compute_features()
        if features is None:
            return {'action': 'hold', 'probability': 0.5,
                    'strength': 0.0, 'reason': 'feature_error'}

        self.feature_buffer.append(features)

        # 序列不够
        if len(self.feature_buffer) < self.seq_length:
            return {'action': 'hold', 'probability': 0.5,
                    'strength': 0.0, 'reason': 'sequence_building'}

        # 模型预测
        signal = self._generate_signal(bar_dict)
        return signal

    def _compute_features(self) -> np.ndarray:
        """从bar_buffer计算特征向量"""
        try:
            df = pd.DataFrame(list(self.bar_buffer))
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])

            from data.feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            fe.feature_names = self.feature_names
            fe.selected_features = self.feature_names

            # 构建特征 (使用内部方法)
            df = fe._add_ma(df)
            df = fe._add_ema(df)
            df = fe._add_rsi(df)
            df = fe._add_macd(df)
            df = fe._add_bollinger(df)
            df = fe._add_atr(df)
            df = fe._add_kdj(df)
            df = fe._add_volume_indicators(df)
            df = fe._add_price_features(df)
            df = fe._add_time_features(df)

            # 取最后一行的选定特征
            last_row = df.iloc[-1]
            feature_vec = []
            for fname in self.feature_names:
                if fname in last_row.index:
                    val = last_row[fname]
                    if np.isfinite(val):
                        feature_vec.append(val)
                    else:
                        feature_vec.append(0.0)
                else:
                    feature_vec.append(0.0)

            feature_vec = np.array(feature_vec, dtype=np.float32)

            # 标准化
            if self.scaler is not None:
                feature_vec = self.scaler.transform(feature_vec.reshape(1, -1)).flatten()

            return feature_vec
        except Exception as e:
            logger.debug(f"特征计算异常: {e}")
            return None

    def _generate_signal(self, bar_dict: dict) -> dict:
        """生成交易信号"""
        # 构建序列
        seq = list(self.feature_buffer)[-self.seq_length:]
        x = np.array(seq, dtype=np.float32).reshape(1, self.seq_length, -1)
        x_tensor = torch.FloatTensor(x)

        if self.ensemble is None:
            return {'action': 'hold', 'probability': 0.5,
                    'strength': 0.0, 'reason': 'no_model'}

        # 预测
        probs = self.ensemble.predict_proba(x_tensor)
        prob_up = probs[0, 1]  # 上涨概率

        bar_dt = bar_dict['datetime']
        if isinstance(bar_dt, str):
            bar_dt = pd.Timestamp(bar_dt)
        price = bar_dict['close']

        # 检查交易限制
        if self.daily_trade_count >= self.max_daily_trades:
            return {'action': 'hold', 'probability': prob_up,
                    'strength': 0.0, 'reason': 'max_daily_trades'}

        if self.last_trade_time is not None:
            elapsed = (bar_dt - self.last_trade_time).total_seconds() / 60
            if elapsed < self.min_trade_interval:
                return {'action': 'hold', 'probability': prob_up,
                        'strength': 0.0, 'reason': 'min_interval'}

        # 止损/止盈检查
        if self.position > 0 and self.entry_price > 0:
            unrealized = (price - self.entry_price) / self.entry_price
            if unrealized <= -BACKTEST_CONFIG['stop_loss_pct']:
                return {'action': 'sell', 'probability': prob_up,
                        'strength': 1.0, 'reason': 'stop_loss'}
            if unrealized >= BACKTEST_CONFIG['take_profit_pct']:
                return {'action': 'sell', 'probability': prob_up,
                        'strength': 1.0, 'reason': 'take_profit'}

        # 信号判断
        strength = abs(prob_up - 0.5) * 2  # 归一化到0-1

        if prob_up >= self.signal_threshold and self.position == 0:
            return {'action': 'buy', 'probability': prob_up,
                    'strength': strength, 'reason': 'model_signal'}
        elif prob_up < (1 - self.signal_threshold) and self.position > 0:
            return {'action': 'sell', 'probability': prob_up,
                    'strength': strength, 'reason': 'model_signal'}

        return {'action': 'hold', 'probability': prob_up,
                'strength': strength, 'reason': 'below_threshold'}

    def update_position(self, action: str, price: float, shares: int, dt):
        """更新持仓状态"""
        if action == 'buy':
            self.position += shares
            self.entry_price = price
        elif action == 'sell':
            self.position = 0
            self.entry_price = 0.0
        self.daily_trade_count += 1
        self.last_trade_time = pd.Timestamp(dt)


class T0Strategy:
    """
    T+0日内交易策略 (VeighNa集成版本)
    可独立运行或作为VeighNa CTA策略使用
    """

    # 策略参数
    author = "QuantResearch"
    strategy_name = "T0_ML_Strategy"

    def __init__(self, scaler=None, feature_names: list = None):
        self.signal_engine = T0SignalEngine(
            scaler=scaler, feature_names=feature_names
        )
        self.config = {**BACKTEST_CONFIG, **T0_CONFIG}

        # 账户
        self.cash = self.config['initial_capital']
        self.position = 0
        self.entry_price = 0.0
        self.equity_history = []
        self.trades = []

        logger.info(f"T0策略初始化: 初始资金={self.cash:,.0f}")

    def on_bar(self, bar_dict: dict):
        """处理K线，生成并执行信号"""
        signal = self.signal_engine.on_bar(bar_dict)
        price = bar_dict['close']
        dt = bar_dict['datetime']

        if signal['action'] == 'buy' and self.position == 0:
            self._execute_buy(price, signal, dt)
        elif signal['action'] == 'sell' and self.position > 0:
            self._execute_sell(price, signal, dt)

        equity = self.cash + self.position * price
        self.equity_history.append({
            'datetime': dt, 'equity': equity,
            'cash': self.cash, 'position': self.position
        })

    def _execute_buy(self, price: float, signal: dict, dt):
        """执行买入"""
        slippage = self.config['slippage']
        commission = self.config['commission_rate']
        max_pos = self.config['max_position_ratio']

        buy_price = price * (1 + slippage)

        # 动态仓位
        prob = signal['probability']
        if prob >= 0.8:
            ratio = 0.9
        elif prob >= 0.7:
            ratio = 0.6
        else:
            ratio = 0.3

        invest = self.cash * max_pos * ratio
        shares = int(invest / buy_price / 100) * 100

        if shares >= 100:
            cost = shares * buy_price * commission
            total = shares * buy_price + cost
            if total <= self.cash:
                self.cash -= total
                self.position = shares
                self.entry_price = buy_price
                self.signal_engine.update_position('buy', buy_price, shares, dt)
                self.trades.append({
                    'datetime': dt, 'action': 'buy', 'price': buy_price,
                    'shares': shares, 'cost': cost, 'reason': signal['reason'],
                    'probability': signal['probability']
                })
                logger.debug(f"买入 {shares}股 @ {buy_price:.2f}, "
                           f"概率={prob:.3f}, 原因={signal['reason']}")

    def _execute_sell(self, price: float, signal: dict, dt):
        """执行卖出"""
        slippage = self.config['slippage']
        commission = self.config['commission_rate']
        stamp_tax = self.config['stamp_tax']

        sell_price = price * (1 - slippage)
        revenue = self.position * sell_price
        cost = revenue * (commission + stamp_tax)
        pnl = (sell_price - self.entry_price) * self.position - cost

        self.cash += revenue - cost
        self.trades.append({
            'datetime': dt, 'action': 'sell', 'price': sell_price,
            'shares': self.position, 'cost': cost, 'pnl': pnl,
            'reason': signal['reason'], 'probability': signal['probability']
        })
        logger.debug(f"卖出 {self.position}股 @ {sell_price:.2f}, "
                   f"PnL={pnl:.2f}, 原因={signal['reason']}")
        self.position = 0
        self.entry_price = 0.0
        self.signal_engine.update_position('sell', sell_price, 0, dt)

    def get_results(self) -> dict:
        """获取策略执行结果"""
        return {
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'equity_history': pd.DataFrame(self.equity_history),
            'final_cash': self.cash,
            'final_position': self.position,
        }


if VNPY_AVAILABLE:
    from vnpy.trader.utility import BarGenerator

    class VnpyT0Strategy:
        """
        VeighNa CTA策略适配器
        将T0Strategy的信号逻辑桥接到VeighNa框架
        """

        author = "QuantResearch"
        parameters = [
            "signal_threshold", "max_daily_trades",
            "stop_loss_pct", "take_profit_pct"
        ]
        variables = [
            "position", "daily_trades", "last_signal"
        ]

        def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
            self.cta_engine = cta_engine
            self.strategy_name = strategy_name
            self.vt_symbol = vt_symbol

            self.signal_engine = T0SignalEngine(
                scaler=setting.get('scaler'),
                feature_names=setting.get('feature_names')
            )

            self.bg = BarGenerator(self.on_bar, 5, self.on_5min_bar)
            self.daily_trades = 0
            self.last_signal = ""

        def on_init(self):
            logger.info(f"{self.strategy_name} 策略初始化")

        def on_start(self):
            logger.info(f"{self.strategy_name} 策略启动")

        def on_stop(self):
            logger.info(f"{self.strategy_name} 策略停止")

        def on_bar(self, bar: 'BarData'):
            self.bg.update_bar(bar)

        def on_5min_bar(self, bar: 'BarData'):
            bar_dict = {
                'datetime': bar.datetime,
                'open': bar.open_price,
                'high': bar.high_price,
                'low': bar.low_price,
                'close': bar.close_price,
                'volume': bar.volume,
                'amount': bar.turnover,
            }
            signal = self.signal_engine.on_bar(bar_dict)
            self.last_signal = f"{signal['action']}({signal['probability']:.3f})"

            pos = self.cta_engine.get_position(self.vt_symbol)
            if signal['action'] == 'buy' and pos == 0:
                self.cta_engine.buy(self.vt_symbol, bar.close_price * 1.01, 100)
            elif signal['action'] == 'sell' and pos > 0:
                self.cta_engine.sell(self.vt_symbol, bar.close_price * 0.99, pos)

        def on_order(self, order: 'OrderData'):
            pass

        def on_trade(self, trade: 'TradeData'):
            self.daily_trades += 1
            self.signal_engine.update_position(
                'buy' if trade.direction == Direction.LONG else 'sell',
                trade.price, trade.volume, trade.datetime
            )
            logger.info(
                f"成交: {trade.direction.value} {trade.volume}@{trade.price:.2f}"
            )
