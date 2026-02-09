"""
回测和风险管理模块
滚动窗口回测、交易成本、绩效指标、仓位管理、止损机制
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BACKTEST_CONFIG, T0_CONFIG, RESULTS_DIR
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__, "backtest.log")


class BacktestEngine:
    """回测引擎: 滚动窗口回测系统"""

    def __init__(self, config: dict = None):
        self.config = config or BACKTEST_CONFIG
        self.t0_config = T0_CONFIG
        self.results_dir = os.path.join(RESULTS_DIR, "backtest")
        ensure_dir(self.results_dir)

    def run_backtest(self, predictions: np.ndarray, probabilities: np.ndarray,
                     prices: pd.DataFrame, timestamps: np.ndarray) -> dict:
        """
        执行回测
        Args:
            predictions: 模型预测 (0/1)
            probabilities: 预测概率 (0~1)
            prices: 包含OHLC的DataFrame
            timestamps: 时间戳
        """
        logger.info("开始回测...")

        initial_capital = self.config['initial_capital']
        commission = self.config['commission_rate']
        slippage = self.config['slippage']
        stamp_tax = self.config['stamp_tax']
        stop_loss = self.config['stop_loss_pct']
        take_profit = self.config['take_profit_pct']
        max_position_ratio = self.config['max_position_ratio']
        signal_threshold = self.t0_config['signal_threshold']

        # 交易记录
        trades = []
        daily_records = []
        equity_curve = [initial_capital]

        cash = initial_capital
        position = 0  # 持仓股数
        entry_price = 0
        daily_trades_count = 0
        last_trade_time = None
        current_date = None

        for i in range(len(predictions)):
            ts = pd.Timestamp(timestamps[i])
            price = prices.iloc[i]['close'] if isinstance(prices, pd.DataFrame) else prices[i]

            # 日期切换重置
            if current_date != ts.date():
                if current_date is not None:
                    # 记录当日结果
                    equity = cash + position * price
                    daily_records.append({
                        'date': current_date,
                        'equity': equity,
                        'cash': cash,
                        'position': position,
                        'daily_trades': daily_trades_count
                    })
                current_date = ts.date()
                daily_trades_count = 0

            equity = cash + position * price
            equity_curve.append(equity)

            # T+0约束: 日内交易次数限制
            if daily_trades_count >= self.t0_config['max_daily_trades']:
                continue

            # 交易间隔限制
            if last_trade_time is not None:
                min_interval = pd.Timedelta(minutes=self.t0_config['min_trade_interval'])
                if ts - last_trade_time < min_interval:
                    continue

            prob = probabilities[i]
            pred = predictions[i]

            # ===== 止损/止盈检查 =====
            if position > 0:
                unrealized_pnl = (price - entry_price) / entry_price
                if unrealized_pnl <= -stop_loss:
                    # 止损卖出
                    sell_price = price * (1 - slippage)
                    revenue = position * sell_price
                    cost = revenue * (commission + stamp_tax)
                    cash += revenue - cost
                    trades.append({
                        'datetime': ts, 'action': 'sell_stop_loss',
                        'price': sell_price, 'shares': position,
                        'cost': cost, 'pnl': (sell_price - entry_price) * position - cost
                    })
                    position = 0
                    entry_price = 0
                    daily_trades_count += 1
                    last_trade_time = ts
                    continue

                if unrealized_pnl >= take_profit:
                    # 止盈卖出
                    sell_price = price * (1 - slippage)
                    revenue = position * sell_price
                    cost = revenue * (commission + stamp_tax)
                    cash += revenue - cost
                    trades.append({
                        'datetime': ts, 'action': 'sell_take_profit',
                        'price': sell_price, 'shares': position,
                        'cost': cost, 'pnl': (sell_price - entry_price) * position - cost
                    })
                    position = 0
                    entry_price = 0
                    daily_trades_count += 1
                    last_trade_time = ts
                    continue

            # ===== 信号交易 =====
            if prob >= signal_threshold and pred == 1 and position == 0:
                # 买入信号
                buy_price = price * (1 + slippage)
                max_invest = cash * max_position_ratio
                # 动态仓位: 根据信号强度调整
                if prob >= 0.8:
                    invest = max_invest * 0.9
                elif prob >= 0.7:
                    invest = max_invest * 0.6
                else:
                    invest = max_invest * 0.3

                shares = int(invest / buy_price / 100) * 100  # 按手买入
                if shares >= 100:
                    cost = shares * buy_price * commission
                    total_cost = shares * buy_price + cost
                    if total_cost <= cash:
                        cash -= total_cost
                        position = shares
                        entry_price = buy_price
                        daily_trades_count += 1
                        last_trade_time = ts
                        trades.append({
                            'datetime': ts, 'action': 'buy',
                            'price': buy_price, 'shares': shares,
                            'cost': cost, 'pnl': 0
                        })

            elif prob < (1 - signal_threshold) and pred == 0 and position > 0:
                # 卖出信号
                sell_price = price * (1 - slippage)
                revenue = position * sell_price
                cost = revenue * (commission + stamp_tax)
                pnl = (sell_price - entry_price) * position - cost
                cash += revenue - cost
                trades.append({
                    'datetime': ts, 'action': 'sell_signal',
                    'price': sell_price, 'shares': position,
                    'cost': cost, 'pnl': pnl
                })
                position = 0
                entry_price = 0
                daily_trades_count += 1
                last_trade_time = ts

        # 最终清仓
        if position > 0:
            final_price = prices.iloc[-1]['close'] if isinstance(prices, pd.DataFrame) else prices[-1]
            sell_price = final_price * (1 - slippage)
            revenue = position * sell_price
            cost = revenue * (commission + stamp_tax)
            cash += revenue - cost
            trades.append({
                'datetime': timestamps[-1], 'action': 'sell_final',
                'price': sell_price, 'shares': position,
                'cost': cost, 'pnl': (sell_price - entry_price) * position - cost
            })
            position = 0

        final_equity = cash
        equity_curve.append(final_equity)

        # 计算绩效指标
        metrics = self._calculate_metrics(equity_curve, trades, initial_capital)

        logger.info(f"回测完成:")
        logger.info(f"  总收益率: {metrics['total_return']*100:.2f}%")
        logger.info(f"  年化收益率: {metrics['annual_return']*100:.2f}%")
        logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  索提诺比率: {metrics['sortino_ratio']:.4f}")
        logger.info(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        logger.info(f"  Calmar比率: {metrics['calmar_ratio']:.4f}")
        logger.info(f"  交易次数: {metrics['total_trades']}")
        logger.info(f"  胜率: {metrics['win_rate']*100:.1f}%")

        result = {
            'metrics': metrics,
            'trades': pd.DataFrame(trades),
            'equity_curve': equity_curve,
            'daily_records': pd.DataFrame(daily_records)
        }

        self._plot_results(result)
        return result

    def _calculate_metrics(self, equity_curve: list, trades: list,
                          initial_capital: float) -> dict:
        """计算关键绩效指标"""
        equity = np.array(equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[np.isfinite(returns)]

        # 总收益率
        total_return = (equity[-1] - initial_capital) / initial_capital

        # 年化收益率 (假设每年250个交易日，每天48个5分钟K线)
        n_periods = len(returns)
        periods_per_year = 250 * 48
        annual_return = (1 + total_return) ** (periods_per_year / max(n_periods, 1)) - 1

        # 夏普比率
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        else:
            sharpe = 0

        # 索提诺比率
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(periods_per_year)
        else:
            sortino = 0

        # 最大回撤
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Calmar比率
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        # 交易统计
        trade_pnls = [t['pnl'] for t in trades if t['action'].startswith('sell')]
        total_trades = len(trade_pnls)
        win_trades = len([p for p in trade_pnls if p > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        avg_win = np.mean([p for p in trade_pnls if p > 0]) if win_trades > 0 else 0
        avg_loss = np.mean([p for p in trade_pnls if p <= 0]) if (total_trades - win_trades) > 0 else 0
        profit_factor = abs(avg_win * win_trades / (avg_loss * (total_trades - win_trades))) \
            if avg_loss != 0 and (total_trades - win_trades) > 0 else 0

        total_costs = sum(t['cost'] for t in trades)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_costs': total_costs,
            'final_equity': equity[-1],
            'initial_capital': initial_capital,
        }

    def _plot_results(self, result: dict):
        """绘制回测结果"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # 1. 权益曲线
        equity = result['equity_curve']
        axes[0].plot(equity, color='steelblue', linewidth=1)
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Portfolio Value (CNY)')
        axes[0].axhline(y=self.config['initial_capital'], color='gray',
                        linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 回撤
        equity_arr = np.array(equity)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak * 100
        axes[1].fill_between(range(len(drawdown)), drawdown, alpha=0.5, color='red')
        axes[1].set_title('Drawdown (%)')
        axes[1].set_ylabel('Drawdown %')
        axes[1].grid(True, alpha=0.3)

        # 3. 交易盈亏分布
        if len(result['trades']) > 0:
            sell_trades = result['trades'][
                result['trades']['action'].str.startswith('sell')
            ]
            if len(sell_trades) > 0:
                pnls = sell_trades['pnl'].values
                colors = ['green' if p > 0 else 'red' for p in pnls]
                axes[2].bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
                axes[2].set_title('Trade P&L')
                axes[2].set_ylabel('Profit/Loss (CNY)')
                axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.results_dir, 'backtest_results.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"回测结果图已保存: {save_path}")

        # 保存指标
        metrics_path = os.path.join(self.results_dir, 'metrics.csv')
        pd.DataFrame([result['metrics']]).to_csv(metrics_path, index=False)
        logger.info(f"绩效指标已保存: {metrics_path}")

        if len(result['trades']) > 0:
            trades_path = os.path.join(self.results_dir, 'trades.csv')
            result['trades'].to_csv(trades_path, index=False)
            logger.info(f"交易记录已保存: {trades_path}")
