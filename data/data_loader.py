"""
数据加载与预处理模块
支持从BaoStock/AKShare/Tushare获取中芯国际(688981)分钟级数据
"""
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, TUSHARE_TOKEN,
    TS_SYMBOL, DATA_START_DATE, DATA_END_DATE, SYMBOL
)
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__, "data_loader.log")


class DataLoader:
    """数据加载器: 获取和管理股票分钟级数据"""

    def __init__(self, symbol: str = TS_SYMBOL):
        self.symbol = symbol
        ensure_dir(RAW_DATA_DIR)
        ensure_dir(PROCESSED_DATA_DIR)

    def fetch_from_baostock(self, start_date: str = None, end_date: str = None,
                            freq: str = "5", adjust: str = "2") -> pd.DataFrame:
        """
        从BaoStock获取分钟级数据 (免费, 无需token, 可回溯至上市日)
        按季度分段获取，避免单次查询迭代过慢
        Args:
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD
            freq: "5", "15", "30", "60"
            adjust: "1"后复权 "2"前复权 "3"不复权
        """
        import baostock as bs

        if start_date is None:
            start_date = datetime.strptime(DATA_START_DATE, "%Y%m%d").strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.strptime(DATA_END_DATE, "%Y%m%d").strftime("%Y-%m-%d")

        bs_symbol = f"sh.{SYMBOL}"
        logger.info(f"从BaoStock获取 {bs_symbol} {freq}分钟K线数据: {start_date} -> {end_date}")

        # 按季度(90天)分段获取
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        chunk_days = 90
        all_chunks = []
        current = start_dt

        while current < end_dt:
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)
            s = current.strftime("%Y-%m-%d")
            e = chunk_end.strftime("%Y-%m-%d")

            lg = bs.login()
            if lg.error_code != '0':
                logger.error(f"BaoStock登录失败: {lg.error_msg}")
                current = chunk_end + timedelta(days=1)
                continue

            try:
                rs = bs.query_history_k_data_plus(
                    bs_symbol,
                    "date,time,open,high,low,close,volume,amount",
                    start_date=s, end_date=e,
                    frequency=freq, adjustflag=adjust
                )
                if rs.error_code != '0':
                    logger.warning(f"  查询失败 [{s}~{e}]: {rs.error_msg}")
                    current = chunk_end + timedelta(days=1)
                    continue

                data = []
                while rs.error_code == '0' and rs.next():
                    data.append(rs.get_row_data())

                if data:
                    chunk_df = pd.DataFrame(data, columns=rs.fields)
                    all_chunks.append(chunk_df)
                    logger.info(f"  获取 [{s} ~ {e}]: {len(data)} 条")
                else:
                    logger.info(f"  [{s} ~ {e}]: 无数据")
            finally:
                bs.logout()

            current = chunk_end + timedelta(days=1)

        if not all_chunks:
            logger.error("BaoStock未获取到任何数据")
            return pd.DataFrame()

        df = pd.concat(all_chunks, ignore_index=True)

        # 预处理: time格式 20240102093500000 -> datetime
        df['datetime'] = df['time'].apply(
            lambda t: datetime.strptime(t[:14], "%Y%m%d%H%M%S")
        )

        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        df = df[(df['close'] > 0) & (df['volume'] > 0)]
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)

        # 保存
        save_path = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_5min.pkl")
        df.to_pickle(save_path)
        logger.info(f"BaoStock数据已保存: {save_path}, 共 {len(df)} 条")
        logger.info(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")

        return df

    def fetch_from_akshare(self, period: str = "5", adjust: str = "qfq") -> pd.DataFrame:
        """从AKShare获取分钟级数据 (免费, 无需token)"""
        import akshare as ak

        # 科创板688981 -> sh688981
        ak_symbol = f"sh{SYMBOL}"
        logger.info(f"从AKShare获取 {ak_symbol} {period}分钟K线数据 (前复权={adjust})")

        try:
            df = ak.stock_zh_a_minute(symbol=ak_symbol, period=period, adjust=adjust)
        except Exception as e:
            logger.error(f"AKShare获取数据失败: {e}")
            return pd.DataFrame()

        if df is None or len(df) == 0:
            logger.error("AKShare未返回数据")
            return pd.DataFrame()

        # 预处理
        df = df.rename(columns={'day': 'datetime'})
        df['datetime'] = pd.to_datetime(df['datetime'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('datetime').reset_index(drop=True)

        # AKShare没有amount列, 用 close*volume 近似
        if 'amount' not in df.columns:
            df['amount'] = (df['close'] * df['volume']).round(2)

        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        # 保存
        save_path = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_5min.pkl")
        df.to_pickle(save_path)
        logger.info(f"AKShare数据已保存: {save_path}, 共 {len(df)} 条")
        logger.info(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")

        return df

    def fetch_from_tushare(self, start_date: str = DATA_START_DATE,
                           end_date: str = DATA_END_DATE,
                           freq: str = "5min") -> pd.DataFrame:
        """从Tushare获取分钟级数据"""
        import tushare as ts
        if not TUSHARE_TOKEN:
            logger.warning("TUSHARE_TOKEN未设置,尝试从本地加载数据")
            return self.load_local_data()

        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()

        logger.info(f"从Tushare获取 {self.symbol} {freq} 数据: {start_date} -> {end_date}")

        all_data = []
        # Tushare分钟数据需要分段获取
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        chunk_days = 30  # 每次获取30天

        current = start
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            try:
                df = ts.pro_bar(
                    ts_code=self.symbol,
                    freq=freq,
                    start_date=current.strftime("%Y%m%d"),
                    end_date=chunk_end.strftime("%Y%m%d"),
                    adj="qfq"  # 前复权
                )
                if df is not None and len(df) > 0:
                    all_data.append(df)
                    logger.info(f"  获取 {current.strftime('%Y%m%d')}-{chunk_end.strftime('%Y%m%d')}: {len(df)} 条")
            except Exception as e:
                logger.error(f"  获取数据失败: {e}")
            current = chunk_end + timedelta(days=1)

        if not all_data:
            logger.error("未获取到任何数据")
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df = self._preprocess_tushare_data(df)

        # 保存原始数据
        save_path = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_{freq}.pkl")
        df.to_pickle(save_path)
        logger.info(f"原始数据已保存: {save_path}, 共 {len(df)} 条")

        return df

    def _preprocess_tushare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理Tushare数据"""
        df = df.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.sort_values('trade_time').reset_index(drop=True)
        df = df.rename(columns={
            'trade_time': 'datetime',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume',
            'amount': 'amount'
        })
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        return df

    def generate_synthetic_data(self, n_days: int = 500, freq_minutes: int = 5) -> pd.DataFrame:
        """生成模拟5分钟K线数据(用于开发和测试)"""
        logger.info(f"生成 {n_days} 天的模拟{freq_minutes}分钟K线数据")
        np.random.seed(42)

        bars_per_day = 240 // freq_minutes  # A股每天240分钟
        total_bars = n_days * bars_per_day

        # 生成交易日期序列
        dates = pd.bdate_range(start="2023-01-03", periods=n_days, freq="B")
        # 生成每天的分钟时间
        intraday_times = []
        for i in range(bars_per_day):
            hour = 9 + (i * freq_minutes + 30) // 60
            minute = (i * freq_minutes + 30) % 60
            if hour == 11 and minute > 30:
                hour = 13
                minute = minute - 30
            elif hour >= 12 and hour < 13:
                hour += 1.5
            intraday_times.append(f"{int(hour):02d}:{int(minute):02d}:00")

        # 简化: 生成连续时间序列
        datetimes = []
        for d in dates:
            for t_idx in range(bars_per_day):
                total_min = t_idx * freq_minutes
                if total_min < 120:  # 上午 9:30 - 11:30
                    h = 9 + (total_min + 30) // 60
                    m = (total_min + 30) % 60
                else:  # 下午 13:00 - 15:00
                    afternoon_min = total_min - 120
                    h = 13 + afternoon_min // 60
                    m = afternoon_min % 60
                dt = d.replace(hour=int(h), minute=int(m), second=0)
                datetimes.append(dt)

        # 生成价格序列 (GBM模型)
        base_price = 80.0  # 中芯国际大致价格区间
        mu = 0.0001  # 微小漂移
        sigma = 0.008  # 5分钟波动率

        returns = np.random.normal(mu, sigma, total_bars)
        # 添加均值回归特性
        prices = np.zeros(total_bars)
        prices[0] = base_price
        for i in range(1, total_bars):
            mean_revert = -0.001 * (prices[i-1] - base_price) / base_price
            prices[i] = prices[i-1] * (1 + returns[i] + mean_revert)

        # 生成OHLCV
        opens = prices.copy()
        noise = np.abs(np.random.normal(0, sigma * 0.5, total_bars))
        highs = prices * (1 + noise)
        lows = prices * (1 - noise)
        closes = prices + np.random.normal(0, sigma * 0.3, total_bars) * prices

        # 确保OHLC关系正确
        for i in range(total_bars):
            bar_prices = [opens[i], closes[i], highs[i], lows[i]]
            highs[i] = max(bar_prices)
            lows[i] = min(bar_prices)

        # 成交量: 与价格变动相关
        base_volume = 50000
        volume_noise = np.abs(np.random.normal(base_volume, base_volume * 0.5, total_bars))
        abs_returns = np.abs(np.diff(np.log(prices), prepend=np.log(prices[0])))
        volumes = volume_noise * (1 + abs_returns * 50)

        # 金额
        amounts = closes * volumes

        df = pd.DataFrame({
            'datetime': datetimes[:total_bars],
            'open': np.round(opens, 2),
            'high': np.round(highs, 2),
            'low': np.round(lows, 2),
            'close': np.round(closes, 2),
            'volume': np.round(volumes, 0).astype(int),
            'amount': np.round(amounts, 2)
        })

        # 保存
        save_path = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_5min_synthetic.pkl")
        df.to_pickle(save_path)
        logger.info(f"模拟数据已保存: {save_path}, 共 {len(df)} 条")
        return df

    def load_local_data(self, filename: str = None) -> pd.DataFrame:
        """从本地加载数据"""
        if filename is None:
            # 优先加载真实数据，其次尝试AKShare，最后模拟数据
            real_path = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_5min.pkl")
            synth_path = os.path.join(RAW_DATA_DIR, f"{SYMBOL}_5min_synthetic.pkl")
            if os.path.exists(real_path):
                df = pd.read_pickle(real_path)
                logger.info(f"加载真实数据: {len(df)} 条")
                return df
            else:
                # 尝试从BaoStock获取(可回溯至上市日)
                logger.info("本地无真实数据，尝试从BaoStock获取...")
                try:
                    df = self.fetch_from_baostock()
                    if len(df) > 0:
                        return df
                except Exception as e:
                    logger.warning(f"BaoStock获取失败: {e}")

                # 其次尝试AKShare(最近1-2个月)
                logger.info("尝试从AKShare获取...")
                try:
                    df = self.fetch_from_akshare()
                    if len(df) > 0:
                        return df
                except Exception as e:
                    logger.warning(f"AKShare获取失败: {e}")

                if os.path.exists(synth_path):
                    df = pd.read_pickle(synth_path)
                    logger.info(f"加载模拟数据: {len(df)} 条")
                    return df
                else:
                    logger.info("未找到任何数据，生成模拟数据")
                    return self.generate_synthetic_data()
        else:
            path = os.path.join(RAW_DATA_DIR, filename)
            if os.path.exists(path):
                return pd.read_pickle(path)
            raise FileNotFoundError(f"数据文件不存在: {path}")

    def prepare_data(self) -> pd.DataFrame:
        """准备数据: 加载 + 基本清洗"""
        df = self.load_local_data()

        # 去除异常值
        df = df[(df['close'] > 0) & (df['volume'] > 0)].copy()

        # 添加日期列
        df['date'] = df['datetime'].dt.date
        df['time'] = df['datetime'].dt.time
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute

        logger.info(f"数据准备完成: {len(df)} 条, 时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        return df


if __name__ == "__main__":
    loader = DataLoader()
    # 生成模拟数据用于开发测试
    df = loader.generate_synthetic_data(n_days=500)
    print(f"\n数据概览:")
    print(df.head(10))
    print(f"\n数据统计:")
    print(df.describe())
    print(f"\n数据量: {len(df)} 条")
