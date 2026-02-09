"""
特征工程模块
构建多类技术指标，进行自动化特征工程，提取时序特征，特征选择
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, f_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    MA_PERIODS, EMA_PERIODS, RSI_PERIODS, MACD_PARAMS, BOLL_PERIOD, BOLL_STD,
    ATR_PERIOD, KDJ_PARAMS, CCI_PERIOD, WILLIAMS_PERIOD, MFI_PERIOD,
    LOOKBACK_WINDOW, PREDICT_HORIZON, SEQUENCE_LENGTH, MAX_FEATURES,
    FEATURE_SELECTION_METHOD, FEATURES_DIR
)
from utils.logger import get_logger
from utils.helpers import ensure_dir

logger = get_logger(__name__, "feature_engineering.log")


class FeatureEngineer:
    """特征工程器: 构建技术指标和时序特征"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.selected_features = None
        self.feature_names = None
        ensure_dir(FEATURES_DIR)

    # ============ 技术指标 ============

    def _add_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """移动平均线"""
        for p in MA_PERIODS:
            df[f'ma_{p}'] = df['close'].rolling(window=p).mean()
            df[f'ma_{p}_slope'] = df[f'ma_{p}'].pct_change(5)
            df[f'close_ma_{p}_ratio'] = df['close'] / df[f'ma_{p}']
        # 均线交叉信号
        for i in range(len(MA_PERIODS)):
            for j in range(i+1, len(MA_PERIODS)):
                short, long = MA_PERIODS[i], MA_PERIODS[j]
                df[f'ma_cross_{short}_{long}'] = df[f'ma_{short}'] - df[f'ma_{long}']
        return df

    def _add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """指数移动平均线"""
        for p in EMA_PERIODS:
            df[f'ema_{p}'] = df['close'].ewm(span=p, adjust=False).mean()
            df[f'close_ema_{p}_ratio'] = df['close'] / df[f'ema_{p}']
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI指标"""
        for p in RSI_PERIODS:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{p}'] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD指标"""
        fast, slow, signal = MACD_PARAMS
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = np.sign(df['macd_hist'])
        return df

    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        """布林带"""
        ma = df['close'].rolling(window=BOLL_PERIOD).mean()
        std = df['close'].rolling(window=BOLL_PERIOD).std()
        df['boll_upper'] = ma + BOLL_STD * std
        df['boll_lower'] = ma - BOLL_STD * std
        df['boll_mid'] = ma
        df['boll_width'] = (df['boll_upper'] - df['boll_lower']) / df['boll_mid']
        df['boll_pct'] = (df['close'] - df['boll_lower']) / (df['boll_upper'] - df['boll_lower'] + 1e-10)
        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR指标"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=ATR_PERIOD).mean()
        df['atr_ratio'] = df['atr'] / df['close']
        return df

    def _add_kdj(self, df: pd.DataFrame) -> pd.DataFrame:
        """KDJ指标"""
        n, m1, m2 = KDJ_PARAMS
        low_min = df['low'].rolling(window=n).min()
        high_max = df['high'].rolling(window=n).max()
        rsv = (df['close'] - low_min) / (high_max - low_min + 1e-10) * 100
        df['kdj_k'] = rsv.ewm(com=m1-1, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=m2-1, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        return df

    def _add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """CCI指标"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma_tp = tp.rolling(window=CCI_PERIOD).mean()
        md = tp.rolling(window=CCI_PERIOD).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - ma_tp) / (0.015 * md + 1e-10)
        return df

    def _add_williams(self, df: pd.DataFrame) -> pd.DataFrame:
        """威廉指标"""
        high_max = df['high'].rolling(window=WILLIAMS_PERIOD).max()
        low_min = df['low'].rolling(window=WILLIAMS_PERIOD).min()
        df['williams_r'] = (high_max - df['close']) / (high_max - low_min + 1e-10) * -100
        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """成交量指标"""
        # OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ma5'] = df['obv'].rolling(5).mean()
        df['obv_slope'] = df['obv'].pct_change(5)

        # 成交量均线
        for p in [5, 10, 20]:
            df[f'vol_ma_{p}'] = df['volume'].rolling(p).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma_5'] + 1e-10)

        # VWAP
        df['vwap'] = (df['amount'].cumsum()) / (df['volume'].cumsum() + 1e-10)
        df['close_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-10)

        # MFI
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        pos_mf = mf.where(tp > tp.shift(1), 0).rolling(MFI_PERIOD).sum()
        neg_mf = mf.where(tp <= tp.shift(1), 0).rolling(MFI_PERIOD).sum()
        df['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """价格衍生特征"""
        # 收益率
        for p in [1, 3, 5, 10, 20]:
            df[f'return_{p}'] = df['close'].pct_change(p)

        # 波动率
        for p in [5, 10, 20]:
            df[f'volatility_{p}'] = df['close'].pct_change().rolling(p).std()

        # K线形态特征
        df['body'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['open'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['open'] + 1e-10)
        df['body_ratio'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)

        # 价格动量
        df['momentum_5'] = df['close'] - df['close'].shift(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        df['roc_5'] = df['close'].pct_change(5)
        df['roc_10'] = df['close'].pct_change(10)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """时间特征"""
        if 'hour' not in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['day_of_week'] = df['datetime'].dt.dayofweek

        # 日内时段编码
        df['intraday_period'] = (df['hour'] * 60 + df['minute'] - 570) / 240  # 归一化到0-1
        df['is_morning'] = (df['hour'] < 12).astype(int)
        df['is_open_30min'] = ((df['hour'] == 9) & (df['minute'] < 60)).astype(int)
        df['is_close_30min'] = ((df['hour'] == 14) & (df['minute'] >= 30)).astype(int)

        # 周期性编码
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)

        return df

    # ============ 特征构建主流程 ============

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建全部技术指标和特征"""
        logger.info("开始构建特征...")
        df = df.copy()

        # 技术指标
        df = self._add_ma(df)
        df = self._add_ema(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger(df)
        df = self._add_atr(df)
        df = self._add_kdj(df)
        df = self._add_cci(df)
        df = self._add_williams(df)
        df = self._add_volume_indicators(df)
        df = self._add_price_features(df)
        df = self._add_time_features(df)

        # 创建标签: 未来1期收益率是否为正
        df['future_return'] = df['close'].shift(-PREDICT_HORIZON) / df['close'] - 1
        df['label'] = (df['future_return'] > 0).astype(int)

        # 去除NaN行
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"特征构建完成: {len(df)} 条 (去除 {initial_len - len(df)} 条NaN)")

        self.feature_names = [c for c in df.columns if c not in
                             ['datetime', 'date', 'time', 'open', 'high', 'low', 'close',
                              'volume', 'amount', 'future_return', 'label',
                              'day_of_week', 'hour', 'minute']]
        logger.info(f"特征数量: {len(self.feature_names)}")

        return df

    # ============ 特征选择 ============

    def select_features(self, df: pd.DataFrame, method: str = FEATURE_SELECTION_METHOD,
                       max_features: int = MAX_FEATURES) -> list:
        """特征选择"""
        logger.info(f"开始特征选择 (方法: {method}, 最大特征数: {max_features})")

        X = df[self.feature_names].values
        y = df['label'].values

        # 替换inf和nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if method == "mutual_info":
            scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
        elif method == "f_classif":
            scores, _ = f_classif(X, y)
            scores = np.nan_to_num(scores, nan=0.0)
        elif method == "random_forest":
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            scores = rf.feature_importances_
        else:
            raise ValueError(f"未知的特征选择方法: {method}")

        # 排序并选择top特征
        feature_scores = list(zip(self.feature_names, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)

        self.selected_features = [f[0] for f in feature_scores[:max_features]]
        logger.info(f"选择的Top 10特征:")
        for fname, fscore in feature_scores[:10]:
            logger.info(f"  {fname}: {fscore:.4f}")

        # 保存特征选择结果
        import pickle
        save_path = os.path.join(FEATURES_DIR, "selected_features.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump({
                'selected_features': self.selected_features,
                'feature_scores': feature_scores,
                'method': method
            }, f)

        return self.selected_features

    # ============ 序列构建 ============

    def create_sequences(self, df: pd.DataFrame, feature_cols: list = None,
                        seq_length: int = SEQUENCE_LENGTH) -> tuple:
        """创建时序序列数据用于模型输入

        Returns:
            X: shape (n_samples, seq_length, n_features)
            y: shape (n_samples,)
            timestamps: 对应的时间戳
        """
        if feature_cols is None:
            feature_cols = self.selected_features or self.feature_names

        logger.info(f"创建序列: 序列长度={seq_length}, 特征数={len(feature_cols)}")

        data = df[feature_cols].values
        labels = df['label'].values
        timestamps = df['datetime'].values

        # 标准化
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = self.scaler.fit_transform(data)

        X, y, ts = [], [], []
        for i in range(seq_length, len(data)):
            X.append(data[i-seq_length:i])
            y.append(labels[i])
            ts.append(timestamps[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        logger.info(f"序列数据: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"标签分布: 涨={np.sum(y==1)} ({np.mean(y==1)*100:.1f}%), 跌={np.sum(y==0)} ({np.mean(y==0)*100:.1f}%)")

        return X, y, np.array(ts)


if __name__ == "__main__":
    from data.data_loader import DataLoader

    loader = DataLoader()
    df = loader.prepare_data()

    fe = FeatureEngineer()
    df = fe.build_features(df)
    print(f"\n特征数量: {len(fe.feature_names)}")
    print(f"特征列表: {fe.feature_names[:20]}...")

    selected = fe.select_features(df)
    print(f"\n选择的特征数量: {len(selected)}")

    X, y, ts = fe.create_sequences(df)
    print(f"\n序列数据: X={X.shape}, y={y.shape}")
