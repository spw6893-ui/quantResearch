"""
Alpha101（WorldQuant 101 Alphas）因子构造（单品种/时间序列版）

重要说明：
- 原始 Alpha101 设计为“横截面因子”（同一时刻对一篮子股票做 rank/scale 等操作）。
- 本项目当前主要是单品种（BTC）时间序列建模，因此这里做了“时间序列近似”：
  - 公式里的 rank(x) / scale(x) 等横截面算子，在单品种场景下会退化。
  - 我们用“滚动窗口 min-max 归一化”近似 rank（0~1），用滚动 L1 归一化近似 scale。
- 这不是严格意义的 Alpha101，但可以作为一组结构化的价格/成交量衍生特征，用于快速试验。

目前实现：Alpha 1、3~20（不含 #2；后续可继续补齐）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def _to_s(x: pd.Series | np.ndarray) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def _delay(s: pd.Series, n: int) -> pd.Series:
    return s.shift(int(n))


def _delta(s: pd.Series, n: int) -> pd.Series:
    return s.diff(int(n))


def _ts_sum(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=max(int(n) // 2, 2)).sum()


def _ts_mean(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=max(int(n) // 2, 2)).mean()


def _ts_std(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=max(int(n) // 2, 2)).std()


def _ts_min(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=max(int(n) // 2, 2)).min()


def _ts_max(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=max(int(n) // 2, 2)).max()


def _ts_corr(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(int(n), min_periods=max(int(n) // 2, 2)).corr(b)


def _ts_cov(a: pd.Series, b: pd.Series, n: int) -> pd.Series:
    return a.rolling(int(n), min_periods=max(int(n) // 2, 2)).cov(b)


def _signed_power(x: pd.Series, a: float) -> pd.Series:
    aa = float(a)
    return np.sign(x) * (np.abs(x) ** aa)


def _ts_argmax(s: pd.Series, n: int) -> pd.Series:
    w = int(n)
    if w <= 0:
        raise ValueError("window 必须 > 0")

    def _fn(x: np.ndarray) -> float:
        if not np.isfinite(x).any():
            return np.nan
        # 返回 1..w（与常见 alpha101 实现一致）
        return float(np.nanargmax(x) + 1)

    return s.rolling(w, min_periods=max(w // 2, 2)).apply(_fn, raw=True)


def _rank_approx_minmax(s: pd.Series, n: int) -> pd.Series:
    """用滚动窗口 min-max 归一化近似 rank（0~1）。"""
    w = int(n)
    mn = s.rolling(w, min_periods=max(w // 2, 2)).min()
    mx = s.rolling(w, min_periods=max(w // 2, 2)).max()
    return (s - mn) / (mx - mn + EPS)


def _scale_approx_l1(s: pd.Series, n: int) -> pd.Series:
    """用滚动 L1 归一化近似 scale。"""
    w = int(n)
    denom = s.abs().rolling(w, min_periods=max(w // 2, 2)).sum()
    return s / (denom + EPS)


def _decay_linear(s: pd.Series, n: int) -> pd.Series:
    """线性衰减加权平均（窗口 n，权重 1..n）。"""
    w = int(n)
    weights = np.arange(1, w + 1, dtype=np.float64)
    wsum = float(weights.sum())

    def _fn(x: np.ndarray) -> float:
        if len(x) != w:
            return np.nan
        xx = np.nan_to_num(x.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return float(np.dot(xx, weights) / wsum)

    return s.rolling(w, min_periods=w).apply(_fn, raw=True)


@dataclass(frozen=True)
class Alpha101Config:
    rank_window: int = 20
    adv_window: int = 20
    scale_window: int = 20
    window_scale: float = 1.0

    def w(self, n: int) -> int:
        """按 window_scale 缩放窗口长度，并做最小值保护。"""
        nn = int(n)
        s = float(self.window_scale)
        if not np.isfinite(s) or s <= 0:
            raise ValueError("window_scale 需为正数")
        return max(2, int(round(nn * s)))


def compute_alpha101_single(
    df: pd.DataFrame,
    alphas: Iterable[int] = (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
    *,
    cfg: Alpha101Config | None = None,
    prefix: str = "alpha101_",
) -> pd.DataFrame:
    """计算 Alpha101（单品种近似版）并返回 DataFrame（与 df 行对齐）。"""
    cfg = cfg or Alpha101Config()
    alphas = [int(a) for a in alphas]
    if not alphas:
        return pd.DataFrame(index=df.index)

    need_cols = ["open", "high", "low", "close", "volume"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"alpha101 需要列: {need_cols}，缺失: {miss}")

    open_ = pd.to_numeric(df["open"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")

    if "vwap" in df.columns:
        vwap = pd.to_numeric(df["vwap"], errors="coerce")
    else:
        vwap = (high + low + close) / 3.0

    returns = close.pct_change()
    adv = _ts_mean(volume, int(cfg.adv_window))

    # “rank/scale”横截面算子的时间序列近似
    rk = lambda s: _rank_approx_minmax(_to_s(s), int(cfg.rank_window))
    sc = lambda s: _scale_approx_l1(_to_s(s), int(cfg.scale_window))

    out = pd.DataFrame(index=df.index)

    # Alpha 1
    if 1 in alphas:
        w20 = cfg.w(20)
        w5 = cfg.w(5)
        x = pd.Series(np.where(returns < 0, _ts_std(returns, w20), close), index=df.index)
        a1 = rk(_ts_argmax(_signed_power(x, 2.0), w5)) - 0.5
        out[f"{prefix}001"] = a1

    # Alpha 3
    if 3 in alphas:
        out[f"{prefix}003"] = -_ts_corr(rk(open_), rk(volume), cfg.w(10))

    # Alpha 4
    if 4 in alphas:
        out[f"{prefix}004"] = -_rank_approx_minmax(rk(low), cfg.w(9))

    # Alpha 5
    if 5 in alphas:
        w10 = cfg.w(10)
        a = open_ - (_ts_sum(vwap, w10) / float(w10))
        b = close - vwap
        out[f"{prefix}005"] = rk(a) * (-np.abs(rk(b)))

    # Alpha 6
    if 6 in alphas:
        out[f"{prefix}006"] = -_ts_corr(open_, volume, cfg.w(10))

    # Alpha 7
    if 7 in alphas:
        w7 = cfg.w(7)
        w60 = cfg.w(60)
        d7 = _delta(close, w7)
        body = (-_rank_approx_minmax(np.abs(d7), w60)) * np.sign(d7)
        out[f"{prefix}007"] = np.where(adv < volume, body, -1.0)

    # Alpha 8
    if 8 in alphas:
        w5 = cfg.w(5)
        w10 = cfg.w(10)
        x = _ts_sum(open_, w5) * _ts_sum(returns, w5)
        out[f"{prefix}008"] = -rk(x - _delay(x, w10))

    # Alpha 9
    if 9 in alphas:
        w1 = cfg.w(1)
        w5 = cfg.w(5)
        d1 = _delta(close, w1)
        m1 = _ts_min(d1, w5)
        m2 = _ts_max(d1, w5)
        a9 = np.where(0 < m1, d1, np.where(m2 < 0, d1, -d1))
        out[f"{prefix}009"] = a9

    # Alpha 10
    if 10 in alphas:
        w1 = cfg.w(1)
        w4 = cfg.w(4)
        d1 = _delta(close, w1)
        m1 = _ts_min(d1, w4)
        m2 = _ts_max(d1, w4)
        a10 = np.where(0 < m1, d1, np.where(m2 < 0, d1, -d1))
        out[f"{prefix}010"] = rk(pd.Series(a10, index=df.index))

    # Alpha 11
    if 11 in alphas:
        x = vwap - close
        w3 = cfg.w(3)
        out[f"{prefix}011"] = (rk(_ts_max(x, w3)) + rk(_ts_min(x, w3))) * rk(_delta(volume, w3))

    # Alpha 12
    if 12 in alphas:
        w1 = cfg.w(1)
        out[f"{prefix}012"] = np.sign(_delta(volume, w1)) * (-_delta(close, w1))

    # Alpha 13
    if 13 in alphas:
        out[f"{prefix}013"] = -rk(_ts_cov(rk(close), rk(volume), cfg.w(5)))

    # Alpha 14
    if 14 in alphas:
        out[f"{prefix}014"] = (-rk(_delta(returns, cfg.w(3)))) * _ts_corr(open_, volume, cfg.w(10))

    # Alpha 15
    if 15 in alphas:
        w3 = cfg.w(3)
        out[f"{prefix}015"] = -_ts_sum(rk(_ts_corr(rk(high), rk(volume), w3)), w3)

    # Alpha 16
    if 16 in alphas:
        out[f"{prefix}016"] = -rk(_ts_cov(rk(high), rk(volume), cfg.w(5)))

    # Alpha 17
    if 17 in alphas:
        w10 = cfg.w(10)
        w1 = cfg.w(1)
        w5 = cfg.w(5)
        out[f"{prefix}017"] = (-rk(_rank_approx_minmax(close, w10))) * rk(_delta(_delta(close, w1), w1)) * rk(_rank_approx_minmax(volume / (adv + EPS), w5))

    # Alpha 18
    if 18 in alphas:
        x = (close - open_).abs()
        out[f"{prefix}018"] = -rk((_ts_std(x, cfg.w(5)) + (close - open_)) + _ts_corr(close, open_, cfg.w(10)))

    # Alpha 19
    if 19 in alphas:
        w7 = cfg.w(7)
        w250 = cfg.w(250)
        out[f"{prefix}019"] = (-np.sign((close - _delay(close, w7)) + _delta(close, w7))) * (1.0 + rk(1.0 + _ts_sum(returns, w250)))

    # Alpha 20
    if 20 in alphas:
        w1 = cfg.w(1)
        out[f"{prefix}020"] = (-rk(open_ - _delay(high, w1))) * rk(open_ - _delay(close, w1)) * rk(open_ - _delay(low, w1))

    # 尽量用 float32，节省内存
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(np.float32)

    return out
