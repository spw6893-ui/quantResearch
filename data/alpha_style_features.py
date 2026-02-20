import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlphaStyleConfig:
    prefixes: tuple[str, ...] = ("oi_", "funding_", "liq_", "at_", "cs_", "ls_")
    cols: tuple[str, ...] = ()
    regex: str | None = None
    max_cols: int = 20
    windows: tuple[int, ...] = (2, 3, 5, 8, 10)
    ops: tuple[str, ...] = ("delta", "pct", "z", "ema", "corr_ret")


def _sanitize_name(name: str) -> str:
    # 特征列名里可能含有特殊字符（例如 ':'），统一替换成 '_'，便于下游选择/保存
    return re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_")


def _select_numeric_columns(
    df: pd.DataFrame,
    cfg: AlphaStyleConfig,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    exclude = set(exclude or set())

    cand: list[str] = []
    if cfg.cols:
        for c in cfg.cols:
            if c in df.columns:
                cand.append(c)
    else:
        prefixes = tuple(p for p in cfg.prefixes if p)
        if prefixes:
            cand.extend([c for c in df.columns if any(str(c).startswith(p) for p in prefixes)])
        if cfg.regex:
            try:
                rr = re.compile(cfg.regex)
            except Exception as e:
                raise ValueError(f"alpha-style regex 编译失败：{cfg.regex}，错误：{e}") from e
            cand.extend([c for c in df.columns if rr.search(str(c))])

    # 去重 + 排除
    seen = set()
    out: list[str] = []
    for c in cand:
        if c in seen or c in exclude:
            continue
        seen.add(c)
        # 只保留数值列（bool 也算数值）
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            out.append(c)

    if not out:
        return []

    # 按方差排序（NaN 跳过），挑前 max_cols，避免一次加太多列导致内存爆
    if int(cfg.max_cols) > 0 and len(out) > int(cfg.max_cols):
        v = []
        for c in out:
            s = pd.to_numeric(df[c], errors="coerce")
            vv = float(s.var(skipna=True)) if s.notna().any() else -1.0
            v.append((vv, c))
        v.sort(reverse=True)
        out = [c for _, c in v[: int(cfg.max_cols)]]
    return out


def compute_alpha_style_features(
    df: pd.DataFrame,
    cfg: AlphaStyleConfig,
    *,
    prefix: str = "as_",
) -> pd.DataFrame:
    """对指定高频列批量生成短窗“alpha-style”连续特征（rolling/ewm/corr）。

    设计原则：
    - 只用历史/当前信息：rolling/ewm 默认不看未来，避免前视泄露
    - 控制规模：默认只选 top variance 的 max_cols 列，避免特征爆炸/内存爆
    - 产生 NaN 是正常的（窗口期不足），下游可用 XGBoost missing 或缩放前填充处理
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(index=getattr(df, "index", None))

    base_exclude = {
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "label",
        "future_return",
        "tb_raw",
    }
    cols = _select_numeric_columns(df, cfg, exclude=base_exclude)
    if not cols:
        return pd.DataFrame(index=df.index)

    windows = [int(w) for w in cfg.windows if int(w) > 1]
    if not windows:
        return pd.DataFrame(index=df.index)

    ops = [str(op).strip().lower() for op in cfg.ops if str(op).strip()]
    if not ops:
        return pd.DataFrame(index=df.index)

    out = pd.DataFrame(index=df.index)

    # 用 close 的 1bar return 做相关系数基准（如果没有 close，则跳过 corr_ret）
    ret1 = None
    if "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        ret1 = close.pct_change()

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        c_name = _sanitize_name(c)

        for w in windows:
            if "delta" in ops:
                out[f"{prefix}{c_name}_delta_{w}"] = x.diff(w)

            if "pct" in ops:
                x_prev = x.shift(w)
                # 避免除 0 导致 inf
                out[f"{prefix}{c_name}_pct_{w}"] = np.where(
                    x_prev.abs() > 0,
                    (x / x_prev) - 1.0,
                    np.nan,
                )

            if "ema" in ops:
                out[f"{prefix}{c_name}_ema_{w}"] = x.ewm(span=w, adjust=False, min_periods=w).mean()

            if "z" in ops:
                m = x.rolling(w, min_periods=w).mean()
                sd = x.rolling(w, min_periods=w).std(ddof=0)
                out[f"{prefix}{c_name}_z_{w}"] = np.where(sd > 0, (x - m) / sd, np.nan)

            if "corr_ret" in ops and ret1 is not None:
                out[f"{prefix}{c_name}_corr_ret_{w}"] = x.rolling(w, min_periods=w).corr(ret1)

    return out


def add_alpha_style_features(
    df: pd.DataFrame,
    cfg: AlphaStyleConfig,
    *,
    prefix: str = "as_",
) -> pd.DataFrame:
    feats = compute_alpha_style_features(df, cfg, prefix=prefix)
    if feats is None or feats.empty:
        return df
    return pd.concat([df, feats], axis=1)

