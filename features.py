# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Médias
    d["ema_10"] = d["close"].ewm(span=10, adjust=False).mean()
    d["ema_20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["sma_50"] = d["close"].rolling(50).mean()
    # RSI
    delta = d["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    rs = up / (down + 1e-9)
    d["rsi_14"] = 100 - (100 / (1 + rs))
    # ATR (14)
    tr = np.maximum(d["high"]-d["low"], np.maximum((d["high"]-d["close"].shift()).abs(),
                                                   (d["low"]-d["close"].shift()).abs()))
    d["atr_14"] = tr.rolling(14).mean()
    # Volatilidade simples
    d["ret"] = d["close"].pct_change()
    d["vol_20"] = d["ret"].rolling(20).std()
    # Limpa NaN
    d = d.dropna().reset_index(drop=True)
    return d

FEATURE_COLS = ["ema_10","ema_20","sma_50","rsi_14","atr_14","vol_20"]
