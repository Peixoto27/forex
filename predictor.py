# -*- coding: utf-8 -*-
import os, json
import numpy as np
import pandas as pd
from typing import List, Dict

from features import FEATURE_COLS

# Tenta carregar modelo e scaler reais
_model, _scaler = None, None
def _try_load_model():
    global _model, _scaler
    if _model is not None: 
        return
    try:
        import joblib
        if os.path.exists("forex_scaler.pkl"):
            _scaler = joblib.load("forex_scaler.pkl")
        if os.path.exists("forex_model.pkl"):
            _model = joblib.load("forex_model.pkl")
    except Exception:
        _model, _scaler = None, None

def _fallback_signal(row) -> int:
    """Heurística simples: cruzamentos de EMA + RSI"""
    buy = (row["ema_10"] > row["ema_20"]) and (row["rsi_14"] < 60)
    sell = (row["ema_10"] < row["ema_20"]) and (row["rsi_14"] > 40)
    return 1 if buy else (0 if sell else 1)  # default BUY

def predict_last(df_feat: pd.DataFrame, symbol: str) -> Dict:
    """
    Retorna um dicionário de previsão para a última barra.
    signal: BUY/SELL, confidence: 0..1
    """
    _try_load_model()
    row = df_feat.iloc[-1]
    X = row[FEATURE_COLS].values.reshape(1, -1).astype(float)
    price = float(row["close"])

    signal = "BUY"
    conf = 0.55

    if _scaler is not None:
        try: 
            X = _scaler.transform(X)
        except Exception: 
            pass

    if _model is not None:
        try:
            # Se for classificador binário 0/1
            prob = getattr(_model, "predict_proba", None)
            pred = getattr(_model, "predict", None)
            if prob:
                p = prob(X)[0]
                y = int(np.argmax(p))
                conf = float(max(p))
            elif pred:
                y = int(pred(X)[0])
                conf = 0.60
            else:
                y = _fallback_signal(row)
            signal = "BUY" if y == 1 else "SELL"
        except Exception:
            y = _fallback_signal(row)
            signal = "BUY" if y == 1 else "SELL"
            conf = 0.58
    else:
        # Fallback heurístico
        y = _fallback_signal(row)
        signal = "BUY" if y == 1 else "SELL"
        conf = 0.57

    # Risco/Retorno estimado simples via ATR
    atr = float(row.get("atr_14", 0) or 0)
    rr = 2.0
    tp = price + (atr * rr) if signal == "BUY" else price - (atr * rr)
    sl = price - (atr * 1.2) if signal == "BUY" else price + (atr * 1.2)

    return {
        "symbol": symbol,
        "signal": signal,
        "price": round(price, 6),
        "confidence": round(conf, 4),
        "rr": rr,
        "tp": round(float(tp), 6),
        "sl": round(float(sl), 6),
    }
