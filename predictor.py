# -*- coding: utf-8 -*-
"""
Preditor simplificado.
- Usa yfinance p/ coletar dados
- Calcula features básicas
- Gera sinal Buy/Sell com confiança
- Notifica Telegram se houver sinal válido
"""

import os, logging
from datetime import datetime, timezone
from typing import List, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf
from notifier_telegram import send_message

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("predictor")

def _utc_now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def fetch_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol, period=period, interval=interval,
        auto_adjust=True, progress=False, threads=False
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Sem dados para {symbol}")
    df = df.rename(columns=str.lower).reset_index(drop=False)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    w_close = df["close"].astype(float)
    feat = pd.DataFrame(index=df.index)
    feat["close"] = w_close
    feat["sma10"] = w_close.rolling(10).mean().fillna(method="bfill")
    feat["sma20"] = w_close.rolling(20).mean().fillna(method="bfill")
    delta = w_close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = (up / (down + 1e-9)).replace([np.inf, -np.inf], 0)
    feat["rsi14"] = 100 - (100 / (1 + rs))
    feat["rsi14"] = feat["rsi14"].fillna(50.0)
    return feat

def predict_symbol(symbol: str, period: str, interval: str, conf_thr: float) -> Dict[str, Any]:
    df = fetch_ohlc(symbol, period, interval)
    feat = build_features(df)
    last = feat.iloc[-1]
    close_price = float(df["close"].iloc[-1])

    # regra simples
    side = "HOLD"; confidence = 50.0
    if last["sma10"] > last["sma20"] and last["rsi14"] < 70:
        side, confidence = "BUY", 95.0
    elif last["sma10"] < last["sma20"] and last["rsi14"] > 30:
        side, confidence = "SELL", 95.0

    # alvo simples
    atr = float(df["high"].tail(14).max() - df["low"].tail(14).min()) / 14.0
    tp = close_price + (+1 if side == "BUY" else -1) * max(atr, close_price*0.001)
    sl = close_price - (+1 if side == "BUY" else -1) * max(atr, close_price*0.001)

    return {
        "symbol": symbol,
        "side": side,
        "price": round(close_price, 5),
        "take_profit": round(tp, 5),
        "stop_loss": round(sl, 5),
        "atr": round(atr, 6),
        "confidence": confidence,
        "time": _utc_now_str(),
        "ok": side != "HOLD" and confidence >= (conf_thr*100)
    }

def predict_last(symbols: List[str], period: str, interval: str, conf_threshold: float) -> Dict[str, Any]:
    signals = []
    updated = 0
    for sym in symbols:
        try:
            s = predict_symbol(sym, period, interval, conf_threshold)
            signals.append(s)
            if s["ok"]:
                updated += 1
                msg = (
                    f"💱 FOREX AI SIGNAL\n"
                    f"{s['symbol']} → {s['side']}\n"
                    f"Entrada: {s['price']}\n"
                    f"TP: {s['take_profit']} | SL: {s['stop_loss']}\n"
                    f"Confiança: {s['confidence']}%\n"
                    f"{s['time']}"
                )
                send_message(msg)
        except Exception as e:
            log.error(f"Erro {sym}: {e}")
    return {"success": True, "signals": signals, "updated": updated}
