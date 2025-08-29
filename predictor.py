# -*- coding: utf-8 -*-
"""
Preditor principal.
- Busca OHLC no Yahoo Finance (yfinance)
- Calcula features simples
- Carrega scaler/model se existirem (joblib pkl)
- Gera sinal Buy/Sell com "confiança"
- Exponibiliza `run_once(...)` e o wrapper `predict_last(...)` (usado pelo web)
"""

from __future__ import annotations
import os, json, time, logging
from typing import List, Dict, Any
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

# Modelos (opcional)
try:
    import joblib
except Exception:
    joblib = None

# =========================
# Logging
# =========================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
log = logging.getLogger("predictor")

# =========================
# Utilidades
# =========================
def _env_list(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [s.strip() for s in raw.split(",") if s.strip()]

def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# =========================
# Coleta de dados
# =========================
def fetch_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """
    Baixa candles do Yahoo. Ex.: period='3mo', interval='60m'
    """
    log.debug(f"[fetch] {symbol} period={period} interval={interval}")
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,  # mais estável em ambientes server
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Sem dados para {symbol} ({period}/{interval})")
    df = df.rename(columns=str.lower).reset_index(drop=False)
    # normaliza nomes
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "Date"})
    if "date" in df.columns:
        df = df.rename(columns={"date": "Date"})
    return df

# =========================
# Features básicas
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features mínimas (robustas) – não assume o pipeline do seu pkl.
    Se existir scaler/model, usa; se não, fazemos heurística com essas features.
    """
    w_close = df["close"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["close"] = w_close
    feat["ret1"]  = w_close.pct_change(1).fillna(0.0)
    feat["ret3"]  = w_close.pct_change(3).fillna(0.0)
    feat["sma10"] = w_close.rolling(10).mean().fillna(method="bfill")
    feat["sma20"] = w_close.rolling(20).mean().fillna(method="bfill")
    feat["ema10"] = w_close.ewm(span=10, adjust=False).mean()
    # RSI simplificado
    delta = w_close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = (up / (down + 1e-9)).replace([np.inf, -np.inf], 0)
    feat["rsi14"] = 100 - (100 / (1 + rs))
    feat["rsi14"] = feat["rsi14"].fillna(50.0)

    # última linha como amostra
    return feat

# =========================
# Modelo (opcional)
# =========================
def load_pipeline():
    """
    Carrega scaler e modelo se existirem. Se não existirem, retorna (None, None).
    """
    model_path  = os.getenv("MODEL_PATH", "forex_model.pkl")
    scaler_path = os.getenv("SCALER_PATH", "forex_scaler.pkl")

    mdl = scl = None
    if joblib:
        try:
            if os.path.exists(scaler_path):
                scl = joblib.load(scaler_path)
                log.info(f"[model] scaler carregado: {scaler_path}")
            if os.path.exists(model_path):
                mdl = joblib.load(model_path)
                log.info(f"[model] modelo carregado: {model_path}")
        except Exception as e:
            log.warning(f"[model] falha ao carregar pkl: {e}")
    else:
        log.info("[model] joblib indisponível – seguindo sem modelo")

    return scl, mdl

SCALER, MODEL = load_pipeline()

# =========================
# Core de predição
# =========================
def _infer_signal_from_features(feat_row: pd.Series) -> Dict[str, Any]:
    """
    Heurística fallback quando não há modelo: cruzamento de médias + RSI.
    """
    sma10 = float(feat_row["sma10"])
    sma20 = float(feat_row["sma20"])
    rsi   = float(feat_row["rsi14"])

    if sma10 > sma20 and rsi < 70:
        return {"side": "BUY", "confidence": 0.60}
    elif sma10 < sma20 and rsi > 30:
        return {"side": "SELL", "confidence": 0.60}
    else:
        return {"side": "HOLD", "confidence": 0.40}

def predict_symbol(symbol: str, period: str, interval: str, conf_threshold: float) -> Dict[str, Any]:
    df = fetch_ohlc(symbol, period, interval)
    feat = build_features(df)

    X_last = feat.iloc[[-1]].copy()
    close_price = float(df["close"].iloc[-1])

    # Com modelo?
    side = "HOLD"
    prob_buy = prob_sell = 0.5
    confidence = 0.5

    try:
        Xp = X_last.values
        if SCALER is not None:
            Xp = SCALER.transform(Xp)
        if MODEL is not None:
            # Binário: 1=BUY, 0=SELL (convencional)
            if hasattr(MODEL, "predict_proba"):
                proba = MODEL.predict_proba(Xp)[0]
                # assume classe 1 = buy
                prob_buy = float(proba[1]) if len(proba) > 1 else float(proba[0])
                prob_sell = 1.0 - prob_buy
                side = "BUY" if prob_buy >= 0.5 else "SELL"
                confidence = max(prob_buy, prob_sell)
            else:
                pred = MODEL.predict(Xp)[0]
                side = "BUY" if int(pred) == 1 else "SELL"
                confidence = 0.60
        else:
            s = _infer_signal_from_features(X_last.iloc[0])
            side, confidence = s["side"], s["confidence"]
            prob_buy = confidence if side == "BUY" else 1.0 - confidence
            prob_sell = 1.0 - prob_buy
    except Exception as e:
        log.warning(f"[predict] fallback heurístico [{symbol}] por erro: {e}")
        s = _infer_signal_from_features(X_last.iloc[0])
        side, confidence = s["side"], s["confidence"]
        prob_buy = confidence if side == "BUY" else 1.0 - confidence
        prob_sell = 1.0 - prob_buy

    # monta alvo simples (TP/SL) como exemplo
    atr = float(df["high"].tail(14).max() - df["low"].tail(14).min()) / 14.0
    tp = close_price + (+1 if side == "BUY" else -1) * max(atr, close_price * 0.001)
    sl = close_price - (+1 if side == "BUY" else -1) * max(atr, close_price * 0.001)

    signal_ok = side in ("BUY", "SELL") and confidence >= conf_threshold

    return {
        "symbol": symbol,
        "side": side,
        "price": round(close_price, 6),
        "take_profit": round(tp, 6),
        "stop_loss": round(sl, 6),
        "atr": round(atr, 6),
        "prob_buy": round(prob_buy * 100, 2),
        "prob_sell": round(prob_sell * 100, 2),
        "confidence": round(confidence * 100, 2),
        "ok": signal_ok,
        "time": _utc_now_str(),
        "model": "XGBoost Forex AI" if MODEL is not None else "Heuristic",
    }

# =========================
# Execução 1 ciclo
# =========================
def run_once(
    symbols: List[str] | None = None,
    period: str | None = None,
    interval: str | None = None,
    conf_threshold: float = 0.60,
    send_telegram: bool = False,
) -> Dict[str, Any]:
    from notifier_telegram import send_message  # import leve

    symbols = symbols or _env_list("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X,BTC-USD")
    period = period or os.getenv("PERIOD", "3mo")
    interval = interval or os.getenv("INTERVAL", "60m")
    conf_threshold = float(conf_threshold or os.getenv("CONF_THRESHOLD", "0.60"))

    log.info(f"[run_once] symbols={symbols} period={period} interval={interval} thr={conf_threshold}")

    signals = []
    updated = 0
    for sym in symbols:
        try:
            s = predict_symbol(sym, period, interval, conf_threshold)
            signals.append(s)
            if s["ok"]:
                updated += 1
                if send_telegram and os.getenv("TELEGRAM_ENABLED", "true").lower() == "true":
                    text = (
                        f"💱 **FOREX AI SIGNAL**\n"
                        f"**{sym}** → **{s['side']}**\n"
                        f"Preço: {s['price']}\n"
                        f"TP: {s['take_profit']} | SL: {s['stop_loss']}\n"
                        f"Confiança: {s['confidence']}%\n"
                        f"({s['model']}) • {s['time']}"
                    )
                    send_message(text)
        except Exception as e:
            log.error(f"[run_once] erro em {sym}: {e}", exc_info=LOG_LEVEL == "DEBUG")

    return {"success": True, "signals": signals, "updated": updated}

# =========================
# Wrapper para o web
# =========================
def predict_last(
    symbols: List[str] | None = None,
    period: str | None = None,
    interval: str | None = None,
    conf_threshold: float = 0.60,
) -> Dict[str, Any]:
    """
    Usado pelo serviço web (API). Não envia Telegram aqui.
    """
    try:
        return run_once(
            symbols=symbols,
            period=period,
            interval=interval,
            conf_threshold=conf_threshold,
            send_telegram=False,
        )
    except Exception as e:
        log.error(f"[predict_last] {e}", exc_info=LOG_LEVEL == "DEBUG")
        return {"success": False, "error": str(e), "signals": [], "updated": 0}
