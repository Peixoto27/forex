# -*- coding: utf-8 -*-
"""
Preditor com envio de alertas (1 mensagem por par).
- Coleta via yfinance (PERIOD, INTERVAL)
- Calcula features (SMA/RSI) + ATR
- IA: usa modelo/scaler se existir; senão, heurística robusta
- Envia Telegram por par com ATR, prob. buy/sell, confiança e hora UTC
- Filtro por confiança (CONF_THRESHOLD = 0.65 ou 65)
- Anti-duplicados por símbolo (muda side/TP/SL ou cooldown)
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import joblib
except Exception:
    joblib = None

from notifier_telegram import send_message

# ---------------- Config ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("predictor")

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X,BTC-USD,ETH-USD").split(",") if s.strip()]
PERIOD = os.getenv("PERIOD", "3mo").strip()      # ex.: 3mo
INTERVAL = os.getenv("INTERVAL", "60m").strip()  # ex.: 60m

_thr = os.getenv("CONF_THRESHOLD", "0.65")
try:
    CONF_THRESHOLD = float(_thr)
    if CONF_THRESHOLD > 1.0:
        CONF_THRESHOLD /= 100.0
except Exception:
    CONF_THRESHOLD = 0.65

COOLDOWN_MIN = int(os.getenv("ALERT_COOLDOWN_MIN", "30"))  # anti-duplicados
LAST_SENT_FILE = os.getenv("LAST_SENT_FILE", "last_sent.json")

MODEL_PATH  = os.getenv("MODEL_PATH", "forex_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "forex_scaler.pkl")

# -------------- Utils --------------
def _utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Falha ao salvar {path}: {e}")

def _fmt(x, nd=5) -> str:
    try:
        return f"{float(x):.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

# -------------- Data --------------
def fetch_ohlc(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"Sem dados para {symbol} ({period}/{interval})")
    df = df.rename(columns=str.lower).reset_index(drop=False)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].astype(float)

    feat = pd.DataFrame(index=df.index)
    feat["close"] = close
    feat["ret1"]  = close.pct_change(1).fillna(0.0)
    feat["sma10"] = close.rolling(10).mean().bfill()
    feat["sma20"] = close.rolling(20).mean().bfill()

    # RSI(14)
    delta = close.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = (up / (down + 1e-9)).replace([np.inf, -np.inf], 0)
    feat["rsi14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    return feat

def calc_atr(df: pd.DataFrame, n: int = 14) -> float:
    # ATR simples baseado em High/Low (sem True Range completo por simplicidade/robustez)
    try:
        atr = (df["high"].tail(n).max() - df["low"].tail(n).min()) / float(n)
        return float(atr)
    except Exception:
        return 0.0

# -------------- Model --------------
def load_pipeline() -> Tuple[Any, Any]:
    scl = mdl = None
    if joblib:
        try:
            if os.path.exists(SCALER_PATH):
                scl = joblib.load(SCALER_PATH)
                log.info(f"[model] scaler carregado: {SCALER_PATH}")
            if os.path.exists(MODEL_PATH):
                mdl = joblib.load(MODEL_PATH)
                log.info(f"[model] modelo carregado: {MODEL_PATH}")
        except Exception as e:
            log.warning(f"[model] falha ao carregar: {e}")
    else:
        log.info("[model] joblib indisponível, usando heurística")
    return scl, mdl

SCALER, MODEL = load_pipeline()

def infer_prob_signal(x_row: np.ndarray) -> Tuple[str, float, float]:
    """
    Retorna (side, prob_buy, prob_sell) com modelo se existir, senão heurística.
    """
    if SCALER is not None:
        try:
            x_row = SCALER.transform(x_row)
        except Exception as e:
            log.warning(f"[scale] falha: {e}")

    if MODEL is not None:
        try:
            if hasattr(MODEL, "predict_proba"):
                proba = MODEL.predict_proba(x_row)[0]
                if len(proba) == 1:
                    pb = float(proba[0])
                    ps = 1 - pb
                else:
                    pb = float(proba[1])
                    ps = float(proba[0])
                side = "BUY" if pb >= 0.5 else "SELL"
                return side, pb, ps
            else:
                pred = int(MODEL.predict(x_row)[0])
                side = "BUY" if pred == 1 else "SELL"
                pb = 0.65 if side == "BUY" else 0.35
                ps = 1 - pb
                return side, pb, ps
        except Exception as e:
            log.warning(f"[model] falha na inferência: {e}")

    # heurística fallback (SMA cruzada + RSI zona)
    # x_row segue a ordem de build_features: [close, ret1, sma10, sma20, rsi14]
    try:
        close, ret1, sma10, sma20, rsi14 = map(float, x_row[0])
        if sma10 > sma20 and rsi14 < 70:
            return "BUY", 0.75, 0.25
        elif sma10 < sma20 and rsi14 > 30:
            return "SELL", 0.25, 0.75
        else:
            return "HOLD", 0.5, 0.5
    except Exception:
        return "HOLD", 0.5, 0.5

# -------------- Core --------------
def predict_symbol(symbol: str) -> Dict[str, Any]:
    df = fetch_ohlc(symbol, PERIOD, INTERVAL)
    feat = build_features(df)

    x_last = feat.iloc[[-1]].copy()
    x_values = x_last.values

    side, pb, ps = infer_prob_signal(x_values)

    price = float(df["close"].iloc[-1])
    atr = calc_atr(df, 14)

    # alvo simples proporcional ao ATR (garante deslocamento mínimo)
    step = max(atr, max(1e-6, price * 0.001))
    if side == "BUY":
        tp = price + step
        sl = price - step
    elif side == "SELL":
        tp = price - step
        sl = price + step
    else:
        tp = sl = price

    conf = max(pb, ps)  # 0..1
    ok = side in ("BUY", "SELL") and conf >= CONF_THRESHOLD

    return {
        "symbol": symbol,
        "side": side,
        "price": price,
        "take_profit": tp,
        "stop_loss": sl,
        "atr": atr,
        "prob_buy": round(pb * 100, 2),
        "prob_sell": round(ps * 100, 2),
        "confidence": round(conf * 100, 2),
        "ok": ok,
        "time": _utc_now_str(),
        "model": "AI" if MODEL is not None else "Heuristic",
    }

def format_msg(sig: Dict[str, Any]) -> str:
    return (
        "💱 **FOREX AI SIGNAL**\n"
        f"**{sig['symbol']}** → **{sig['side']}**\n"
        f"Preço: `{_fmt(sig['price'])}`\n"
        f"TP: `{_fmt(sig['take_profit'])}`  |  SL: `{_fmt(sig['stop_loss'])}`\n"
        f"ATR: `{_fmt(sig['atr'])}`\n"
        f"Prob. Compra: **{_fmt(sig['prob_buy'])}%**  |  Prob. Venda: **{_fmt(sig['prob_sell'])}%**\n"
        f"Confiança: **{_fmt(sig['confidence'])}%**\n"
        f"🕒 {sig['time']}"
    )

# -------------- Anti-duplicados --------------
def _key(sig: Dict[str, Any]) -> str:
    # define “unicidade” por combo side/TP/SL arredondados
    return f"{sig['symbol']}|{sig['side']}|{_fmt(sig['take_profit'])}|{_fmt(sig['stop_loss'])}"

def _can_send(sig: Dict[str, Any], cache: dict) -> bool:
    from time import time
    k = _key(sig)
    rec = cache.get(k)
    if rec is None:
        return True
    # cooldown
    last_ts = rec.get("ts", 0)
    return (time() - last_ts) >= (COOLDOWN_MIN * 60)

def _mark_sent(sig: Dict[str, Any], cache: dict) -> None:
    from time import time
    k = _key(sig)
    cache[k] = {"ts": time()}

# -------------- API principal --------------
def predict_last(symbols: List[str] = None,
                 period: str = None,
                 interval: str = None,
                 conf_threshold: float = None) -> Dict[str, Any]:
    """
    Compatível com o web: retorna {"success", "signals", "updated"}
    e envia 1 mensagem por par (se ok=True e passar no anti-duplicados).
    """
    global PERIOD, INTERVAL, CONF_THRESHOLD
    if symbols is None:
        symbols = SYMBOLS
    if period:
        PERIOD = period
    if interval:
        INTERVAL = interval
    if conf_threshold is not None:
        # aceita 0.65 ou 65
        CONF_THRESHOLD = float(conf_threshold)
        if CONF_THRESHOLD > 1.0:
            CONF_THRESHOLD /= 100.0

    cache = _load_json(LAST_SENT_FILE)

    signals: List[Dict[str, Any]] = []
    updated = 0
    for sym in symbols:
        try:
            sig = predict_symbol(sym)
            signals.append(sig)
            if sig["ok"] and _can_send(sig, cache):
                msg = format_msg(sig)
                ok, info = send_message(msg)
                if ok:
                    _mark_sent(sig, cache)
                    updated += 1
                    log.info(f"[telegram] enviado {sym}")
                else:
                    log.error(f"[telegram] erro {sym}: {info}")
        except Exception as e:
            log.error(f"[predict_last] erro com {sym}: {e}")

    _save_json(LAST_SENT_FILE, cache)

    return {"success": True, "signals": signals, "updated": updated}
