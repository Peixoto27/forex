# -*- coding: utf-8 -*-
"""
Runner: coleta OHLC, extrai features, roda predição e (opcional) notifica no Telegram.
- Lê env vars: SYMBOLS, PERIOD (ex.: 3mo), INTERVAL (ex.: 60m), CONF_THRESHOLD, PRED_FILE, LOG_LEVEL
- Logs detalhados por símbolo, com contadores ao final.
"""

import os
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any

from data_api import fetch_ohlcv
from features import add_indicators
from predictor import predict_last  # compat layer no predictor.py
try:
    from notifier_telegram import send_prediction_alerts
except Exception:
    send_prediction_alerts = None

# ----------------------
# Configuração de logging
# ----------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("runner")

# ----------------------
# ENV VARS
# ----------------------
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X,BTC-USD").split(",") if s.strip()]
INTERVAL = os.getenv("INTERVAL", "60m").strip()   # ex.: 60m
PERIOD   = os.getenv("PERIOD", "3mo").strip()     # ex.: 3mo
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))
PRED_FILE = os.getenv("PRED_FILE", "forex_predictions.json")

USE_TELEGRAM = bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))

def write_predictions(preds: List[Dict[str, Any]]):
    """Grava apenas a LISTA (compatível com web que já aceita lista ou wrapper)."""
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

def _normalize_for_telegram(preds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    O notifier espera campo 'confidence'. Nosso predictor retorna 'score'.
    Fazemos um mapeamento leve para não quebrar.
    """
    out = []
    for p in preds:
        q = dict(p)
        if "confidence" not in q and "score" in q:
            q["confidence"] = float(q["score"])
        out.append(q)
    return out

def run_once() -> Dict[str, Any]:
    log.info("==== runner start ====")
    log.info("PERIOD=%s | INTERVAL=%s | THR=%.2f | SYMBOLS=%s", PERIOD, INTERVAL, CONF_THRESHOLD, SYMBOLS)

    preds: List[Dict[str, Any]] = []
    updated_count = 0
    sent_count = 0

    for idx, sym in enumerate(SYMBOLS, start=1):
        try:
            log.info("[%d/%d] Coletando %s ...", idx, len(SYMBOLS), sym)
            df = fetch_ohlcv(sym, period=PERIOD, interval=INTERVAL)
            if df is None or df.empty:
                log.warning("Sem dados para %s (df vazio).", sym)
                continue

            rows = len(df)
            log.debug("%s: linhas=%d (period=%s, interval=%s)", sym, rows, PERIOD, INTERVAL)
            if rows < 60:
                log.warning("%s: histórico insuficiente (%d linhas). Pulando.", sym, rows)
                continue

            df_feat = add_indicators(df)
            if df_feat is None or df_feat.empty:
                log.warning("%s: features vazias após add_indicators.", sym)
                continue

            pred = predict_last(df_feat)
            if not pred:
                log.warning("%s: predição retornou None.", sym)
                continue

            # garantir campos esperados (signal, score/price)
            sym_out = pred.get("symbol") or sym
            score = float(pred.get("score", 0.0))
            signal = pred.get("signal", "HOLD")
            price = pred.get("price")
            from telegram_utils import send_telegram_alert

# ...
score = float(pred.get("score", 0.0))
signal = pred.get("signal", "HOLD")
price = pred.get("price")

# critério de envio
if signal != "HOLD" and score >= float(os.getenv("CONF_THRESHOLD", 60)):
    send_telegram_alert({
        "symbol": sym_out,
        "side": signal,
        "price": price,
        "confidence": score,
        "take_profit": pred.get("take_profit"),
        "stop_loss": pred.get("stop_loss"),
        "time": pred.get("time"),
    })
    
            log.info("%s: price=%s score=%.3f thr=%.2f -> %s", sym_out, price, score, CONF_THRESHOLD, signal)

            # guarda para JSON final
            preds.append({
                "symbol": sym_out,
                "signal": signal,
                "price": price,
                "confidence": score,  # compat com notifier/consumidores
                "rr": pred.get("rr", 2.0),
                "tp": pred.get("tp"),
                "sl": pred.get("sl"),
                "time": datetime.now(timezone.utc).isoformat(),
            })
            updated_count += 1

            # gentileza com provedor
            time.sleep(0.25)

        except Exception as e:
            log.exception("Erro processando %s: %s", sym, e)

    # grava cache
    write_predictions(preds)
    log.info("Predictions gravadas em %s (qtd=%d)", PRED_FILE, len(preds))

    # telegram (opcional)
    if USE_TELEGRAM and send_prediction_alerts is not None and preds:
        try:
            sent_count = send_prediction_alerts(_normalize_for_telegram(preds), threshold=CONF_THRESHOLD)
            log.info("Telegram: mensagens enviadas=%d", sent_count)
        except Exception as e:
            log.exception("Falha ao notificar Telegram: %s", e)
    else:
        if not USE_TELEGRAM:
            log.debug("Telegram desativado (env vars ausentes).")
        elif send_prediction_alerts is None:
            log.debug("notifier_telegram não disponível.")

    summary = {"updated": updated_count, "telegram_sent": sent_count}
    log.info("==== runner done | updated=%d | telegram_sent=%d ====", updated_count, sent_count)
    return summary

if __name__ == "__main__":
    res = run_once()
    print(json.dumps({"ok": True, **res}, ensure_ascii=False))
