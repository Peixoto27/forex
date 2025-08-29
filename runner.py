# -*- coding: utf-8 -*-
"""
Runner offline para gerar sinais e (opcionalmente) notificar no Telegram.
Use com:  python runner.py
Ideal para Scheduler/Cron no Railway (serviço worker).
"""
import os, json, time
from datetime import datetime, timezone
from typing import List

from data_api import fetch_ohlcv
from features import add_indicators
from predictor import predict_last

# Telegram (opcional)
USE_TELEGRAM = bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))
if USE_TELEGRAM:
    try:
        from notifier_telegram import send_prediction_alerts
    except Exception:
        send_prediction_alerts = None
else:
    send_prediction_alerts = None

PRED_FILE = os.getenv("PRED_FILE", "forex_predictions.json")
SYMBOLS = os.getenv("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X,BTC-USD").split(",")
INTERVAL = os.getenv("INTERVAL", "60m")
PERIOD   = os.getenv("PERIOD", "3mo")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))

print(f"[runner] PERIOD={PERIOD} INTERVAL={INTERVAL} CONF_THRESHOLD={CONF_THRESHOLD}")
print(f"[runner] SYMBOLS={SYMBOLS}")

def write_predictions(preds: List[dict]):
    data = {"generated_at": datetime.now(timezone.utc).isoformat(), "predictions": preds}
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(data["predictions"], f, ensure_ascii=False, indent=2)

def run_once() -> dict:
    preds = []
    for sym in SYMBOLS:
        try:
            df = fetch_ohlcv(sym, period=PERIOD, interval=INTERVAL)
            if df.empty or len(df) < 60:
                continue
            df_feat = add_indicators(df)
            if df_feat.empty:
                continue
            preds.append(predict_last(df_feat, symbol=sym))
            time.sleep(0.3)  # gentileza
        except Exception:
            continue

    write_predictions(preds)

    sent = 0
    if send_prediction_alerts is not None and preds:
        try:
            sent = send_prediction_alerts(preds, threshold=CONF_THRESHOLD)
        except Exception:
            sent = 0

    return {"updated": len(preds), "telegram_sent": sent}

if __name__ == "__main__":
    res = run_once()
    print(json.dumps({"ok": True, **res}, ensure_ascii=False))
