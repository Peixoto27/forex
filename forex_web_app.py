# -*- coding: utf-8 -*-
import os, json, time
from datetime import datetime, timezone
from typing import List
from flask import Flask, jsonify
from flask_cors import CORS

from data_api import fetch_ohlcv
from features import add_indicators
from predictor import predict_last

# --- Telegram notifier (opcional) ---
USE_TELEGRAM = bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))
if USE_TELEGRAM:
    try:
        from notifier_telegram import send_prediction_alerts
    except Exception:
        send_prediction_alerts = None
else:
    send_prediction_alerts = None

APP = Flask(__name__)
CORS(APP)

PRED_FILE = os.getenv("PRED_FILE", "forex_predictions.json")
SYMBOLS = [s for s in os.getenv("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X,BTC-USD").split(",") if s]
INTERVAL = os.getenv("INTERVAL", "1h")
PERIOD = os.getenv("PERIOD", "60d")

def write_predictions(preds: List[dict]):
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": preds
    }
    with open(PRED_FILE, "w", encoding="utf-8") as f:
        json.dump(data["predictions"], f, ensure_ascii=False, indent=2)

@APP.get("/")
def home():
    return jsonify({
        "service":"Forex Trading AI",
        "mode":"live",
        "endpoints":["/api/forex/status","/api/forex/predictions","/api/forex/force-update"]
    })

@APP.get("/api/forex/status")
def status():
    ok = os.path.exists(PRED_FILE)
    return jsonify({
        "success": True,
        "symbols": SYMBOLS,
        "interval": INTERVAL,
        "period": PERIOD,
        "pred_file_exists": ok,
        "telegram_enabled": bool(USE_TELEGRAM and send_prediction_alerts is not None),
        "conf_threshold": CONF_THRESHOLD,
        "time": datetime.utcnow().isoformat() + "Z"
    })

@APP.get("/api/forex/predictions")
def predictions():
    try:
        with open(PRED_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
        preds = arr if isinstance(arr, list) else arr.get("predictions", [])
        return jsonify({"success": True, "predictions": preds})
    except Exception:
        return jsonify({"success": True, "predictions": []})

@APP.get("/api/forex/force-update")
def force_update():
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
            time.sleep(0.3)
        except Exception:
            continue

    write_predictions(preds)

    sent = 0
    if send_prediction_alerts is not None and preds:
        try:
            sent = send_prediction_alerts(preds, threshold=CONF_THRESHOLD)
        except Exception:
            sent = 0

    return jsonify({"success": True, "updated": len(preds), "telegram_sent": sent})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    APP.run(host="0.0.0.0", port=port)
