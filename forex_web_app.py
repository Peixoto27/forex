# -*- coding: utf-8 -*-
"""
API Flask (usada com Gunicorn).
Ex.: gunicorn -w 2 -b 0.0.0.0:$PORT forex_web_app:APP
"""

import os
from flask import Flask, jsonify, request
from predictor import predict_last

APP = Flask(__name__)

@APP.get("/")
def index():
    return jsonify({
        "name": "forex-web",
        "ok": True,
        "endpoints": ["/healthz", "/api/forex/status", "/api/forex/force-update"],
    })

@APP.get("/healthz")
def healthz():
    return jsonify({"ok": True})

@APP.get("/api/forex/status")
def status():
    return jsonify({
        "ok": True,
        "period": os.getenv("PERIOD", "3mo"),
        "interval": os.getenv("INTERVAL", "60m"),
        "symbols": os.getenv("SYMBOLS", "EURUSD=X,GBPUSD=X,USDJPY=X,BTC-USD"),
    })

@APP.get("/api/forex/force-update")
def force_update():
    # (opcional) proteção simples por key
    need_key = os.getenv("FORCE_KEY", "")
    if need_key:
        if request.args.get("key") != need_key:
            return jsonify({"ok": False, "error": "unauthorized"}), 401

    symbols  = (os.getenv("SYMBOLS") or "EURUSD=X,GBPUSD=X,USDJPY=X,BTC-USD").split(",")
    period   = os.getenv("PERIOD", "3mo")
    interval = os.getenv("INTERVAL", "60m")
    thr      = float(os.getenv("CONF_THRESHOLD", "0.60"))

    out = predict_last(symbols=symbols, period=period, interval=interval, conf_threshold=thr)
    code = 200 if out.get("success") else 500
    return jsonify(out), code
