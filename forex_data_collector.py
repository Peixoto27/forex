# -*- coding: utf-8 -*-
"""
Coletor de dados OHLC para Yahoo Finance usando parâmetros do ambiente.
Lê:
  - INTERVAL  (ex.: "1h", "60m", "15m", "1d"...)
  - PERIOD    (ex.: "5d", "60d", "1mo", "3mo"...)
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any, List


VALID_RANGES = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
VALID_INTERVALS = {
    "1m", "2m", "5m", "15m", "30m", "60m", "90m",
    "1h", "1d", "5d", "1wk", "1mo", "3mo"
}


def _normalize_interval(itv: str) -> str:
    """Normaliza intervalos comuns (ex.: 1h -> 60m)."""
    itv = (itv or "").strip()
    if itv == "1h":
        return "60m"   # compatível com Yahoo
    return itv


def _pick_env_params() -> (str, str):
    """Lê e valida PERIOD/INTERVAL do ambiente."""
    period = os.getenv("PERIOD", "60d").strip()
    interval = _normalize_interval(os.getenv("INTERVAL", "1h").strip())

    # Correções comuns
    if period.isdigit():
        # Usuário colocou "60" sem sufixo -> assume "60d"
        period = f"{period}d"

    if period not in VALID_RANGES:
        # "60d" não está na lista do Yahoo "range", mas funciona via yfinance.
        # Para a API de 'get_stock_chart', use o mais próximo: "3mo" (≈ 90d)
        # ou "1mo"/"6mo" conforme seu caso.
        # Aqui vamos mapear 60d -> 3mo.
        if period == "60d":
            period = "3mo"
        elif period.endswith("d"):
            # fallback genérico pra não quebrar
            num = int(period[:-1]) if period[:-1].isdigit() else 30
            period = "1mo" if num <= 30 else "3mo"

    if interval not in VALID_INTERVALS:
        interval = "60m"

    return period, interval


class ForexDataCollector:
    def __init__(self, api_client):
        self.api_client = api_client

    def fetch_ohlcv(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Baixa OHLC com base nas envs PERIOD/INTERVAL e retorna
        um dicionário com listas: timestamps, open, high, low, close, volume.
        """
        period, interval = _pick_env_params()
        print(f"[collector] symbol={symbol} period={period} interval={interval}")

        # Chamada à API do Yahoo (o mesmo endpoint que você já usava)
        response = self.api_client.call_api(
            "YahooFinance/get_stock_chart",
            query={
                "symbol": symbol,
                "region": "US",
                "interval": interval,
                "range": period,                 # 'range' no Yahoo == 'period'
                "includeAdjustedClose": True,
            }
        )

        if not response or "chart" not in response:
            print(f"❌ Erro: Resposta inválida para {symbol}")
            return None

        chart = response["chart"]
        if not chart.get("result"):
            print(f"❌ Erro: Sem dados para {symbol}")
            return None

        result = chart["result"][0]

        # Extrai OHLC
        ts: List[int] = result.get("timestamp", [])
        indicators = result.get("indicators", {})
        quote = (indicators.get("quote") or [{}])[0]

        if not ts or not quote:
            print(f"❌ Erro: Dados incompletos para {symbol}")
            return None

        o = quote.get("open", [])
        h = quote.get("high", [])
        l = quote.get("low", [])
        c = quote.get("close", [])
        v = quote.get("volume", [])

        n = min(len(ts), len(o), len(h), len(l), len(c), len(v))
        if n == 0:
            print(f"❌ Erro: Listas vazias para {symbol}")
            return None

        return {
            "timestamps": ts[:n],
            "open": o[:n],
            "high": h[:n],
            "low": l[:n],
            "close": c[:n],
            "volume": v[:n],
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "last_update": datetime.utcnow().isoformat() + "Z",
        }
