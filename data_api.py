# -*- coding: utf-8 -*-
import datetime as dt
import yfinance as yf
import pandas as pd

def fetch_ohlcv(symbol: str, period="60d", interval="1h") -> pd.DataFrame:
    """
    symbol: ex. 'EURUSD=X', 'GBPUSD=X', 'BTC-USD'
    period: '7d','30d','60d','1y'...
    interval: '15m','1h','4h','1d'
    """
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns=str.lower).rename(columns={
        "open":"open", "high":"high", "low":"low", "close":"close", "volume":"volume"
    })
    df = df.dropna().reset_index().rename(columns={"index": "datetime"})
    if "Datetime" in df.columns: df.rename(columns={"Datetime":"datetime"}, inplace=True)
    return df
