# -*- coding: utf-8 -*-
"""
Predictor robusto para sinais de trading.
- Evita FutureWarnings (usa iloc[0])
- Aceita modelos com predict_proba ou decision_function
- Seleciona colunas de features automaticamente
- Carrega caminhos via env MODEL_PATH e SCALER_PATH
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional

# Caminhos via Variables (com defaults)
MODEL_PATH = os.getenv("MODEL_PATH", "forex_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "forex_scaler.pkl")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.60"))

# Colunas “não-features” (mantidas fora do vetor X)
NON_FEATURE_COLS = {
    "timestamp", "time", "date", "datetime",
    "open", "high", "low", "close", "volume",
    "symbol", "pair"
}

def _log(msg: str) -> None:
    print(f"[predictor] {msg}", flush=True)

def _load_pickle(path: str):
    if not os.path.exists(path):
        _log(f"ERRO: arquivo não encontrado: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        _log(f"ERRO ao carregar {path}: {e}")
        return None

def _pick_feature_cols(df: pd.DataFrame):
    # tudo que não for OHLC/timestamp/symbol vira feature
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return cols

def _as_float_scalar(series_or_value) -> Optional[float]:
    """
    Converte com segurança Series/DataFrame/célula para float escalar.
    Evita FutureWarning 'float on single element Series'.
    """
    try:
        if isinstance(series_or_value, pd.Series):
            if series_or_value.shape[0] == 0:
                return None
            return float(series_or_value.iloc[0])
        if isinstance(series_or_value, (pd.DataFrame,)):
            if series_or_value.size == 0:
                return None
            return float(series_or_value.iloc[0, 0])
        # valor já escalar
        return float(series_or_value)
    except Exception:
        return None

class Predictor:
    def __init__(self):
        self.model = _load_pickle(MODEL_PATH)
        self.scaler = _load_pickle(SCALER_PATH)

        if self.model is None:
            _log("ATENÇÃO: model=None (sem modelo carregado).")
        if self.scaler is None:
            _log("ATENÇÃO: scaler=None (sem scaler carregado).")

    def _predict_score(self, X: np.ndarray) -> float:
        """
        Retorna “confiança” de compra:
        - Se houver predict_proba => prob da classe 1
        - Senão, usa decision_function normalizado para [0,1]
        """
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            # supõe classe positiva = 1
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return float(proba[0, 1])
            return float(proba.ravel()[0])

        if hasattr(self.model, "decision_function"):
            score = float(np.atleast_1d(self.model.decision_function(X))[0])
            # normalização sigmoide para [0,1]
            return float(1.0 / (1.0 + np.exp(-score)))

        # último recurso: predict (0/1)
        pred = int(np.atleast_1d(self.model.predict(X))[0])
        return float(pred)

    def predict_from_df(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Pega a ÚLTIMA linha do df e gera o sinal.
        Retorna dict com: price, score, signal, features_usadas
        """
        if df is None or df.empty:
            _log("ERRO: df vazio.")
            return None
        if self.model is None or self.scaler is None:
            _log("ERRO: modelo/scaler não carregados.")
            return None

        last = df.tail(1).copy()
        # preço “close” para log/telemetria
        price = _as_float_scalar(last.get("close"))
        symbol = last.get("symbol")
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0] if len(symbol) else None

        feature_cols = _pick_feature_cols(df)
        if not feature_cols:
            _log("ERRO: não há colunas de features (apenas OHLC?).")
            return None

        # Garantir numérico nas features
        X_row = last[feature_cols].apply(pd.to_numeric, errors="coerce")
        if X_row.isna().any(axis=None):
            # substitui NaNs por 0 para não quebrar (ou use outro tratamento)
            X_row = X_row.fillna(0.0)

        try:
            X_scaled = self.scaler.transform(X_row.values)
        except Exception as e:
            _log(f"ERRO ao aplicar scaler: {e}")
            return None

        try:
            score = self._predict_score(X_scaled)
        except Exception as e:
            _log(f"ERRO ao inferir: {e}")
            return None

        signal = "BUY" if score >= CONF_THRESHOLD else "HOLD"

        _log(
            f"symbol={symbol} price={price} "
            f"score={score:.3f} thr={CONF_THRESHOLD:.2f} -> signal={signal}"
        )

        return {
            "symbol": symbol,
            "price": price,
            "score": float(score),
            "threshold": CONF_THRESHOLD,
            "signal": signal,
            "features_used": feature_cols,
        }
