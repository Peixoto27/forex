# -*- coding: utf-8 -*-
"""
Sistema de predição para trading Forex
"""
import os, json, joblib, numpy as np
from datetime import datetime
from indicators import rsi, macd, ema, bollinger
import pandas as pd

MODEL_FILE = "forex_model.pkl"
SCALER_FILE = "forex_scaler.pkl"

class ForexPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Carrega modelo e scaler treinados"""
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
            self.model = joblib.load(MODEL_FILE)
            self.scaler = joblib.load(SCALER_FILE)
            print("✅ Modelo Forex carregado com sucesso")
            return True
        else:
            print("❌ Modelo Forex não encontrado")
            return False
    
    def calculate_atr(self, highs, lows, closes, period=14):
        """Calcula Average True Range (ATR)"""
        tr_list = []
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr = max(tr1, tr2, tr3)
            tr_list.append(tr)
        
        atr = [0]
        for i in range(len(tr_list)):
            if i < period - 1:
                atr.append(0)
            else:
                atr_value = sum(tr_list[i-period+1:i+1]) / period
                atr.append(atr_value)
        
        return atr

    def calculate_features_for_prediction(self, ohlc_data):
        """Calcula features para predição a partir dos dados OHLC"""
        if len(ohlc_data) < 100:
            return None
        
        # Extrair preços
        closes = [candle['close'] for candle in ohlc_data]
        highs = [candle['high'] for candle in ohlc_data]
        lows = [candle['low'] for candle in ohlc_data]
        opens = [candle['open'] for candle in ohlc_data]
        
        # Calcular indicadores
        rsi_values = rsi(closes, 14)
        macd_line, signal_line, histogram = macd(closes, 12, 26, 9)
        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50)
        ema100 = ema(closes, 100)
        bb_upper, bb_middle, bb_lower = bollinger(closes, 20, 2.0)
        
        # Indicadores adicionais
        sma10 = pd.Series(closes).rolling(10).mean().tolist()
        sma200 = pd.Series(closes).rolling(200).mean().tolist()
        atr = self.calculate_atr(highs, lows, closes, 14)
        price_change = [(closes[i] - closes[i-1]) / closes[i-1] * 100 if i > 0 else 0 for i in range(len(closes))]
        volatility = pd.Series(closes).rolling(20).std().tolist()
        spread = [(highs[i] - lows[i]) / closes[i] * 100 for i in range(len(closes))]
        
        # Usar último ponto para predição
        i = -1
        
        if (rsi_values[i] is None or macd_line[i] is None or 
            ema20[i] is None or bb_upper[i] is None or atr[i] is None):
            return None
        
        features = [
            rsi_values[i],
            macd_line[i],
            signal_line[i] if signal_line[i] is not None else 0,
            histogram[i] if histogram[i] is not None else 0,
            ema20[i],
            ema50[i] if ema50[i] is not None else ema20[i],
            ema100[i] if ema100[i] is not None else ema20[i],
            bb_upper[i],
            bb_middle[i],
            bb_lower[i],
            sma10[i] if sma10[i] is not None else closes[i],
            sma200[i] if sma200[i] is not None else closes[i],
            atr[i],
            price_change[i],
            volatility[i] if volatility[i] is not None else 0,
            spread[i],
        ]
        
        # Features de contexto
        current_price = closes[i]
        features.extend([
            current_price / ema20[i] - 1,
            current_price / bb_middle[i] - 1,
            (bb_upper[i] - bb_lower[i]) / bb_middle[i],
            atr[i] / current_price,
            spread[i],
        ])
        
        return np.array(features).reshape(1, -1)

    def predict_forex_signal(self, symbol, ohlc_data):
        """Faz predição para um par de moedas específico"""
        if self.model is None or self.scaler is None:
            return None, "Modelo não carregado"
        
        features = self.calculate_features_for_prediction(ohlc_data)
        if features is None:
            return None, "Dados insuficientes para predição"
        
        # Normalizar features
        features_scaled = self.scaler.transform(features)
        
        # Fazer predição
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        confidence = max(probability)
        signal = "COMPRA" if prediction == 1 else "VENDA"
        
        # Calcular níveis de entrada, TP e SL baseados no ATR
        current_price = ohlc_data[-1]['close']
        atr_value = self.calculate_atr(
            [c['high'] for c in ohlc_data[-20:]], 
            [c['low'] for c in ohlc_data[-20:]], 
            [c['close'] for c in ohlc_data[-20:]]
        )[-1]
        
        # Para Forex, usamos ATR para calcular níveis
        if signal == "COMPRA":
            entry_price = current_price
            tp_price = current_price + (atr_value * 2)  # 2x ATR para TP
            sl_price = current_price - (atr_value * 1)  # 1x ATR para SL
        else:
            entry_price = current_price
            tp_price = current_price - (atr_value * 2)
            sl_price = current_price + (atr_value * 1)
        
        return {
            'symbol': symbol,
            'pair_name': self.get_pair_name(symbol),
            'signal': signal,
            'confidence': float(confidence),
            'probability_buy': float(probability[1]),
            'probability_sell': float(probability[0]),
            'entry_price': float(entry_price),
            'tp_price': float(tp_price),
            'sl_price': float(sl_price),
            'atr': float(atr_value),
            'current_price': float(current_price),
            'timestamp': datetime.now().isoformat()
        }, None
    
    def get_pair_name(self, symbol):
        """Converte símbolo para nome legível"""
        pair_names = {
            "EURUSD=X": "EUR/USD",
            "GBPUSD=X": "GBP/USD", 
            "USDJPY=X": "USD/JPY",
            "USDCHF=X": "USD/CHF",
            "AUDUSD=X": "AUD/USD",
            "USDCAD=X": "USD/CAD"
        }
        return pair_names.get(symbol, symbol)

    def test_predictions(self):
        """Testa predições com dados atuais"""
        print("🔮 Testando predições do modelo Forex...")
        
        # Carregar dados atuais
        if not os.path.exists("forex_data_raw.json"):
            print("❌ Arquivo forex_data_raw.json não encontrado")
            return []
        
        with open("forex_data_raw.json", 'r') as f:
            forex_data = json.load(f)
        
        predictions = []
        
        for pair_data in forex_data:
            symbol = pair_data['symbol']
            ohlc = pair_data['ohlc']
            
            result, error = self.predict_forex_signal(symbol, ohlc)
            
            if result:
                predictions.append(result)
                print(f"📊 {result['pair_name']}: {result['signal']} (confiança: {result['confidence']:.3f})")
            else:
                print(f"❌ {symbol}: {error}")
        
        # Salvar predições
        with open("forex_predictions.json", 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"💾 Predições Forex salvas em forex_predictions.json")
        return predictions

if __name__ == "__main__":
    predictor = ForexPredictor()
    if predictor.model is not None:
        predictions = predictor.test_predictions()
        print(f"\n🎉 {len(predictions)} predições Forex geradas com sucesso!")
    else:
        print("❌ Falha ao carregar modelo Forex")

