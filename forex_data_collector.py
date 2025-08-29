# -*- coding: utf-8 -*-
"""
Coletor de dados históricos para pares de moedas Forex
Adaptado do sistema de criptomoedas para trabalhar com Forex
"""
import sys
import os
sys.path.append('/opt/.manus/.sandbox-runtime')

import requests
import json
import time
from datetime import datetime, timedelta
import pandas as pd
from data_api import ApiClient

# Principais pares de moedas Forex
FOREX_PAIRS = [
    "EURUSD=X",  # Euro/Dólar
    "GBPUSD=X",  # Libra/Dólar  
    "USDJPY=X",  # Dólar/Iene
    "USDCHF=X",  # Dólar/Franco Suíço
    "AUDUSD=X",  # Dólar Australiano/Dólar
    "USDCAD=X"   # Dólar/Dólar Canadense
]

class ForexDataCollector:
    def __init__(self):
        self.api_client = ApiClient()
        self.data_file = "forex_data_raw.json"
        
    def fetch_forex_data(self, symbol, days=30):
        """
        Coleta dados históricos de um par de moedas
        """
        try:
            print(f"📊 Coletando dados para {symbol} ({days} dias)...")
            
            # Determinar range baseado nos dias
            if days <= 5:
                range_param = "5d"
            elif days <= 30:
                range_param = "1mo"
            elif days <= 90:
                range_param = "3mo"
            else:
                range_param = "6mo"
           interval = "60m"  # sempre velas de 1 hora
        
            response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                'symbol': symbol,
                'region': 'US',
                'interval': interval,  # Dados horários para Forex
                'range': range_param,
                'includeAdjustedClose': True
            })
            
            if not response or 'chart' not in response:
                print(f"❌ Erro: Resposta inválida para {symbol}")
                return None
                
            chart_data = response['chart']
            if not chart_data.get('result'):
                print(f"❌ Erro: Sem dados para {symbol}")
                return None
                
            result = chart_data['result'][0]
            
            # Extrair dados OHLC
            timestamps = result.get('timestamp', [])
            quotes = result.get('indicators', {}).get('quote', [{}])[0]
            
            if not timestamps or not quotes:
                print(f"❌ Erro: Dados incompletos para {symbol}")
                return None
            
            # Converter para formato padronizado
            ohlc_data = []
            for i, ts in enumerate(timestamps):
                if (i < len(quotes.get('open', [])) and 
                    quotes['open'][i] is not None and
                    quotes['high'][i] is not None and
                    quotes['low'][i] is not None and
                    quotes['close'][i] is not None):
                    
                    ohlc_data.append({
                        "timestamp": ts,
                        "open": float(quotes['open'][i]),
                        "high": float(quotes['high'][i]),
                        "low": float(quotes['low'][i]),
                        "close": float(quotes['close'][i]),
                        "volume": int(quotes.get('volume', [0])[i] or 0)
                    })
            
            print(f"✅ {symbol}: {len(ohlc_data)} candles coletados")
            return ohlc_data
            
        except Exception as e:
            print(f"❌ Erro ao coletar {symbol}: {str(e)}")
            return None
    
    def collect_all_pairs(self, days=30):
        """
        Coleta dados de todos os pares de moedas
        """
        print("🌍 Iniciando coleta de dados Forex...")
        all_data = []
        
        for pair in FOREX_PAIRS:
            ohlc_data = self.fetch_forex_data(pair, days)
            
            if ohlc_data and len(ohlc_data) >= 50:  # Mínimo de dados
                all_data.append({
                    "symbol": pair,
                    "pair_name": self.get_pair_name(pair),
                    "ohlc": ohlc_data,
                    "collected_at": datetime.now().isoformat()
                })
            
            # Delay para evitar rate limiting
            time.sleep(1)
        
        # Salvar dados
        with open(self.data_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        print(f"💾 Dados salvos: {len(all_data)} pares em {self.data_file}")
        return all_data
    
    def get_pair_name(self, symbol):
        """
        Converte símbolo para nome legível
        """
        pair_names = {
            "EURUSD=X": "EUR/USD",
            "GBPUSD=X": "GBP/USD", 
            "USDJPY=X": "USD/JPY",
            "USDCHF=X": "USD/CHF",
            "AUDUSD=X": "AUD/USD",
            "USDCAD=X": "USD/CAD"
        }
        return pair_names.get(symbol, symbol)
    
    def get_current_prices(self):
        """
        Obtém preços atuais dos pares
        """
        current_data = []
        
        for pair in FOREX_PAIRS:
            try:
                response = self.api_client.call_api('YahooFinance/get_stock_chart', query={
                    'symbol': pair,
                    'region': 'US',
                    'interval': '1m',
                    'range': '1d'
                })
                
                if response and 'chart' in response and response['chart'].get('result'):
                    result = response['chart']['result'][0]
                    meta = result.get('meta', {})
                    
                    current_data.append({
                        'symbol': pair,
                        'pair_name': self.get_pair_name(pair),
                        'current_price': meta.get('regularMarketPrice', 0),
                        'change_24h': meta.get('regularMarketChangePercent', 0),
                        'high_24h': meta.get('regularMarketDayHigh', 0),
                        'low_24h': meta.get('regularMarketDayLow', 0),
                        'timestamp': datetime.now().isoformat()
                    })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Erro ao obter preço atual de {pair}: {e}")
        
        return current_data

def test_forex_collector():
    """
    Testa o coletor de dados Forex
    """
    print("🧪 Testando coletor de dados Forex...")
    
    collector = ForexDataCollector()
    
    # Testar um par específico
    test_data = collector.fetch_forex_data("EURUSD=X", days=7)
    
    if test_data:
        print(f"✅ Teste bem-sucedido: {len(test_data)} candles para EUR/USD")
        print(f"📊 Primeiro candle: {test_data[0]}")
        print(f"📊 Último candle: {test_data[-1]}")
        return True
    else:
        print("❌ Teste falhou")
        return False

if __name__ == "__main__":
    # Testar primeiro
    if test_forex_collector():
        # Se teste passou, coletar todos os dados
        collector = ForexDataCollector()
        data = collector.collect_all_pairs(days=30)
        
        print(f"\n🎉 Coleta concluída!")
        print(f"📈 Pares coletados: {len(data)}")
        for item in data:
            print(f"   {item['pair_name']}: {len(item['ohlc'])} candles")
    else:
        print("❌ Falha no teste inicial")

