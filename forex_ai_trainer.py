# -*- coding: utf-8 -*-
"""
Sistema de treinamento de IA para trading Forex
Adaptado do sistema de criptomoedas para pares de moedas
"""
import os, json, joblib, numpy as np, pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Importar funções de indicadores técnicos
from indicators import rsi, macd, ema, bollinger

# Configurações específicas para Forex
DATA_RAW_FILE = "forex_data_raw.json"
MODEL_FILE = "forex_model.pkl"
SCALER_FILE = "forex_scaler.pkl"
MIN_SAMPLES = 100  # Mínimo de amostras para treinar

class ForexAITrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_forex_data(self):
        """Carrega dados Forex do arquivo JSON"""
        if not os.path.exists(DATA_RAW_FILE):
            raise FileNotFoundError(f"{DATA_RAW_FILE} não encontrado.")
        
        with open(DATA_RAW_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calculate_forex_indicators(self, ohlc_data):
        """Calcula indicadores técnicos específicos para Forex"""
        if len(ohlc_data) < 100:  # Forex precisa de mais dados históricos
            return None
        
        # Extrair preços
        closes = [candle['close'] for candle in ohlc_data]
        highs = [candle['high'] for candle in ohlc_data]
        lows = [candle['low'] for candle in ohlc_data]
        opens = [candle['open'] for candle in ohlc_data]
        
        # Calcular indicadores técnicos
        rsi_values = rsi(closes, 14)
        macd_line, signal_line, histogram = macd(closes, 12, 26, 9)
        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50)
        ema100 = ema(closes, 100)  # EMA adicional para Forex
        bb_upper, bb_middle, bb_lower = bollinger(closes, 20, 2.0)
        
        # Indicadores específicos para Forex
        sma10 = pd.Series(closes).rolling(10).mean().tolist()
        sma200 = pd.Series(closes).rolling(200).mean().tolist()  # Tendência de longo prazo
        
        # Volatilidade (importante no Forex)
        atr = self.calculate_atr(highs, lows, closes, 14)  # Average True Range
        
        # Momentum e mudanças de preço
        price_change = [(closes[i] - closes[i-1]) / closes[i-1] * 100 if i > 0 else 0 for i in range(len(closes))]
        volatility = pd.Series(closes).rolling(20).std().tolist()
        
        # Spread (diferença entre high e low)
        spread = [(highs[i] - lows[i]) / closes[i] * 100 for i in range(len(closes))]
        
        return {
            'rsi': rsi_values,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram,
            'ema20': ema20,
            'ema50': ema50,
            'ema100': ema100,
            'bb_upper': bb_upper,
            'bb_middle': bb_middle,
            'bb_lower': bb_lower,
            'sma10': sma10,
            'sma200': sma200,
            'atr': atr,
            'price_change': price_change,
            'volatility': volatility,
            'spread': spread,
            'closes': closes,
            'highs': highs,
            'lows': lows,
            'opens': opens
        }
    
    def calculate_atr(self, highs, lows, closes, period=14):
        """Calcula Average True Range (ATR) - importante para Forex"""
        tr_list = []
        
        for i in range(1, len(highs)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            tr = max(tr1, tr2, tr3)
            tr_list.append(tr)
        
        # Calcular ATR usando média móvel
        atr = [0]  # Primeiro valor é 0
        for i in range(len(tr_list)):
            if i < period - 1:
                atr.append(0)
            else:
                atr_value = sum(tr_list[i-period+1:i+1]) / period
                atr.append(atr_value)
        
        return atr

    def create_forex_features_and_targets(self, forex_data):
        """Cria features e targets específicos para Forex"""
        X, y, pairs = [], [], []
        
        for pair_data in forex_data:
            symbol = pair_data['symbol']
            ohlc = pair_data['ohlc']
            
            indicators = self.calculate_forex_indicators(ohlc)
            if indicators is None:
                continue
            
            # Criar features para cada ponto temporal (últimos 50 pontos para Forex)
            for i in range(50, len(indicators['closes'])):
                features = []
                
                # Verificar se todos os indicadores estão disponíveis
                if (indicators['rsi'][i] is not None and 
                    indicators['macd_line'][i] is not None and
                    indicators['ema20'][i] is not None and
                    indicators['bb_upper'][i] is not None and
                    indicators['atr'][i] is not None):
                    
                    # Features básicas
                    features.extend([
                        indicators['rsi'][i],
                        indicators['macd_line'][i],
                        indicators['signal_line'][i] if indicators['signal_line'][i] is not None else 0,
                        indicators['histogram'][i] if indicators['histogram'][i] is not None else 0,
                        indicators['ema20'][i],
                        indicators['ema50'][i] if indicators['ema50'][i] is not None else indicators['ema20'][i],
                        indicators['ema100'][i] if indicators['ema100'][i] is not None else indicators['ema20'][i],
                        indicators['bb_upper'][i],
                        indicators['bb_middle'][i],
                        indicators['bb_lower'][i],
                        indicators['sma10'][i] if indicators['sma10'][i] is not None else indicators['closes'][i],
                        indicators['sma200'][i] if indicators['sma200'][i] is not None else indicators['closes'][i],
                        indicators['atr'][i],
                        indicators['price_change'][i],
                        indicators['volatility'][i] if indicators['volatility'][i] is not None else 0,
                        indicators['spread'][i],
                    ])
                    
                    # Features de contexto específicas para Forex
                    current_price = indicators['closes'][i]
                    features.extend([
                        current_price / indicators['ema20'][i] - 1,  # Distância da EMA20
                        current_price / indicators['bb_middle'][i] - 1,  # Posição nas Bollinger Bands
                        (indicators['bb_upper'][i] - indicators['bb_lower'][i]) / indicators['bb_middle'][i],  # Largura das BB
                        indicators['atr'][i] / current_price,  # ATR normalizado
                        indicators['spread'][i],  # Spread atual
                    ])
                    
                    # Target: movimento futuro (próximos 10 períodos para Forex)
                    future_prices = indicators['closes'][i+1:i+11] if i+10 < len(indicators['closes']) else []
                    if len(future_prices) >= 5:
                        # Para Forex, consideramos movimentos menores (0.5% em vez de 2%)
                        future_return = (max(future_prices) - current_price) / current_price
                        target = 1 if future_return > 0.005 else 0  # 0.5% de movimento mínimo
                        
                        X.append(features)
                        y.append(target)
                        pairs.append(symbol)
        
        return np.array(X), np.array(y), pairs

    def train_forex_model(self, X, y):
        """Treina o modelo de IA para Forex"""
        print(f"📊 Treinando modelo Forex com {len(X)} amostras...")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Tentar XGBoost primeiro, depois Random Forest
        try:
            from xgboost import XGBClassifier
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=8,  # Maior profundidade para Forex
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                eval_metric='logloss'
            )
            print("🚀 Usando XGBoost para Forex")
        except ImportError:
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42
            )
            print("🌲 Usando Random Forest para Forex")
        
        # Treinar modelo
        self.model.fit(X_train_scaled, y_train)
        
        # Avaliar modelo
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Acurácia Forex: {accuracy:.3f}")
        print(f"📈 Distribuição de classes - Treino: {np.bincount(y_train)}")
        print(f"📈 Distribuição de classes - Teste: {np.bincount(y_test)}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"🔄 Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Relatório detalhado
        print("\n📋 Relatório de classificação Forex:")
        print(classification_report(y_test, y_pred))
        
        return accuracy

    def save_model(self):
        """Salva o modelo e scaler treinados"""
        if self.model is not None and self.scaler is not None:
            joblib.dump(self.model, MODEL_FILE)
            joblib.dump(self.scaler, SCALER_FILE)
            print(f"💾 Modelo Forex salvo em {MODEL_FILE}")
            print(f"💾 Scaler Forex salvo em {SCALER_FILE}")
        else:
            print("❌ Modelo ou scaler não disponível para salvar")

    def train_complete_system(self):
        """Executa o treinamento completo do sistema Forex"""
        print("🌍 Iniciando treinamento do sistema Forex AI...")
        
        # Carregar dados
        try:
            forex_data = self.load_forex_data()
            print(f"📁 Carregados dados de {len(forex_data)} pares Forex")
        except FileNotFoundError as e:
            print(f"❌ Erro: {e}")
            return None
        
        # Criar features e targets
        X, y, pairs = self.create_forex_features_and_targets(forex_data)
        
        if len(X) < MIN_SAMPLES:
            print(f"⚠️ Poucos dados para treinamento: {len(X)} < {MIN_SAMPLES}")
            return None
        
        print(f"🔢 Features criadas: {X.shape}")
        print(f"🎯 Distribuição de targets: {np.bincount(y)}")
        
        # Treinar modelo
        accuracy = self.train_forex_model(X, y)
        
        # Salvar modelo
        self.save_model()
        
        print(f"🕒 Treinamento Forex concluído: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return accuracy

def main():
    """Função principal"""
    trainer = ForexAITrainer()
    accuracy = trainer.train_complete_system()
    
    if accuracy:
        print(f"\n🎉 Sistema Forex AI treinado com sucesso!")
        print(f"📊 Acurácia final: {accuracy:.1%}")
    else:
        print("\n❌ Falha no treinamento do sistema Forex")

if __name__ == "__main__":
    main()

