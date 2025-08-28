# -*- coding: utf-8 -*-
"""
Sistema de validação e backtesting para trading Forex
"""
import os, json, joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix
from indicators import rsi, macd, ema, bollinger

MODEL_FILE = "forex_model.pkl"
SCALER_FILE = "forex_scaler.pkl"
DATA_RAW_FILE = "forex_data_raw.json"

class ForexValidator:
    def __init__(self):
        self.model = joblib.load(MODEL_FILE)
        self.scaler = joblib.load(SCALER_FILE)
        
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

    def calculate_features_for_validation(self, ohlc_data):
        """Calcula features para validação"""
        if len(ohlc_data) < 100:
            return None, None, None
        
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
        
        sma10 = pd.Series(closes).rolling(10).mean().tolist()
        sma200 = pd.Series(closes).rolling(200).mean().tolist()
        atr = self.calculate_atr(highs, lows, closes, 14)
        price_change = [(closes[i] - closes[i-1]) / closes[i-1] * 100 if i > 0 else 0 for i in range(len(closes))]
        volatility = pd.Series(closes).rolling(20).std().tolist()
        spread = [(highs[i] - lows[i]) / closes[i] * 100 for i in range(len(closes))]
        
        features_list = []
        targets_list = []
        timestamps = []
        
        # Criar features para cada ponto temporal
        for i in range(50, len(closes) - 10):
            if (rsi_values[i] is not None and macd_line[i] is not None and
                ema20[i] is not None and bb_upper[i] is not None and atr[i] is not None):
                
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
                
                # Target real (movimento futuro)
                future_prices = closes[i+1:i+11]
                future_return = (max(future_prices) - current_price) / current_price
                target = 1 if future_return > 0.005 else 0  # 0.5% para Forex
                
                features_list.append(features)
                targets_list.append(target)
                timestamps.append(i)
        
        return np.array(features_list), np.array(targets_list), timestamps

    def backtest_forex_strategy(self, forex_data):
        """Executa backtesting da estratégia Forex"""
        print("📈 Executando backtesting Forex...")
        
        all_predictions = []
        all_targets = []
        all_returns = []
        trades = []
        
        for pair_data in forex_data:
            symbol = pair_data['symbol']
            pair_name = pair_data['pair_name']
            ohlc = pair_data['ohlc']
            
            features, targets, timestamps = self.calculate_features_for_validation(ohlc)
            if features is None or len(features) == 0:
                continue
            
            # Fazer predições
            features_scaled = self.scaler.transform(features)
            predictions = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            closes = [candle['close'] for candle in ohlc]
            highs = [candle['high'] for candle in ohlc]
            lows = [candle['low'] for candle in ohlc]
            
            # Calcular ATR para níveis de TP/SL
            atr_values = self.calculate_atr(highs, lows, closes, 14)
            
            # Simular trades
            for i, (pred, prob, target, ts) in enumerate(zip(predictions, probabilities, targets, timestamps)):
                confidence = max(prob)
                
                if confidence > 0.8:  # Alta confiança para Forex
                    entry_price = closes[ts]
                    atr_value = atr_values[ts] if ts < len(atr_values) else 0.001
                    
                    # Simular resultado do trade (próximos 10 períodos)
                    if ts + 10 < len(closes):
                        future_prices = closes[ts+1:ts+11]
                        
                        if pred == 1:  # Compra
                            tp_price = entry_price + (atr_value * 2)
                            sl_price = entry_price - (atr_value * 1)
                            
                            # Verificar se TP ou SL foi atingido
                            max_price = max(future_prices)
                            min_price = min(future_prices)
                            
                            if max_price >= tp_price:
                                trade_return = (tp_price - entry_price) / entry_price
                            elif min_price <= sl_price:
                                trade_return = (sl_price - entry_price) / entry_price
                            else:
                                exit_price = future_prices[-1]
                                trade_return = (exit_price - entry_price) / entry_price
                        else:  # Venda
                            tp_price = entry_price - (atr_value * 2)
                            sl_price = entry_price + (atr_value * 1)
                            
                            max_price = max(future_prices)
                            min_price = min(future_prices)
                            
                            if min_price <= tp_price:
                                trade_return = (entry_price - tp_price) / entry_price
                            elif max_price >= sl_price:
                                trade_return = (entry_price - sl_price) / entry_price
                            else:
                                exit_price = future_prices[-1]
                                trade_return = (entry_price - exit_price) / entry_price
                        
                        trades.append({
                            'symbol': symbol,
                            'pair_name': pair_name,
                            'prediction': pred,
                            'confidence': confidence,
                            'entry_price': entry_price,
                            'tp_price': tp_price,
                            'sl_price': sl_price,
                            'return': trade_return,
                            'target': target,
                            'timestamp': ts,
                            'atr': atr_value
                        })
                        
                        all_returns.append(trade_return)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
        
        return trades, all_predictions, all_targets, all_returns

    def create_forex_performance_report(self, trades, predictions, targets, returns):
        """Cria relatório de performance para Forex"""
        print("\n📊 RELATÓRIO DE PERFORMANCE FOREX")
        print("=" * 50)
        
        # Métricas básicas
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t['return'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_return = np.mean(returns) if returns else 0
        total_return = np.sum(returns) if returns else 0
        max_return = max(returns) if returns else 0
        min_return = min(returns) if returns else 0
        
        # Métricas específicas para Forex
        avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loss = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        print(f"📈 Total de trades: {total_trades}")
        print(f"🎯 Taxa de acerto: {win_rate:.2%}")
        print(f"💰 Retorno médio por trade: {avg_return:.4f} ({avg_return*100:.2f}%)")
        print(f"📊 Retorno total: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"🚀 Melhor trade: {max_return:.4f} ({max_return*100:.2f}%)")
        print(f"📉 Pior trade: {min_return:.4f} ({min_return*100:.2f}%)")
        print(f"💎 Profit Factor: {profit_factor:.2f}")
        
        # Análise por par de moedas
        print(f"\n📋 PERFORMANCE POR PAR FOREX")
        print("=" * 40)
        
        pair_stats = {}
        for trade in trades:
            pair = trade['pair_name']
            if pair not in pair_stats:
                pair_stats[pair] = {'trades': [], 'returns': []}
            pair_stats[pair]['trades'].append(trade)
            pair_stats[pair]['returns'].append(trade['return'])
        
        for pair, stats in pair_stats.items():
            pair_trades = len(stats['trades'])
            pair_wins = len([t for t in stats['trades'] if t['return'] > 0])
            pair_win_rate = pair_wins / pair_trades if pair_trades > 0 else 0
            pair_avg_return = np.mean(stats['returns']) if stats['returns'] else 0
            
            print(f"{pair}: {pair_trades} trades, {pair_win_rate:.1%} win rate, {pair_avg_return*100:.3f}% avg return")
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'pair_stats': pair_stats
        }

    def create_forex_visualizations(self, trades, returns):
        """Cria visualizações específicas para Forex"""
        print("\n📊 Criando visualizações Forex...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise de Performance do Modelo Forex AI', fontsize=16, fontweight='bold')
        
        # 1. Distribuição de retornos
        axes[0, 0].hist(returns, bins=50, alpha=0.7, color='darkblue', edgecolor='black')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Distribuição de Retornos Forex')
        axes[0, 0].set_xlabel('Retorno (%)')
        axes[0, 0].set_ylabel('Frequência')
        
        # 2. Retornos cumulativos
        cumulative_returns = np.cumsum(returns)
        axes[0, 1].plot(cumulative_returns, color='darkgreen', linewidth=2)
        axes[0, 1].axhline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Retornos Cumulativos Forex')
        axes[0, 1].set_xlabel('Trade #')
        axes[0, 1].set_ylabel('Retorno Cumulativo')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Performance por par
        pair_returns = {}
        for trade in trades:
            pair = trade['pair_name']
            if pair not in pair_returns:
                pair_returns[pair] = []
            pair_returns[pair].append(trade['return'])
        
        pairs = list(pair_returns.keys())
        avg_returns = [np.mean(pair_returns[p]) * 100 for p in pairs]
        
        bars = axes[1, 0].bar(pairs, avg_returns, color=['darkgreen' if r > 0 else 'darkred' for r in avg_returns])
        axes[1, 0].set_title('Retorno Médio por Par Forex')
        axes[1, 0].set_xlabel('Par de Moedas')
        axes[1, 0].set_ylabel('Retorno Médio (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, avg_returns):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + (0.001 if height > 0 else -0.002),
                           f'{value:.3f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        # 4. Distribuição de confiança
        confidences = [trade['confidence'] for trade in trades]
        axes[1, 1].hist(confidences, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Distribuição de Confiança das Predições')
        axes[1, 1].set_xlabel('Confiança')
        axes[1, 1].set_ylabel('Frequência')
        
        plt.tight_layout()
        plt.savefig('forex_performance.png', dpi=300, bbox_inches='tight')
        print("💾 Gráficos Forex salvos em forex_performance.png")
        
        return fig

    def validate_forex_model(self):
        """Executa validação completa do modelo Forex"""
        print("🔍 Iniciando validação do modelo Forex...")
        
        # Carregar dados
        with open(DATA_RAW_FILE, 'r') as f:
            forex_data = json.load(f)
        
        # Executar backtesting
        trades, predictions, targets, returns = self.backtest_forex_strategy(forex_data)
        
        if not trades:
            print("❌ Nenhum trade foi executado no backtesting Forex")
            return
        
        # Criar relatório
        performance = self.create_forex_performance_report(trades, predictions, targets, returns)
        
        # Criar visualizações
        fig = self.create_forex_visualizations(trades, returns)
        
        # Salvar resultados
        results = {
            'performance_summary': performance,
            'trades': trades,
            'total_trades': len(trades),
            'timestamp': datetime.now().isoformat()
        }
        
        with open('forex_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print("💾 Resultados Forex salvos em forex_backtest_results.json")
        print("✅ Validação Forex concluída!")

def main():
    """Função principal"""
    validator = ForexValidator()
    validator.validate_forex_model()

if __name__ == "__main__":
    main()

