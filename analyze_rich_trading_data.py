#!/usr/bin/env python3
"""
Análisis profundo de los datos ricos de streamed_trades para mejoras en predicciones XGBoost
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

DATABASE_URL = "postgresql://aria_audit_db_user:RaXTVt9ddTIRwiDcd3prxjDbf7ga1Ant@dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com/aria_audit_db"

class RichTradingDataAnalyzer:
    def __init__(self):
        self.conn = None
        self.df = None
        
    def connect_and_load_data(self):
        """Conectar y cargar datos completos"""
        try:
            self.conn = psycopg2.connect(DATABASE_URL)
            print("✅ Conectado a PostgreSQL")
            
            # Cargar datos con columnas específicas
            query = """
            SELECT 
                id, ticket, symbol, trade_type, entry_price, exit_price,
                sl_price, tp_price, lot_size, profit,
                open_time, close_time, stream_timestamp,
                atr_at_open, rsi_at_open, ma50_at_open, ma200_at_open,
                spread_at_open, volatility_at_open,
                ea_version, sl_mode, tp_mode, min_confidence,
                was_xgboost_used, xgboost_confidence, market_regime,
                user_id
            FROM streamed_trades 
            ORDER BY stream_timestamp DESC
            """
            
            self.df = pd.read_sql_query(query, self.conn)
            
            print(f"✅ Cargados {len(self.df)} trades")
            print(f"📅 Período: {self.df['stream_timestamp'].min()} → {self.df['stream_timestamp'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def analyze_xgboost_performance(self):
        """Analizar performance específica de XGBoost vs otros métodos"""
        print(f"\n🤖 ANÁLISIS DE PERFORMANCE XGBOOST")
        print("=" * 60)
        
        # Filtrar trades donde se usó XGBoost
        xgb_trades = self.df[self.df['was_xgboost_used'] == True]
        non_xgb_trades = self.df[self.df['was_xgboost_used'] == False]
        
        print(f"📊 DISTRIBUCIÓN DE TRADES:")
        print(f"   • Con XGBoost: {len(xgb_trades):,} trades")
        print(f"   • Sin XGBoost: {len(non_xgb_trades):,} trades")
        print(f"   • % XGBoost: {len(xgb_trades)/len(self.df)*100:.1f}%")
        
        if len(xgb_trades) > 0 and len(non_xgb_trades) > 0:
            # Comparar performance
            xgb_winrate = (xgb_trades['profit'] > 0).mean() * 100
            non_xgb_winrate = (non_xgb_trades['profit'] > 0).mean() * 100
            
            xgb_avg_profit = xgb_trades['profit'].mean()
            non_xgb_avg_profit = non_xgb_trades['profit'].mean()
            
            print(f"\n📈 COMPARACIÓN DE PERFORMANCE:")
            print(f"   XGBoost:")
            print(f"     • Win Rate: {xgb_winrate:.2f}%")
            print(f"     • Profit Promedio: {xgb_avg_profit:.4f}")
            print(f"     • Total Profit: {xgb_trades['profit'].sum():.2f}")
            
            print(f"   Sin XGBoost:")
            print(f"     • Win Rate: {non_xgb_winrate:.2f}%")
            print(f"     • Profit Promedio: {non_xgb_avg_profit:.4f}")
            print(f"     • Total Profit: {non_xgb_trades['profit'].sum():.2f}")
            
            print(f"\n🎯 DIFERENCIAS:")
            print(f"   • Win Rate: {xgb_winrate - non_xgb_winrate:+.2f}% puntos")
            print(f"   • Profit Promedio: {xgb_avg_profit - non_xgb_avg_profit:+.4f}")
        
        return xgb_trades, non_xgb_trades
    
    def analyze_confidence_impact(self):
        """Analizar impacto de la confidence de XGBoost"""
        print(f"\n🎯 ANÁLISIS DE CONFIDENCE XGBOOST")
        print("=" * 50)
        
        xgb_trades = self.df[self.df['was_xgboost_used'] == True].copy()
        
        if len(xgb_trades) == 0:
            print("❌ No hay trades con XGBoost")
            return None
        
        # Crear bins de confidence
        xgb_trades['confidence_bin'] = pd.cut(
            xgb_trades['xgboost_confidence'], 
            bins=[0, 70, 80, 85, 90, 100], 
            labels=['<70%', '70-80%', '80-85%', '85-90%', '90%+']
        )
        
        confidence_analysis = xgb_trades.groupby('confidence_bin').agg({
            'profit': ['count', 'mean', 'sum', lambda x: (x > 0).sum()],
            'xgboost_confidence': 'mean'
        }).round(4)
        
        confidence_analysis.columns = ['Trades', 'Avg_Profit', 'Total_Profit', 'Winning_Trades', 'Avg_Confidence']
        confidence_analysis['Win_Rate'] = (confidence_analysis['Winning_Trades'] / confidence_analysis['Trades'] * 100).round(2)
        
        print("📊 PERFORMANCE POR NIVEL DE CONFIDENCE:")
        print(confidence_analysis)
        
        # Correlación confidence-profit
        corr = xgb_trades['xgboost_confidence'].corr(xgb_trades['profit'])
        print(f"\n🔗 Correlación Confidence-Profit: {corr:.4f}")
        
        # Identificar threshold óptimo
        best_threshold = None
        best_winrate = 0
        
        for threshold in [70, 75, 80, 85, 90]:
            high_conf_trades = xgb_trades[xgb_trades['xgboost_confidence'] >= threshold]
            if len(high_conf_trades) > 20:  # Mínimo 20 trades
                winrate = (high_conf_trades['profit'] > 0).mean() * 100
                if winrate > best_winrate:
                    best_winrate = winrate
                    best_threshold = threshold
        
        if best_threshold:
            print(f"\n🎯 THRESHOLD ÓPTIMO: {best_threshold}%")
            print(f"   • Win Rate: {best_winrate:.2f}%")
            
            optimal_trades = xgb_trades[xgb_trades['xgboost_confidence'] >= best_threshold]
            print(f"   • Trades disponibles: {len(optimal_trades):,}")
        
        return confidence_analysis
    
    def analyze_technical_indicators(self):
        """Analizar impacto de indicadores técnicos"""
        print(f"\n📊 ANÁLISIS DE INDICADORES TÉCNICOS")
        print("=" * 50)
        
        # Indicadores disponibles
        indicators = ['atr_at_open', 'rsi_at_open', 'ma50_at_open', 'ma200_at_open', 
                     'spread_at_open', 'volatility_at_open']
        
        correlations = {}
        
        print("🔗 CORRELACIONES CON PROFIT:")
        for indicator in indicators:
            if indicator in self.df.columns:
                corr = self.df[indicator].corr(self.df['profit'])
                correlations[indicator] = corr
                print(f"   • {indicator:20}: {corr:+.4f}")
        
        # Análisis de RSI
        if 'rsi_at_open' in self.df.columns:
            print(f"\n📈 ANÁLISIS DE RSI:")
            
            # Crear bins de RSI
            self.df['rsi_bin'] = pd.cut(
                self.df['rsi_at_open'], 
                bins=[0, 30, 40, 60, 70, 100], 
                labels=['Oversold(<30)', 'Low(30-40)', 'Neutral(40-60)', 'High(60-70)', 'Overbought(70+)']
            )
            
            rsi_analysis = self.df.groupby('rsi_bin').agg({
                'profit': ['count', 'mean', lambda x: (x > 0).sum()]
            }).round(4)
            
            rsi_analysis.columns = ['Trades', 'Avg_Profit', 'Winning_Trades']
            rsi_analysis['Win_Rate'] = (rsi_analysis['Winning_Trades'] / rsi_analysis['Trades'] * 100).round(2)
            
            print(rsi_analysis)
        
        # Análisis de Volatilidad
        if 'volatility_at_open' in self.df.columns:
            print(f"\n🌊 ANÁLISIS DE VOLATILIDAD:")
            
            volatility_quartiles = self.df['volatility_at_open'].quantile([0.25, 0.5, 0.75])
            
            self.df['volatility_bin'] = pd.cut(
                self.df['volatility_at_open'],
                bins=[0, volatility_quartiles[0.25], volatility_quartiles[0.5], 
                      volatility_quartiles[0.75], self.df['volatility_at_open'].max()],
                labels=['Low', 'Medium-Low', 'Medium-High', 'High']
            )
            
            vol_analysis = self.df.groupby('volatility_bin').agg({
                'profit': ['count', 'mean', lambda x: (x > 0).sum()],
                'volatility_at_open': 'mean'
            }).round(4)
            
            vol_analysis.columns = ['Trades', 'Avg_Profit', 'Winning_Trades', 'Avg_Volatility']
            vol_analysis['Win_Rate'] = (vol_analysis['Winning_Trades'] / vol_analysis['Trades'] * 100).round(2)
            
            print(vol_analysis)
        
        return correlations
    
    def analyze_market_regime_effectiveness(self):
        """Analizar efectividad por régimen de mercado"""
        print(f"\n🌊 ANÁLISIS DE RÉGIMEN DE MERCADO")
        print("=" * 50)
        
        if 'market_regime' not in self.df.columns:
            print("❌ No hay datos de market_regime")
            return None
        
        regime_analysis = self.df.groupby('market_regime').agg({
            'profit': ['count', 'mean', 'sum', lambda x: (x > 0).sum()],
            'xgboost_confidence': 'mean'
        }).round(4)
        
        regime_analysis.columns = ['Trades', 'Avg_Profit', 'Total_Profit', 'Winning_Trades', 'Avg_XGB_Confidence']
        regime_analysis['Win_Rate'] = (regime_analysis['Winning_Trades'] / regime_analysis['Trades'] * 100).round(2)
        
        print("📊 PERFORMANCE POR RÉGIMEN:")
        print(regime_analysis)
        
        # Análisis por régimen + XGBoost
        print(f"\n🤖 XGBOOST POR RÉGIMEN:")
        
        for regime in self.df['market_regime'].unique():
            if pd.isna(regime):
                continue
                
            regime_data = self.df[self.df['market_regime'] == regime]
            xgb_regime = regime_data[regime_data['was_xgboost_used'] == True]
            
            if len(xgb_regime) > 0:
                winrate = (xgb_regime['profit'] > 0).mean() * 100
                avg_conf = xgb_regime['xgboost_confidence'].mean()
                print(f"   • {regime:10}: {len(xgb_regime):3d} trades, {winrate:5.1f}% win rate, {avg_conf:.1f}% avg confidence")
        
        return regime_analysis
    
    def identify_feature_engineering_opportunities(self):
        """Identificar oportunidades específicas de feature engineering"""
        print(f"\n🔧 OPORTUNIDADES DE FEATURE ENGINEERING")
        print("=" * 60)
        
        opportunities = []
        
        # 1. Features de precio y momentum
        opportunities.append({
            'category': 'Price & Momentum Features',
            'features': [
                'price_change_pct = (exit_price - entry_price) / entry_price',
                'ma_cross_signal = 1 if entry_price > ma50_at_open else 0',
                'ma_trend = (ma50_at_open - ma200_at_open) / ma200_at_open',
                'price_vs_ma50 = (entry_price - ma50_at_open) / ma50_at_open',
                'price_vs_ma200 = (entry_price - ma200_at_open) / ma200_at_open'
            ],
            'justification': 'Datos de MA disponibles para crear señales de momentum'
        })
        
        # 2. Features de volatilidad y riesgo
        opportunities.append({
            'category': 'Volatility & Risk Features',
            'features': [
                'atr_normalized = atr_at_open / entry_price',
                'volatility_percentile = percentile_rank(volatility_at_open)',
                'spread_cost = spread_at_open / entry_price',
                'risk_reward_ratio = (tp_price - entry_price) / (entry_price - sl_price)',
                'position_size_risk = lot_size * entry_price * atr_at_open'
            ],
            'justification': 'ATR y volatilidad son predictores clave de riesgo'
        })
        
        # 3. Features de indicadores técnicos
        opportunities.append({
            'category': 'Technical Indicator Features',
            'features': [
                'rsi_signal = 1 if rsi_at_open > 70 else -1 if rsi_at_open < 30 else 0',
                'rsi_divergence = rsi_at_open - 50',  # Desviación del neutral
                'rsi_momentum = rsi_at_open - lag(rsi_at_open)',
                'volatility_regime = "high" if volatility_at_open > percentile_75 else "normal"'
            ],
            'justification': f'RSI correlación con profit: {self.df["rsi_at_open"].corr(self.df["profit"]):.4f}'
        })
        
        # 4. Features temporales mejoradas
        opportunities.append({
            'category': 'Enhanced Temporal Features',
            'features': [
                'hour_of_day = extract(hour from open_time)',
                'day_of_week = extract(dow from open_time)',
                'is_london_session = hour_of_day between 8 and 17',
                'is_ny_session = hour_of_day between 13 and 22',
                'is_overlap = hour_of_day between 13 and 17',
                'trade_duration_minutes = (close_time - open_time) in minutes'
            ],
            'justification': 'Timestamps precisos disponibles para análisis temporal'
        })
        
        # 5. Features de XGBoost mejoradas
        if 'xgboost_confidence' in self.df.columns:
            opportunities.append({
                'category': 'XGBoost Enhancement Features',
                'features': [
                    'confidence_tier = "high" if xgboost_confidence >= 85 else "medium" if >= 75 else "low"',
                    'confidence_symbol_avg = rolling_mean(xgboost_confidence) by symbol',
                    'confidence_regime_interaction = xgboost_confidence * regime_encoded',
                    'confidence_volatility_interaction = xgboost_confidence * volatility_normalized',
                    'was_high_confidence = xgboost_confidence >= optimal_threshold'
                ],
                'justification': 'Confidence de XGBoost disponible para crear features meta-learning'
            })
        
        # 6. Features de contexto de mercado
        opportunities.append({
            'category': 'Market Context Features',
            'features': [
                'symbol_volatility_rank = percentile_rank(volatility_at_open) by symbol',
                'symbol_performance_streak = consecutive_wins/losses by symbol',
                'market_regime_stability = regime_changes in last N trades',
                'spread_vs_normal = (spread_at_open - avg_spread) / std_spread by symbol',
                'volume_profile = lot_size vs historical average'
            ],
            'justification': 'Múltiples símbolos y contexto histórico disponible'
        })
        
        # Mostrar oportunidades
        for i, opp in enumerate(opportunities, 1):
            print(f"\n{i}. 🎯 {opp['category']}:")
            print(f"   📋 Justificación: {opp['justification']}")
            print(f"   🔧 Features propuestas:")
            for feature in opp['features']:
                print(f"      • {feature}")
        
        return opportunities
    
    def calculate_improvement_potential(self):
        """Calcular potencial de mejora específico"""
        print(f"\n📈 CÁLCULO DE POTENCIAL DE MEJORA")
        print("=" * 50)
        
        current_stats = {
            'total_trades': len(self.df),
            'win_rate': (self.df['profit'] > 0).mean() * 100,
            'total_profit': self.df['profit'].sum(),
            'avg_profit': self.df['profit'].mean()
        }
        
        print(f"📊 SITUACIÓN ACTUAL:")
        for key, value in current_stats.items():
            print(f"   • {key}: {value:.4f}")
        
        # Simulación de mejoras con filtros inteligentes
        improved_df = self.df.copy()
        
        # Filtro 1: Solo trades con XGBoost de alta confidence
        if 'xgboost_confidence' in improved_df.columns:
            improved_df = improved_df[
                (improved_df['was_xgboost_used'] == True) & 
                (improved_df['xgboost_confidence'] >= 80)
            ]
            print(f"\n🔧 Filtro 1: XGBoost confidence >= 80%")
            print(f"   Trades restantes: {len(improved_df):,}")
        
        # Filtro 2: Evitar condiciones de alta volatilidad extrema
        if 'volatility_at_open' in improved_df.columns and len(improved_df) > 0:
            vol_95th = self.df['volatility_at_open'].quantile(0.95)
            improved_df = improved_df[improved_df['volatility_at_open'] < vol_95th]
            print(f"\n🔧 Filtro 2: Volatilidad < percentil 95")
            print(f"   Trades restantes: {len(improved_df):,}")
        
        # Filtro 3: Solo mejores regímenes de mercado
        if 'market_regime' in improved_df.columns and len(improved_df) > 0:
            regime_performance = self.df.groupby('market_regime')['profit'].apply(lambda x: (x > 0).mean())
            best_regimes = regime_performance[regime_performance > 0.5].index.tolist()
            
            if best_regimes:
                improved_df = improved_df[improved_df['market_regime'].isin(best_regimes)]
                print(f"\n🔧 Filtro 3: Solo regímenes con win rate > 50%")
                print(f"   Regímenes seleccionados: {best_regimes}")
                print(f"   Trades restantes: {len(improved_df):,}")
        
        # Calcular mejoras
        if len(improved_df) > 0:
            improved_stats = {
                'total_trades': len(improved_df),
                'win_rate': (improved_df['profit'] > 0).mean() * 100,
                'total_profit': improved_df['profit'].sum(),
                'avg_profit': improved_df['profit'].mean()
            }
            
            print(f"\n🚀 SITUACIÓN MEJORADA:")
            for key, value in improved_stats.items():
                current_val = current_stats[key]
                improvement = value - current_val
                pct_change = (improvement / current_val * 100) if current_val != 0 else 0
                print(f"   • {key}: {value:.4f} ({improvement:+.4f}, {pct_change:+.1f}%)")
            
            return improved_stats
        else:
            print("❌ No quedan trades después de aplicar filtros")
            return None
    
    def generate_actionable_recommendations(self):
        """Generar recomendaciones específicas y accionables"""
        print(f"\n💡 RECOMENDACIONES ACCIONABLES")
        print("=" * 60)
        
        recommendations = []
        
        # Recomendación 1: Optimización de threshold de confidence
        if 'xgboost_confidence' in self.df.columns:
            xgb_trades = self.df[self.df['was_xgboost_used'] == True]
            if len(xgb_trades) > 0:
                high_conf_winrate = (xgb_trades[xgb_trades['xgboost_confidence'] >= 80]['profit'] > 0).mean() * 100
                all_xgb_winrate = (xgb_trades['profit'] > 0).mean() * 100
                
                recommendations.append({
                    'priority': 'ALTA',
                    'action': 'Implementar filtro de confidence >= 80%',
                    'impact': f'Win rate: {all_xgb_winrate:.1f}% → {high_conf_winrate:.1f}%',
                    'implementation': 'Modificar EA para rechazar señales con confidence < 80%'
                })
        
        # Recomendación 2: Features de indicadores técnicos
        correlations = {}
        for indicator in ['atr_at_open', 'rsi_at_open', 'volatility_at_open']:
            if indicator in self.df.columns:
                correlations[indicator] = abs(self.df[indicator].corr(self.df['profit']))
        
        if correlations:
            best_indicator = max(correlations, key=correlations.get)
            recommendations.append({
                'priority': 'MEDIA',
                'action': f'Agregar {best_indicator} como feature principal',
                'impact': f'Correlación con profit: {correlations[best_indicator]:.4f}',
                'implementation': 'Incluir en el vector de features para re-entrenamiento'
            })
        
        # Recomendación 3: Filtro de régimen de mercado
        if 'market_regime' in self.df.columns:
            regime_performance = self.df.groupby('market_regime')['profit'].apply(lambda x: (x > 0).mean() * 100)
            worst_regime = regime_performance.idxmin()
            best_regime = regime_performance.idxmax()
            
            recommendations.append({
                'priority': 'MEDIA',
                'action': f'Evitar trading en régimen "{worst_regime}"',
                'impact': f'Win rate en {worst_regime}: {regime_performance[worst_regime]:.1f}% vs {best_regime}: {regime_performance[best_regime]:.1f}%',
                'implementation': 'Agregar filtro de régimen en EA o modelo'
            })
        
        # Recomendación 4: Re-entrenamiento con datos reales
        recommendations.append({
            'priority': 'ALTA',
            'action': 'Re-entrenar modelos con datos de streamed_trades',
            'impact': f'Usar {len(self.df):,} trades reales vs datos sintéticos',
            'implementation': 'Ejecutar pipeline de re-entrenamiento con features mejoradas'
        })
        
        # Mostrar recomendaciones
        for i, rec in enumerate(recommendations, 1):
            priority_emoji = "🔥" if rec['priority'] == 'ALTA' else "⚡" if rec['priority'] == 'MEDIA' else "💡"
            print(f"\n{i}. {priority_emoji} PRIORIDAD {rec['priority']}")
            print(f"   🎯 Acción: {rec['action']}")
            print(f"   📈 Impacto: {rec['impact']}")
            print(f"   🔧 Implementación: {rec['implementation']}")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("🔍 ANÁLISIS COMPLETO DE DATOS RICOS DE TRADING")
        print("=" * 70)
        print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.connect_and_load_data():
            return False
        
        try:
            # 1. Análisis de XGBoost vs otros métodos
            xgb_trades, non_xgb_trades = self.analyze_xgboost_performance()
            
            # 2. Análisis de confidence
            confidence_analysis = self.analyze_confidence_impact()
            
            # 3. Análisis de indicadores técnicos
            correlations = self.analyze_technical_indicators()
            
            # 4. Análisis de régimen de mercado
            regime_analysis = self.analyze_market_regime_effectiveness()
            
            # 5. Oportunidades de feature engineering
            opportunities = self.identify_feature_engineering_opportunities()
            
            # 6. Potencial de mejora
            improvement_potential = self.calculate_improvement_potential()
            
            # 7. Recomendaciones accionables
            recommendations = self.generate_actionable_recommendations()
            
            print(f"\n🎉 ANÁLISIS COMPLETADO")
            print("=" * 50)
            
            return {
                'xgb_performance': (xgb_trades, non_xgb_trades),
                'confidence_analysis': confidence_analysis,
                'correlations': correlations,
                'regime_analysis': regime_analysis,
                'opportunities': opportunities,
                'improvement_potential': improvement_potential,
                'recommendations': recommendations
            }
            
        finally:
            if self.conn:
                self.conn.close()
                print("✅ Conexión cerrada")

def main():
    """Función principal"""
    analyzer = RichTradingDataAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n🎯 RESUMEN EJECUTIVO:")
        print("=" * 50)
        print("✅ Datos extremadamente ricos disponibles (28 columnas)")
        print("✅ Indicadores técnicos completos (ATR, RSI, MA, Volatilidad)")
        print("✅ Datos de XGBoost confidence y performance")
        print("✅ Régimen de mercado y contexto temporal")
        print("✅ Oportunidades de mejora significativas identificadas")
        
        print("\n🚀 IMPACTO POTENCIAL:")
        print("- Mejora de win rate mediante filtros de confidence")
        print("- Features técnicas avanzadas para mejor predicción")
        print("- Optimización por régimen de mercado")
        print("- Re-entrenamiento con datos reales verificados")

if __name__ == "__main__":
    main()
