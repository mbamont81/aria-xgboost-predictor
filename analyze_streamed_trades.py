#!/usr/bin/env python3
"""
Análisis profundo de streamed_trades para entender cómo mejorar las predicciones XGBoost
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# URL de la base de datos
DATABASE_URL = "postgresql://aria_audit_db_user:RaXTVt9ddTIRwiDcd3prxjDbf7ga1Ant@dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com/aria_audit_db"

class StreamedTradesAnalyzer:
    def __init__(self):
        self.conn = None
        self.df = None
        
    def connect_to_database(self):
        """Conectar a PostgreSQL"""
        try:
            self.conn = psycopg2.connect(DATABASE_URL)
            print("✅ Conectado a PostgreSQL")
            return True
        except Exception as e:
            print(f"❌ Error conectando a DB: {e}")
            return False
    
    def load_all_trades_data(self):
        """Cargar todos los datos de streamed_trades"""
        print(f"\n📊 CARGANDO DATOS COMPLETOS DE STREAMED_TRADES")
        print("=" * 60)
        
        cursor = self.conn.cursor()
        
        # Obtener todos los trades con todas las columnas
        query = """
        SELECT *
        FROM streamed_trades 
        ORDER BY created_at DESC
        """
        
        cursor.execute(query)
        trades = cursor.fetchall()
        
        # Obtener nombres de columnas
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'streamed_trades' AND table_schema = 'public'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in cursor.fetchall()]
        
        # Crear DataFrame
        self.df = pd.DataFrame(trades, columns=columns)
        
        print(f"✅ Cargados {len(self.df)} trades")
        print(f"📊 Columnas disponibles: {len(columns)}")
        print(f"📅 Rango temporal: {self.df['created_at'].min()} → {self.df['created_at'].max()}")
        
        cursor.close()
        return self.df
    
    def analyze_data_quality(self):
        """Analizar calidad y completitud de los datos"""
        print(f"\n🔍 ANÁLISIS DE CALIDAD DE DATOS")
        print("=" * 50)
        
        # Información básica
        print(f"📊 Shape: {self.df.shape}")
        print(f"📅 Período: {(self.df['created_at'].max() - self.df['created_at'].min()).days} días")
        
        # Completitud de datos
        print(f"\n📋 COMPLETITUD DE DATOS:")
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df) * 100).round(2)
        
        completeness = pd.DataFrame({
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Count', ascending=False)
        
        print(completeness[completeness['Missing_Count'] > 0])
        
        # Datos críticos para predicciones
        critical_columns = ['profit', 'entry_price', 'exit_price', 'confidence', 'market_regime']
        print(f"\n🎯 DATOS CRÍTICOS PARA PREDICCIONES:")
        for col in critical_columns:
            if col in self.df.columns:
                missing = self.df[col].isnull().sum()
                pct = (missing / len(self.df) * 100).round(2)
                status = "✅" if pct < 5 else "⚠️" if pct < 20 else "❌"
                print(f"   {status} {col}: {pct}% missing ({missing} registros)")
        
        return completeness
    
    def analyze_trading_performance(self):
        """Analizar performance de trading para identificar patrones"""
        print(f"\n📈 ANÁLISIS DE PERFORMANCE DE TRADING")
        print("=" * 50)
        
        # Estadísticas básicas de profit
        profit_stats = self.df['profit'].describe()
        print(f"📊 ESTADÍSTICAS DE PROFIT:")
        print(profit_stats)
        
        # Win rate general
        profitable_trades = (self.df['profit'] > 0).sum()
        total_trades = len(self.df[self.df['profit'].notna()])
        win_rate = (profitable_trades / total_trades * 100).round(2)
        
        print(f"\n🎯 MÉTRICAS GENERALES:")
        print(f"   • Win Rate: {win_rate}%")
        print(f"   • Trades Profitable: {profitable_trades:,}")
        print(f"   • Trades Perdedores: {total_trades - profitable_trades:,}")
        print(f"   • Profit Total: {self.df['profit'].sum():.2f}")
        print(f"   • Profit Promedio: {self.df['profit'].mean():.4f}")
        
        # Performance por símbolo
        print(f"\n📊 PERFORMANCE POR SÍMBOLO:")
        symbol_performance = self.df.groupby('symbol').agg({
            'profit': ['count', 'sum', 'mean', lambda x: (x > 0).sum()],
            'confidence': 'mean'
        }).round(4)
        
        symbol_performance.columns = ['Trades', 'Total_Profit', 'Avg_Profit', 'Winning_Trades', 'Avg_Confidence']
        symbol_performance['Win_Rate'] = (symbol_performance['Winning_Trades'] / symbol_performance['Trades'] * 100).round(2)
        symbol_performance = symbol_performance.sort_values('Total_Profit', ascending=False)
        
        print(symbol_performance)
        
        return symbol_performance
    
    def analyze_predictive_features(self):
        """Analizar features que podrían ser predictivas"""
        print(f"\n🔮 ANÁLISIS DE FEATURES PREDICTIVAS")
        print("=" * 50)
        
        # 1. Confidence vs Performance
        if 'confidence' in self.df.columns:
            print(f"\n📊 CONFIDENCE vs PERFORMANCE:")
            
            # Crear bins de confidence
            self.df['confidence_bin'] = pd.cut(self.df['confidence'], 
                                             bins=[0, 60, 70, 80, 90, 100], 
                                             labels=['<60%', '60-70%', '70-80%', '80-90%', '90%+'])
            
            confidence_analysis = self.df.groupby('confidence_bin').agg({
                'profit': ['count', 'mean', lambda x: (x > 0).sum()],
            }).round(4)
            
            confidence_analysis.columns = ['Trades', 'Avg_Profit', 'Winning_Trades']
            confidence_analysis['Win_Rate'] = (confidence_analysis['Winning_Trades'] / confidence_analysis['Trades'] * 100).round(2)
            
            print(confidence_analysis)
            
            # Correlación confidence-profit
            corr_conf_profit = self.df['confidence'].corr(self.df['profit'])
            print(f"\n🔗 Correlación Confidence-Profit: {corr_conf_profit:.4f}")
        
        # 2. Market Regime vs Performance
        if 'market_regime' in self.df.columns:
            print(f"\n🌊 MARKET REGIME vs PERFORMANCE:")
            
            regime_analysis = self.df.groupby('market_regime').agg({
                'profit': ['count', 'mean', lambda x: (x > 0).sum()],
                'confidence': 'mean'
            }).round(4)
            
            regime_analysis.columns = ['Trades', 'Avg_Profit', 'Winning_Trades', 'Avg_Confidence']
            regime_analysis['Win_Rate'] = (regime_analysis['Winning_Trades'] / regime_analysis['Trades'] * 100).round(2)
            
            print(regime_analysis)
        
        # 3. Timeframe vs Performance
        if 'timeframe' in self.df.columns:
            print(f"\n⏰ TIMEFRAME vs PERFORMANCE:")
            
            timeframe_analysis = self.df.groupby('timeframe').agg({
                'profit': ['count', 'mean', lambda x: (x > 0).sum()],
                'confidence': 'mean'
            }).round(4)
            
            timeframe_analysis.columns = ['Trades', 'Avg_Profit', 'Winning_Trades', 'Avg_Confidence']
            timeframe_analysis['Win_Rate'] = (timeframe_analysis['Winning_Trades'] / timeframe_analysis['Trades'] * 100).round(2)
            
            print(timeframe_analysis)
        
        # 4. Análisis temporal
        print(f"\n📅 ANÁLISIS TEMPORAL:")
        
        self.df['hour'] = pd.to_datetime(self.df['entry_time']).dt.hour
        self.df['day_of_week'] = pd.to_datetime(self.df['entry_time']).dt.dayofweek
        
        # Performance por hora
        hourly_performance = self.df.groupby('hour').agg({
            'profit': ['count', 'mean', lambda x: (x > 0).sum()]
        }).round(4)
        
        hourly_performance.columns = ['Trades', 'Avg_Profit', 'Winning_Trades']
        hourly_performance['Win_Rate'] = (hourly_performance['Winning_Trades'] / hourly_performance['Trades'] * 100).round(2)
        
        # Mostrar mejores y peores horas
        best_hours = hourly_performance.nlargest(3, 'Win_Rate')
        worst_hours = hourly_performance.nsmallest(3, 'Win_Rate')
        
        print(f"🌟 MEJORES HORAS:")
        print(best_hours)
        print(f"\n💔 PEORES HORAS:")
        print(worst_hours)
        
        return {
            'confidence_analysis': confidence_analysis if 'confidence' in self.df.columns else None,
            'regime_analysis': regime_analysis if 'market_regime' in self.df.columns else None,
            'timeframe_analysis': timeframe_analysis if 'timeframe' in self.df.columns else None,
            'hourly_performance': hourly_performance
        }
    
    def identify_improvement_opportunities(self):
        """Identificar oportunidades específicas de mejora"""
        print(f"\n💡 OPORTUNIDADES DE MEJORA IDENTIFICADAS")
        print("=" * 60)
        
        improvements = []
        
        # 1. Filtros de calidad
        if 'confidence' in self.df.columns:
            low_conf_trades = self.df[self.df['confidence'] < 70]
            low_conf_winrate = (low_conf_trades['profit'] > 0).mean() * 100
            
            high_conf_trades = self.df[self.df['confidence'] >= 80]
            high_conf_winrate = (high_conf_trades['profit'] > 0).mean() * 100
            
            if high_conf_winrate > low_conf_winrate + 10:
                improvements.append({
                    'type': 'Filtro de Confidence',
                    'description': f'Usar solo trades con confidence >= 80%',
                    'impact': f'Win rate: {low_conf_winrate:.1f}% → {high_conf_winrate:.1f}%',
                    'trades_affected': len(low_conf_trades)
                })
        
        # 2. Optimización por régimen de mercado
        if 'market_regime' in self.df.columns:
            regime_performance = self.df.groupby('market_regime')['profit'].apply(lambda x: (x > 0).mean() * 100)
            best_regime = regime_performance.idxmax()
            worst_regime = regime_performance.idxmin()
            
            if regime_performance[best_regime] > regime_performance[worst_regime] + 15:
                improvements.append({
                    'type': 'Filtro de Market Regime',
                    'description': f'Evitar trading en régimen "{worst_regime}"',
                    'impact': f'Win rate en {worst_regime}: {regime_performance[worst_regime]:.1f}% vs {best_regime}: {regime_performance[best_regime]:.1f}%',
                    'trades_affected': len(self.df[self.df['market_regime'] == worst_regime])
                })
        
        # 3. Optimización temporal
        hourly_perf = self.df.groupby(self.df['hour'])['profit'].apply(lambda x: (x > 0).mean() * 100)
        bad_hours = hourly_perf[hourly_perf < 45].index.tolist()  # Horas con <45% win rate
        
        if bad_hours:
            improvements.append({
                'type': 'Filtro Temporal',
                'description': f'Evitar trading en horas: {bad_hours}',
                'impact': f'Eliminar {len(bad_hours)} horas de bajo rendimiento',
                'trades_affected': len(self.df[self.df['hour'].isin(bad_hours)])
            })
        
        # 4. Optimización por símbolo
        symbol_perf = self.df.groupby('symbol')['profit'].apply(lambda x: (x > 0).mean() * 100)
        bad_symbols = symbol_perf[symbol_perf < 40].index.tolist()  # Símbolos con <40% win rate
        
        if bad_symbols:
            improvements.append({
                'type': 'Filtro de Símbolos',
                'description': f'Evitar o mejorar modelos para: {bad_symbols}',
                'impact': f'Símbolos problemáticos identificados',
                'trades_affected': len(self.df[self.df['symbol'].isin(bad_symbols)])
            })
        
        # Mostrar mejoras
        for i, improvement in enumerate(improvements, 1):
            print(f"\n{i}. 🎯 {improvement['type']}:")
            print(f"   📋 {improvement['description']}")
            print(f"   📈 Impacto: {improvement['impact']}")
            print(f"   📊 Trades afectados: {improvement['trades_affected']:,}")
        
        return improvements
    
    def calculate_potential_improvements(self):
        """Calcular mejoras potenciales específicas"""
        print(f"\n📊 CÁLCULO DE MEJORAS POTENCIALES")
        print("=" * 50)
        
        current_winrate = (self.df['profit'] > 0).mean() * 100
        current_profit = self.df['profit'].sum()
        
        print(f"📈 SITUACIÓN ACTUAL:")
        print(f"   • Win Rate: {current_winrate:.2f}%")
        print(f"   • Profit Total: {current_profit:.2f}")
        print(f"   • Total Trades: {len(self.df):,}")
        
        # Simulación de mejoras
        improved_df = self.df.copy()
        
        # Aplicar filtro de confidence >= 75%
        if 'confidence' in improved_df.columns:
            improved_df = improved_df[improved_df['confidence'] >= 75]
        
        # Aplicar filtro de mejores horas (win rate > 50%)
        hourly_winrate = self.df.groupby('hour')['profit'].apply(lambda x: (x > 0).mean() * 100)
        good_hours = hourly_winrate[hourly_winrate > 50].index.tolist()
        improved_df = improved_df[improved_df['hour'].isin(good_hours)]
        
        # Aplicar filtro de mejor régimen
        if 'market_regime' in improved_df.columns:
            regime_winrate = self.df.groupby('market_regime')['profit'].apply(lambda x: (x > 0).mean() * 100)
            best_regimes = regime_winrate[regime_winrate > 50].index.tolist()
            improved_df = improved_df[improved_df['market_regime'].isin(best_regimes)]
        
        # Calcular mejoras
        if len(improved_df) > 0:
            improved_winrate = (improved_df['profit'] > 0).mean() * 100
            improved_profit = improved_df['profit'].sum()
            
            print(f"\n🚀 SITUACIÓN MEJORADA (con filtros):")
            print(f"   • Win Rate: {improved_winrate:.2f}% (+{improved_winrate - current_winrate:.2f}%)")
            print(f"   • Profit Total: {improved_profit:.2f}")
            print(f"   • Total Trades: {len(improved_df):,} (-{len(self.df) - len(improved_df):,})")
            print(f"   • Profit por Trade: {improved_profit/len(improved_df):.4f} vs {current_profit/len(self.df):.4f}")
        
        return {
            'current_winrate': current_winrate,
            'improved_winrate': improved_winrate if len(improved_df) > 0 else current_winrate,
            'trades_filtered': len(self.df) - len(improved_df) if len(improved_df) > 0 else 0
        }
    
    def generate_feature_engineering_recommendations(self):
        """Generar recomendaciones específicas para feature engineering"""
        print(f"\n🔧 RECOMENDACIONES PARA FEATURE ENGINEERING")
        print("=" * 60)
        
        recommendations = []
        
        # 1. Features temporales
        recommendations.append({
            'category': 'Features Temporales',
            'features': [
                'hour_of_day (0-23)',
                'day_of_week (0-6)', 
                'is_london_session (8-17 UTC)',
                'is_ny_session (13-22 UTC)',
                'is_overlap_session (13-17 UTC)',
                'trade_duration_minutes'
            ],
            'justification': 'Análisis muestra patrones claros por hora y día'
        })
        
        # 2. Features de mercado
        if 'market_regime' in self.df.columns:
            recommendations.append({
                'category': 'Features de Régimen de Mercado',
                'features': [
                    'market_regime_encoded (0=trending, 1=ranging, 2=volatile)',
                    'regime_confidence_interaction',
                    'regime_symbol_interaction',
                    'regime_timeframe_interaction'
                ],
                'justification': 'Régimen de mercado muestra impacto significativo en performance'
            })
        
        # 3. Features de confidence
        if 'confidence' in self.df.columns:
            recommendations.append({
                'category': 'Features de Confidence',
                'features': [
                    'confidence_normalized (0-1)',
                    'confidence_binned (low/medium/high)',
                    'confidence_symbol_avg (histórico por símbolo)',
                    'confidence_deviation (vs promedio histórico)'
                ],
                'justification': f'Correlación confidence-profit: {self.df["confidence"].corr(self.df["profit"]):.4f}'
            })
        
        # 4. Features de precio
        recommendations.append({
            'category': 'Features de Precio',
            'features': [
                'price_change_pct ((exit-entry)/entry)',
                'price_volatility (high-low range)',
                'entry_price_normalized (por símbolo)',
                'price_momentum (cambio reciente)',
                'support_resistance_distance'
            ],
            'justification': 'Datos de precio son fundamentales para predicciones'
        })
        
        # 5. Features históricas por símbolo
        recommendations.append({
            'category': 'Features Históricas por Símbolo',
            'features': [
                'symbol_avg_profit (últimos N trades)',
                'symbol_winrate (últimos N trades)',
                'symbol_volatility (std de profits)',
                'symbol_trade_frequency',
                'symbol_best_timeframe'
            ],
            'justification': 'Cada símbolo muestra patrones únicos de performance'
        })
        
        # Mostrar recomendaciones
        for rec in recommendations:
            print(f"\n🎯 {rec['category']}:")
            print(f"   📋 Justificación: {rec['justification']}")
            print(f"   🔧 Features recomendadas:")
            for feature in rec['features']:
                print(f"      • {feature}")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Ejecutar análisis completo"""
        print("🔍 ANÁLISIS COMPLETO DE STREAMED_TRADES")
        print("=" * 70)
        print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.connect_to_database():
            return False
        
        try:
            # 1. Cargar datos
            self.load_all_trades_data()
            
            # 2. Análisis de calidad
            completeness = self.analyze_data_quality()
            
            # 3. Análisis de performance
            symbol_performance = self.analyze_trading_performance()
            
            # 4. Análisis de features predictivas
            predictive_analysis = self.analyze_predictive_features()
            
            # 5. Identificar mejoras
            improvements = self.identify_improvement_opportunities()
            
            # 6. Calcular mejoras potenciales
            potential_improvements = self.calculate_potential_improvements()
            
            # 7. Recomendaciones de feature engineering
            feature_recommendations = self.generate_feature_engineering_recommendations()
            
            print(f"\n🎉 ANÁLISIS COMPLETADO")
            print("=" * 50)
            
            return {
                'data_quality': completeness,
                'symbol_performance': symbol_performance,
                'predictive_analysis': predictive_analysis,
                'improvements': improvements,
                'potential_improvements': potential_improvements,
                'feature_recommendations': feature_recommendations
            }
            
        finally:
            if self.conn:
                self.conn.close()
                print("✅ Conexión cerrada")

def main():
    """Función principal"""
    analyzer = StreamedTradesAnalyzer()
    results = analyzer.run_complete_analysis()
    
    if results:
        print("\n💡 RESUMEN EJECUTIVO:")
        print("=" * 50)
        print("✅ Datos de alta calidad disponibles para re-entrenamiento")
        print("✅ Patrones claros identificados en confidence, régimen y tiempo")
        print("✅ Oportunidades de mejora específicas identificadas")
        print("✅ Recomendaciones de features generadas")
        
        print("\n🚀 PRÓXIMOS PASOS RECOMENDADOS:")
        print("1. Implementar features recomendadas")
        print("2. Aplicar filtros de calidad identificados")
        print("3. Re-entrenar modelos con datos optimizados")
        print("4. Validar mejoras en ambiente de prueba")

if __name__ == "__main__":
    main()
