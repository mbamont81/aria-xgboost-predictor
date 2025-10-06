#!/usr/bin/env python3
"""
Sistema de entrenamiento continuo para mejorar las predicciones XGBoost
usando datos reales de streamed_trades
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib
import os
import json

DATABASE_URL = "postgresql://aria_audit_db_user:RaXTVt9ddTIRwiDcd3prxjDbf7ga1Ant@dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com/aria_audit_db"

class ContinuousLearningSystem:
    def __init__(self):
        self.conn = None
        self.current_models = {}
        self.performance_history = {}
        
    def connect_to_database(self):
        """Conectar a PostgreSQL"""
        try:
            self.conn = psycopg2.connect(DATABASE_URL)
            print("✅ Conectado a PostgreSQL")
            return True
        except Exception as e:
            print(f"❌ Error conectando: {e}")
            return False
    
    def extract_recent_trades_for_learning(self, days_back=7):
        """Extraer trades recientes para aprendizaje continuo"""
        print(f"\n📊 EXTRAYENDO TRADES PARA APRENDIZAJE CONTINUO")
        print("=" * 60)
        
        cursor = self.conn.cursor()
        
        # Obtener trades recientes con resultados conocidos
        query = """
        SELECT 
            symbol, trade_type, entry_price, exit_price, profit,
            open_time, close_time, 
            atr_at_open, rsi_at_open, ma50_at_open, ma200_at_open,
            spread_at_open, volatility_at_open,
            was_xgboost_used, xgboost_confidence, market_regime,
            stream_timestamp
        FROM streamed_trades 
        WHERE stream_timestamp >= %s
        AND profit IS NOT NULL 
        AND was_xgboost_used = true  -- Solo trades donde se usó XGBoost
        AND xgboost_confidence IS NOT NULL
        ORDER BY stream_timestamp DESC
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cursor.execute(query, (cutoff_date,))
        
        trades = cursor.fetchall()
        
        if len(trades) < 20:
            print(f"⚠️  Solo {len(trades)} trades XGBoost recientes (mínimo: 20)")
            return None
        
        columns = [
            'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit',
            'open_time', 'close_time', 'atr_at_open', 'rsi_at_open', 
            'ma50_at_open', 'ma200_at_open', 'spread_at_open', 'volatility_at_open',
            'was_xgboost_used', 'xgboost_confidence', 'market_regime', 'stream_timestamp'
        ]
        
        df = pd.DataFrame(trades, columns=columns)
        
        print(f"✅ Extraídos {len(df)} trades XGBoost de los últimos {days_back} días")
        print(f"📅 Rango: {df['stream_timestamp'].min()} → {df['stream_timestamp'].max()}")
        print(f"📊 Símbolos: {df['symbol'].nunique()} únicos")
        print(f"🎯 Win rate actual: {(df['profit'] > 0).mean()*100:.1f}%")
        
        cursor.close()
        return df
    
    def analyze_prediction_accuracy(self, df):
        """Analizar precisión de las predicciones actuales"""
        print(f"\n🔍 ANÁLISIS DE PRECISIÓN DE PREDICCIONES")
        print("=" * 50)
        
        # Crear features que el modelo actual usa
        df['price_change_actual'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
        df['was_profitable'] = (df['profit'] > 0).astype(int)
        df['profit_pips'] = df['profit'] / 0.01  # Aproximación para pips
        
        # Análisis por nivel de confidence
        confidence_bins = pd.cut(df['xgboost_confidence'], 
                               bins=[0, 70, 80, 90, 100], 
                               labels=['<70%', '70-80%', '80-90%', '90%+'])
        
        accuracy_by_confidence = df.groupby(confidence_bins).agg({
            'was_profitable': ['count', 'mean'],
            'profit': 'mean',
            'xgboost_confidence': 'mean'
        }).round(4)
        
        print("📊 PRECISIÓN POR NIVEL DE CONFIDENCE:")
        print(accuracy_by_confidence)
        
        # Análisis por símbolo
        symbol_accuracy = df.groupby('symbol').agg({
            'was_profitable': ['count', 'mean'],
            'profit': 'mean',
            'xgboost_confidence': 'mean'
        }).round(4)
        
        symbol_accuracy.columns = ['trades', 'win_rate', 'avg_profit', 'avg_confidence']
        symbol_accuracy = symbol_accuracy.sort_values('win_rate', ascending=False)
        
        print(f"\n📊 PRECISIÓN POR SÍMBOLO (Top 10):")
        print(symbol_accuracy.head(10))
        
        # Identificar patrones de error
        print(f"\n🔍 PATRONES DE ERROR IDENTIFICADOS:")
        
        # Trades perdedores con alta confidence
        high_conf_losses = df[(df['xgboost_confidence'] >= 80) & (df['profit'] <= 0)]
        print(f"   • Trades perdedores con alta confidence: {len(high_conf_losses)}")
        
        if len(high_conf_losses) > 0:
            print(f"   • Confidence promedio en pérdidas: {high_conf_losses['xgboost_confidence'].mean():.1f}%")
            print(f"   • Símbolos más problemáticos:")
            problem_symbols = high_conf_losses['symbol'].value_counts().head(5)
            for symbol, count in problem_symbols.items():
                print(f"     - {symbol}: {count} trades perdedores")
        
        return {
            'accuracy_by_confidence': accuracy_by_confidence,
            'symbol_accuracy': symbol_accuracy,
            'high_conf_losses': high_conf_losses
        }
    
    def create_improved_features(self, df):
        """Crear features mejoradas basadas en análisis de errores"""
        print(f"\n🔧 CREANDO FEATURES MEJORADAS")
        print("=" * 50)
        
        # Features temporales
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['hour'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['is_london_session'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
        df['is_ny_session'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
        
        # Features de precio mejoradas
        df['ma_trend'] = np.where(
            df['ma200_at_open'] > 0,
            (df['ma50_at_open'] - df['ma200_at_open']) / df['ma200_at_open'],
            0
        )
        df['price_vs_ma50'] = np.where(
            df['ma50_at_open'] > 0,
            (df['entry_price'] - df['ma50_at_open']) / df['ma50_at_open'],
            0
        )
        
        # Features de volatilidad
        df['volatility_normalized'] = df.groupby('symbol')['volatility_at_open'].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        df['atr_normalized'] = df['atr_at_open'] / df['entry_price']
        
        # Features de contexto histórico
        df = df.sort_values(['symbol', 'stream_timestamp'])
        
        # Performance histórica por símbolo (rolling)
        df['symbol_recent_winrate'] = df.groupby('symbol')['was_profitable'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        
        # Volatilidad reciente por símbolo
        df['symbol_recent_volatility'] = df.groupby('symbol')['volatility_at_open'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        
        # Features de régimen de mercado
        regime_map = {'trending': 1, 'volatile': 2, 'ranging': 0}
        df['market_regime_encoded'] = df['market_regime'].map(regime_map).fillna(0)
        
        print(f"✅ Features mejoradas creadas")
        print(f"📊 Total columnas: {df.shape[1]}")
        
        return df
    
    def train_improved_models_by_symbol(self, df):
        """Entrenar modelos mejorados por símbolo usando datos reales"""
        print(f"\n🤖 ENTRENANDO MODELOS MEJORADOS POR SÍMBOLO")
        print("=" * 60)
        
        # Features para entrenamiento
        feature_columns = [
            'trade_type', 'hour', 'day_of_week', 'is_london_session', 'is_ny_session',
            'ma_trend', 'price_vs_ma50', 'volatility_normalized', 'atr_normalized',
            'market_regime_encoded', 'symbol_recent_winrate', 'symbol_recent_volatility'
        ]
        
        improved_models = {}
        
        # Entrenar por símbolo (donde hay suficientes datos)
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 30:  # Mínimo 30 trades por símbolo
                print(f"⚠️  {symbol}: Solo {len(symbol_data)} trades, saltando")
                continue
            
            print(f"\n🎯 Entrenando {symbol} ({len(symbol_data)} trades)")
            
            # Preparar datos
            X = symbol_data[feature_columns].fillna(0)
            y_profitable = symbol_data['was_profitable']
            y_profit_amount = symbol_data['profit']
            
            # Limpiar datos problemáticos
            X = X.replace([np.inf, -np.inf], 0)
            
            try:
                # Modelo 1: Clasificación de rentabilidad
                if len(y_profitable.unique()) > 1:  # Necesita variabilidad
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_profitable, test_size=0.3, random_state=42
                    )
                    
                    model_profitable = xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        random_state=42
                    )
                    
                    model_profitable.fit(X_train, y_train)
                    
                    # Evaluar
                    y_pred = model_profitable.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    current_winrate = y_test.mean() * 100
                    predicted_winrate = y_pred.mean() * 100
                    
                    # Feature importance
                    feature_importance = dict(zip(feature_columns, model_profitable.feature_importances_))
                    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
                    
                    improved_models[f'{symbol}_profitable'] = {
                        'model': model_profitable,
                        'type': 'classification',
                        'accuracy': accuracy,
                        'current_winrate': current_winrate,
                        'predicted_winrate': predicted_winrate,
                        'improvement': predicted_winrate - current_winrate,
                        'top_features': top_features,
                        'samples': len(X_train)
                    }
                    
                    print(f"   ✅ Clasificación: Accuracy {accuracy:.4f}")
                    print(f"   📊 Win rate actual: {current_winrate:.1f}%")
                    print(f"   🎯 Win rate predicho: {predicted_winrate:.1f}%")
                    print(f"   📈 Mejora: {predicted_winrate - current_winrate:+.1f}%")
                    print(f"   🔝 Top features: {[f[0] for f in top_features]}")
                
                # Modelo 2: Regresión de profit (cuánto profit)
                X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
                    X, y_profit_amount, test_size=0.3, random_state=42
                )
                
                model_profit = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model_profit.fit(X_train_r, y_train_r)
                
                # Evaluar
                y_pred_r = model_profit.predict(X_test_r)
                mse = mean_squared_error(y_test_r, y_pred_r)
                
                improved_models[f'{symbol}_profit'] = {
                    'model': model_profit,
                    'type': 'regression',
                    'mse': mse,
                    'samples': len(X_train_r)
                }
                
                print(f"   ✅ Regresión: MSE {mse:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error entrenando {symbol}: {e}")
        
        print(f"\n✅ Modelos mejorados entrenados: {len(improved_models)}")
        return improved_models
    
    def create_feedback_loop_system(self, df):
        """Crear sistema de feedback loop para mejora continua"""
        print(f"\n🔄 CREANDO SISTEMA DE FEEDBACK LOOP")
        print("=" * 50)
        
        # Analizar errores de predicción
        df['prediction_error'] = abs(df['xgboost_confidence'] - (df['profit'] > 0).astype(int) * 100)
        
        # Identificar patrones en errores
        error_analysis = df.groupby(['symbol', 'market_regime']).agg({
            'prediction_error': 'mean',
            'was_profitable': 'count',
            'profit': 'mean'
        }).round(4)
        
        error_analysis.columns = ['avg_prediction_error', 'trades', 'avg_profit']
        error_analysis = error_analysis.sort_values('avg_prediction_error', ascending=False)
        
        print("📊 ANÁLISIS DE ERRORES DE PREDICCIÓN:")
        print("(Símbolos/regímenes con mayor error de predicción)")
        print(error_analysis.head(10))
        
        # Crear dataset de corrección
        correction_features = [
            'symbol', 'market_regime', 'hour', 'volatility_at_open',
            'atr_at_open', 'ma_trend', 'xgboost_confidence'
        ]
        
        # Target: diferencia entre confidence predicha y resultado real
        df['confidence_calibration_error'] = df['xgboost_confidence'] - (df['profit'] > 0).astype(int) * 100
        
        return {
            'error_analysis': error_analysis,
            'correction_data': df[correction_features + ['confidence_calibration_error']].copy()
        }
    
    def generate_improved_prediction_system(self, improved_models, feedback_data):
        """Generar sistema de predicción mejorado"""
        print(f"\n🚀 GENERANDO SISTEMA DE PREDICCIÓN MEJORADO")
        print("=" * 60)
        
        # Crear configuración del nuevo sistema
        improved_system = {
            'version': '2.0_CONTINUOUS_LEARNING',
            'created_date': datetime.now().isoformat(),
            'models': {},
            'feedback_system': feedback_data,
            'improvements': []
        }
        
        # Procesar modelos mejorados
        for model_name, model_info in improved_models.items():
            symbol = model_name.split('_')[0]
            model_type = model_name.split('_')[1]
            
            if model_type == 'profitable':
                current_winrate = model_info['current_winrate']
                predicted_winrate = model_info['predicted_winrate']
                improvement = model_info['improvement']
                
                if improvement > 2.0:  # Mejora significativa
                    improved_system['models'][model_name] = {
                        'symbol': symbol,
                        'type': 'classification',
                        'current_performance': current_winrate,
                        'expected_performance': predicted_winrate,
                        'improvement': improvement,
                        'top_features': model_info['top_features'],
                        'ready_for_deployment': True
                    }
                    
                    improved_system['improvements'].append({
                        'symbol': symbol,
                        'improvement_type': 'win_rate',
                        'current': current_winrate,
                        'expected': predicted_winrate,
                        'gain': improvement
                    })
                    
                    print(f"✅ {symbol}: Win rate {current_winrate:.1f}% → {predicted_winrate:.1f}% (+{improvement:.1f}%)")
        
        # Calcular impacto total
        total_improvement = sum(imp['gain'] for imp in improved_system['improvements'])
        avg_improvement = total_improvement / len(improved_system['improvements']) if improved_system['improvements'] else 0
        
        print(f"\n📈 IMPACTO TOTAL DEL SISTEMA MEJORADO:")
        print(f"   • Símbolos mejorados: {len(improved_system['improvements'])}")
        print(f"   • Mejora promedio: +{avg_improvement:.1f}% win rate")
        print(f"   • Modelos listos: {len(improved_system['models'])}")
        
        return improved_system
    
    def save_improved_system(self, improved_system, output_dir="continuous_learning_models"):
        """Guardar sistema mejorado"""
        print(f"\n💾 GUARDANDO SISTEMA DE APRENDIZAJE CONTINUO")
        print("=" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar configuración del sistema
        config_path = os.path.join(output_dir, "continuous_learning_config.json")
        with open(config_path, 'w') as f:
            # Remover objetos no serializables
            config_to_save = improved_system.copy()
            config_to_save['models'] = {k: {**v, 'model': 'saved_separately'} 
                                       for k, v in config_to_save['models'].items()}
            json.dump(config_to_save, f, indent=2)
        
        print(f"   ✅ continuous_learning_config.json")
        
        # Guardar modelos individuales
        saved_models = 0
        for model_name, model_info in improved_system['models'].items():
            if 'model' in model_info:
                model_path = os.path.join(output_dir, f"{model_name}_improved.pkl")
                joblib.dump(model_info, model_path)
                saved_models += 1
                print(f"   ✅ {model_name}_improved.pkl")
        
        print(f"\n✅ Sistema guardado en {output_dir}/")
        print(f"📊 Modelos mejorados: {saved_models}")
        
        return output_dir
    
    def run_continuous_learning_pipeline(self):
        """Ejecutar pipeline completo de aprendizaje continuo"""
        print("🔄 PIPELINE DE APRENDIZAJE CONTINUO XGBOOST")
        print("=" * 70)
        print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Objetivo: Mejorar predicciones usando resultados reales")
        
        if not self.connect_to_database():
            return False
        
        try:
            # 1. Extraer trades recientes
            df = self.extract_recent_trades_for_learning(days_back=14)  # 2 semanas
            if df is None:
                return False
            
            # 2. Analizar precisión actual
            analysis = self.analyze_prediction_accuracy(df)
            
            # 3. Crear features mejoradas
            df_improved = self.create_improved_features(df)
            
            # 4. Entrenar modelos mejorados
            improved_models = self.train_improved_models_by_symbol(df_improved)
            
            if not improved_models:
                print("❌ No se crearon modelos mejorados")
                return False
            
            # 5. Crear sistema de feedback
            feedback_data = self.create_feedback_loop_system(df_improved)
            
            # 6. Generar sistema mejorado
            improved_system = self.generate_improved_prediction_system(improved_models, feedback_data)
            
            # 7. Guardar sistema
            output_dir = self.save_improved_system(improved_system)
            
            print(f"\n🎉 APRENDIZAJE CONTINUO COMPLETADO")
            print("=" * 50)
            print(f"📁 Sistema guardado en: {output_dir}")
            print(f"🤖 Modelos mejorados: {len(improved_models)}")
            print(f"📈 Mejoras identificadas: {len(improved_system['improvements'])}")
            
            return True
            
        finally:
            if self.conn:
                self.conn.close()
                print("✅ Conexión cerrada")

def main():
    """Función principal"""
    learner = ContinuousLearningSystem()
    
    print("🎯 INICIANDO SISTEMA DE APRENDIZAJE CONTINUO")
    print("Este sistema va a:")
    print("• Analizar errores de predicciones actuales")
    print("• Identificar patrones en fallos")
    print("• Re-entrenar modelos con datos reales")
    print("• Crear sistema de feedback loop")
    print("• Mejorar predicciones continuamente")
    
    success = learner.run_continuous_learning_pipeline()
    
    if success:
        print("\n💡 PRÓXIMOS PASOS:")
        print("1. ✅ Modelos mejorados creados con datos reales")
        print("2. 🔄 Implementar en servicio Render")
        print("3. 📊 Configurar actualización automática semanal")
        print("4. 🎯 Monitorear mejoras en predicciones")
        print("5. 🔄 Feedback loop automático activo")
    else:
        print("\n❌ Aprendizaje continuo falló")

if __name__ == "__main__":
    main()
