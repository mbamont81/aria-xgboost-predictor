#!/usr/bin/env python3
"""
Sistema de re-entrenamiento de XGBoost usando datos de streamed_trades
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

# URL de la base de datos
DATABASE_URL = "postgresql://aria_audit_db_user:RaXTVt9ddTIRwiDcd3prxjDbf7ga1Ant@dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com/aria_audit_db"

class XGBoostRetrainer:
    def __init__(self):
        self.conn = None
        self.models = {}
        
    def connect_to_database(self):
        """Conectar a PostgreSQL"""
        try:
            self.conn = psycopg2.connect(DATABASE_URL)
            print("✅ Conectado a PostgreSQL")
            return True
        except Exception as e:
            print(f"❌ Error conectando a DB: {e}")
            return False
    
    def extract_trades_data(self, days_back=30, min_trades=50):
        """Extraer datos de trades para entrenamiento"""
        print(f"\n📊 EXTRAYENDO DATOS DE TRADES")
        print("=" * 50)
        
        cursor = self.conn.cursor()
        
        # Obtener trades recientes
        query = """
        SELECT 
            symbol,
            trade_type,
            entry_price,
            exit_price,
            profit,
            entry_time,
            exit_time,
            confidence,
            sl_pips,
            tp_pips,
            volume,
            market_regime,
            timeframe,
            created_at
        FROM streamed_trades 
        WHERE created_at >= %s
        AND profit IS NOT NULL
        AND entry_price > 0
        AND exit_price > 0
        ORDER BY created_at DESC
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cursor.execute(query, (cutoff_date,))
        
        trades = cursor.fetchall()
        
        if len(trades) < min_trades:
            print(f"⚠️  Solo {len(trades)} trades encontrados (mínimo: {min_trades})")
            return None
        
        # Convertir a DataFrame
        columns = [
            'symbol', 'trade_type', 'entry_price', 'exit_price', 'profit',
            'entry_time', 'exit_time', 'confidence', 'sl_pips', 'tp_pips',
            'volume', 'market_regime', 'timeframe', 'created_at'
        ]
        
        df = pd.DataFrame(trades, columns=columns)
        
        print(f"✅ Extraídos {len(df)} trades de los últimos {days_back} días")
        print(f"📅 Rango: {df['created_at'].min()} → {df['created_at'].max()}")
        print(f"📊 Símbolos únicos: {df['symbol'].nunique()}")
        print(f"🎯 Símbolos: {list(df['symbol'].unique())}")
        
        cursor.close()
        return df
    
    def engineer_features(self, df):
        """Crear features para entrenamiento"""
        print(f"\n🔧 INGENIERÍA DE FEATURES")
        print("=" * 50)
        
        # Crear copia para no modificar original
        features_df = df.copy()
        
        # 1. Features básicas de precio
        features_df['price_change'] = (features_df['exit_price'] - features_df['entry_price']) / features_df['entry_price']
        features_df['price_range'] = abs(features_df['exit_price'] - features_df['entry_price'])
        
        # 2. Features temporales
        features_df['entry_hour'] = pd.to_datetime(features_df['entry_time']).dt.hour
        features_df['entry_day_of_week'] = pd.to_datetime(features_df['entry_time']).dt.dayofweek
        features_df['trade_duration_minutes'] = (
            pd.to_datetime(features_df['exit_time']) - pd.to_datetime(features_df['entry_time'])
        ).dt.total_seconds() / 60
        
        # 3. Features de resultado
        features_df['is_profitable'] = (features_df['profit'] > 0).astype(int)
        features_df['profit_pips'] = features_df['profit'] / features_df['volume']  # Aproximación
        features_df['risk_reward'] = np.where(
            features_df['sl_pips'] > 0,
            features_df['tp_pips'] / features_df['sl_pips'],
            0
        )
        
        # 4. Encoding categóricas
        features_df['trade_type_encoded'] = features_df['trade_type'].astype(int)
        
        # Market regime encoding
        regime_mapping = {'trending': 0, 'ranging': 1, 'volatile': 2}
        features_df['market_regime_encoded'] = features_df['market_regime'].map(regime_mapping).fillna(0)
        
        # Timeframe encoding
        timeframe_mapping = {'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30, 'H1': 60, 'H4': 240, 'D1': 1440}
        features_df['timeframe_minutes'] = features_df['timeframe'].map(timeframe_mapping).fillna(60)
        
        # 5. Features por símbolo (estadísticas históricas)
        symbol_stats = features_df.groupby('symbol').agg({
            'profit': ['mean', 'std', 'count'],
            'confidence': 'mean',
            'trade_duration_minutes': 'mean'
        }).round(4)
        
        symbol_stats.columns = ['_'.join(col).strip() for col in symbol_stats.columns]
        symbol_stats = symbol_stats.add_prefix('symbol_')
        
        # Merge symbol stats
        features_df = features_df.merge(
            symbol_stats.reset_index(), 
            on='symbol', 
            how='left'
        )
        
        print(f"✅ Features creadas: {features_df.shape[1]} columnas")
        print(f"📊 Samples: {len(features_df)}")
        
        return features_df
    
    def prepare_training_data(self, features_df):
        """Preparar datos para entrenamiento"""
        print(f"\n📋 PREPARANDO DATOS DE ENTRENAMIENTO")
        print("=" * 50)
        
        # Seleccionar features numéricas para entrenamiento
        feature_columns = [
            'trade_type_encoded', 'entry_price', 'confidence', 'sl_pips', 'tp_pips',
            'volume', 'market_regime_encoded', 'timeframe_minutes',
            'price_change', 'price_range', 'entry_hour', 'entry_day_of_week',
            'trade_duration_minutes', 'risk_reward',
            'symbol_profit_mean', 'symbol_profit_std', 'symbol_profit_count',
            'symbol_confidence_mean', 'symbol_trade_duration_minutes_mean'
        ]
        
        # Filtrar columnas que existen
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        X = features_df[available_features].fillna(0)
        
        # Targets múltiples
        targets = {
            'profit_classification': (features_df['profit'] > 0).astype(int),  # Clasificación: profitable o no
            'profit_regression': features_df['profit'],  # Regresión: cantidad de profit
            'sl_prediction': features_df['sl_pips'],  # Predicción de SL óptimo
            'tp_prediction': features_df['tp_pips']   # Predicción de TP óptimo
        }
        
        print(f"✅ Features seleccionadas: {len(available_features)}")
        print(f"📊 Features: {available_features}")
        print(f"🎯 Targets: {list(targets.keys())}")
        
        return X, targets, available_features
    
    def train_models_by_symbol(self, X, targets, features_df, feature_names):
        """Entrenar modelos específicos por símbolo"""
        print(f"\n🤖 ENTRENANDO MODELOS POR SÍMBOLO")
        print("=" * 50)
        
        symbols = features_df['symbol'].unique()
        trained_models = {}
        
        for symbol in symbols:
            symbol_mask = features_df['symbol'] == symbol
            symbol_count = symbol_mask.sum()
            
            if symbol_count < 20:  # Mínimo 20 trades por símbolo
                print(f"⚠️  {symbol}: Solo {symbol_count} trades, saltando")
                continue
            
            print(f"\n🎯 Entrenando {symbol} ({symbol_count} trades)")
            
            X_symbol = X[symbol_mask]
            trained_models[symbol] = {}
            
            for target_name, y in targets.items():
                y_symbol = y[symbol_mask]
                
                if len(y_symbol.unique()) < 2:  # Necesita variabilidad
                    print(f"   ⚠️  {target_name}: Sin variabilidad, saltando")
                    continue
                
                try:
                    # Split train/test
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_symbol, y_symbol, test_size=0.2, random_state=42
                    )
                    
                    # Configurar modelo según tipo de target
                    if 'classification' in target_name:
                        model = xgb.XGBClassifier(
                            n_estimators=100,
                            max_depth=4,
                            learning_rate=0.1,
                            random_state=42
                        )
                        metric_func = accuracy_score
                    else:
                        model = xgb.XGBRegressor(
                            n_estimators=100,
                            max_depth=4,
                            learning_rate=0.1,
                            random_state=42
                        )
                        metric_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)  # Negative MSE
                    
                    # Entrenar
                    model.fit(X_train, y_train)
                    
                    # Evaluar
                    y_pred = model.predict(X_test)
                    score = metric_func(y_test, y_pred)
                    
                    trained_models[symbol][target_name] = {
                        'model': model,
                        'score': score,
                        'feature_names': feature_names,
                        'train_samples': len(X_train),
                        'test_samples': len(X_test)
                    }
                    
                    print(f"   ✅ {target_name}: Score = {score:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ {target_name}: Error = {e}")
        
        print(f"\n📊 RESUMEN DE ENTRENAMIENTO:")
        print(f"   • Símbolos procesados: {len(trained_models)}")
        total_models = sum(len(models) for models in trained_models.values())
        print(f"   • Modelos entrenados: {total_models}")
        
        return trained_models
    
    def save_models(self, trained_models, output_dir="retrained_models"):
        """Guardar modelos entrenados"""
        print(f"\n💾 GUARDANDO MODELOS")
        print("=" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        for symbol, models in trained_models.items():
            for target_name, model_info in models.items():
                filename = f"{symbol}_{target_name}_retrained.pkl"
                filepath = os.path.join(output_dir, filename)
                
                # Guardar modelo y metadatos
                save_data = {
                    'model': model_info['model'],
                    'score': model_info['score'],
                    'feature_names': model_info['feature_names'],
                    'train_samples': model_info['train_samples'],
                    'test_samples': model_info['test_samples'],
                    'retrained_date': datetime.now().isoformat(),
                    'symbol': symbol,
                    'target': target_name
                }
                
                joblib.dump(save_data, filepath)
                saved_count += 1
                print(f"   ✅ {filename}")
        
        # Guardar resumen
        summary = {
            'retrain_date': datetime.now().isoformat(),
            'total_models': saved_count,
            'symbols': list(trained_models.keys()),
            'targets': list(set(
                target for models in trained_models.values() 
                for target in models.keys()
            ))
        }
        
        summary_path = os.path.join(output_dir, "retrain_summary.json")
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Guardados {saved_count} modelos en {output_dir}/")
        return output_dir
    
    def run_retraining_pipeline(self, days_back=30):
        """Ejecutar pipeline completo de re-entrenamiento"""
        print("🚀 PIPELINE DE RE-ENTRENAMIENTO XGBOOST")
        print("=" * 60)
        print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Datos de los últimos {days_back} días")
        
        # 1. Conectar a DB
        if not self.connect_to_database():
            return False
        
        try:
            # 2. Extraer datos
            df = self.extract_trades_data(days_back=days_back)
            if df is None:
                return False
            
            # 3. Ingeniería de features
            features_df = self.engineer_features(df)
            
            # 4. Preparar datos
            X, targets, feature_names = self.prepare_training_data(features_df)
            
            # 5. Entrenar modelos
            trained_models = self.train_models_by_symbol(X, targets, features_df, feature_names)
            
            if not trained_models:
                print("❌ No se entrenaron modelos")
                return False
            
            # 6. Guardar modelos
            output_dir = self.save_models(trained_models)
            
            print(f"\n🎉 RE-ENTRENAMIENTO COMPLETADO")
            print(f"📁 Modelos guardados en: {output_dir}")
            
            return True
            
        finally:
            if self.conn:
                self.conn.close()
                print("✅ Conexión cerrada")

def main():
    """Función principal"""
    retrainer = XGBoostRetrainer()
    
    # Ejecutar re-entrenamiento con datos de los últimos 30 días
    success = retrainer.run_retraining_pipeline(days_back=30)
    
    if success:
        print("\n💡 PRÓXIMOS PASOS:")
        print("1. Revisar modelos en carpeta retrained_models/")
        print("2. Comparar performance con modelos actuales")
        print("3. Desplegar mejores modelos a Render")
        print("4. Configurar re-entrenamiento automático")
    else:
        print("\n❌ Re-entrenamiento falló")

if __name__ == "__main__":
    main()
