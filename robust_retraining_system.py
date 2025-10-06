#!/usr/bin/env python3
"""
Sistema robusto de re-entrenamiento XGBoost con manejo de datos problemáticos
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

DATABASE_URL = "postgresql://aria_audit_db_user:RaXTVt9ddTIRwiDcd3prxjDbf7ga1Ant@dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com/aria_audit_db"

class RobustXGBoostRetrainer:
    def __init__(self):
        self.conn = None
        self.df = None
        
    def connect_and_load_data(self):
        """Conectar y cargar datos con validación"""
        print("🔌 CONECTANDO Y CARGANDO DATOS")
        print("=" * 50)
        
        try:
            self.conn = psycopg2.connect(DATABASE_URL)
            print("✅ Conectado a PostgreSQL")
            
            # Query con filtros de calidad
            query = """
            SELECT 
                symbol, trade_type, entry_price, exit_price, profit,
                open_time, close_time, stream_timestamp,
                atr_at_open, rsi_at_open, ma50_at_open, ma200_at_open,
                spread_at_open, volatility_at_open,
                was_xgboost_used, xgboost_confidence, market_regime
            FROM streamed_trades 
            WHERE profit IS NOT NULL 
            AND entry_price > 0 
            AND exit_price > 0
            AND atr_at_open > 0
            AND volatility_at_open > 0
            AND stream_timestamp >= %s
            ORDER BY stream_timestamp DESC
            """
            
            cutoff_date = datetime.now() - timedelta(days=30)
            self.df = pd.read_sql_query(query, self.conn, params=[cutoff_date])
            
            print(f"✅ Cargados {len(self.df)} trades")
            print(f"📅 Período: {self.df['stream_timestamp'].min()} → {self.df['stream_timestamp'].max()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def clean_and_engineer_features(self):
        """Limpiar datos y crear features robustas"""
        print("\n🧹 LIMPIEZA Y FEATURE ENGINEERING ROBUSTO")
        print("=" * 50)
        
        df = self.df.copy()
        
        # 1. Limpiar valores problemáticos
        print("🧹 Limpiando valores infinitos y NaN...")
        
        # Reemplazar infinitos con NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Estadísticas antes de limpieza
        print(f"   📊 Datos antes: {len(df)} trades")
        print(f"   📊 NaN por columna:")
        nan_counts = df.isnull().sum()
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"      {col}: {count}")
        
        # 2. Features básicas y robustas
        print("🔧 Creando features básicas...")
        
        # Features de precio (con protección contra división por cero)
        df['price_change_pct'] = np.where(
            df['entry_price'] > 0,
            (df['exit_price'] - df['entry_price']) / df['entry_price'],
            0
        )
        
        # Features temporales
        df['open_time'] = pd.to_datetime(df['open_time'])
        df['hour_of_day'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['is_london_session'] = ((df['hour_of_day'] >= 8) & (df['hour_of_day'] <= 17)).astype(int)
        df['is_ny_session'] = ((df['hour_of_day'] >= 13) & (df['hour_of_day'] <= 22)).astype(int)
        
        # Features de volatilidad (con caps para evitar outliers)
        df['volatility_capped'] = np.clip(df['volatility_at_open'], 0, df['volatility_at_open'].quantile(0.99))
        df['atr_normalized'] = np.where(
            df['entry_price'] > 0,
            np.clip(df['atr_at_open'] / df['entry_price'], 0, 0.1),  # Cap al 10%
            0
        )
        
        # Features de MA (con protección)
        df['ma_trend'] = np.where(
            (df['ma200_at_open'] > 0) & (df['ma50_at_open'] > 0),
            np.clip((df['ma50_at_open'] - df['ma200_at_open']) / df['ma200_at_open'], -0.1, 0.1),
            0
        )
        
        # Features de confidence
        df['xgboost_confidence'] = df['xgboost_confidence'].fillna(0)
        df['confidence_high'] = (df['xgboost_confidence'] >= 90).astype(int)
        df['confidence_medium'] = ((df['xgboost_confidence'] >= 80) & (df['xgboost_confidence'] < 90)).astype(int)
        
        # Encoding de régimen de mercado
        regime_map = {'trending': 1, 'volatile': 2, 'ranging': 0}
        df['market_regime_encoded'] = df['market_regime'].map(regime_map).fillna(0)
        
        # Target
        df['is_profitable'] = (df['profit'] > 0).astype(int)
        
        # 3. Seleccionar features finales (solo numéricas y limpias)
        feature_columns = [
            'trade_type', 'hour_of_day', 'day_of_week',
            'is_london_session', 'is_ny_session',
            'price_change_pct', 'volatility_capped', 'atr_normalized', 'ma_trend',
            'xgboost_confidence', 'confidence_high', 'confidence_medium',
            'market_regime_encoded'
        ]
        
        # Verificar que todas las features existen
        available_features = [col for col in feature_columns if col in df.columns]
        
        # Crear dataset final
        X = df[available_features].copy()
        y = df['is_profitable'].copy()
        
        # Limpiar NaN finales
        X = X.fillna(0)
        
        # Verificar que no hay infinitos
        inf_check = np.isinf(X.values).any()
        if inf_check:
            print("⚠️  Encontrados infinitos, limpiando...")
            X = X.replace([np.inf, -np.inf], 0)
        
        print(f"✅ Features finales: {len(available_features)}")
        print(f"✅ Samples limpios: {len(X)}")
        print(f"✅ Win rate: {y.mean()*100:.1f}%")
        
        return X, y, available_features, df
    
    def train_robust_models(self, X, y, feature_names, df):
        """Entrenar modelos robustos"""
        print("\n🤖 ENTRENANDO MODELOS ROBUSTOS")
        print("=" * 50)
        
        models = {}
        
        # 1. Modelo general
        print("🎯 Entrenando modelo general...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            model_general = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            model_general.fit(X_train, y_train)
            
            y_pred = model_general.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            win_rate_test = y_test.mean() * 100
            win_rate_pred = y_pred.mean() * 100
            
            models['general'] = {
                'model': model_general,
                'accuracy': accuracy,
                'win_rate_test': win_rate_test,
                'win_rate_pred': win_rate_pred,
                'feature_names': feature_names,
                'samples': len(X_train)
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   📊 Win Rate Test: {win_rate_test:.1f}%")
            print(f"   🎯 Win Rate Pred: {win_rate_pred:.1f}%")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 2. Modelo de alta confidence
        print("\n⭐ Entrenando modelo de alta confidence...")
        try:
            high_conf_mask = df['xgboost_confidence'] >= 90
            if high_conf_mask.sum() > 100:
                X_high = X[high_conf_mask]
                y_high = y[high_conf_mask]
                
                X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
                    X_high, y_high, test_size=0.2, random_state=42, stratify=y_high
                )
                
                model_high_conf = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model_high_conf.fit(X_train_h, y_train_h)
                
                y_pred_h = model_high_conf.predict(X_test_h)
                accuracy_h = accuracy_score(y_test_h, y_pred_h)
                win_rate_test_h = y_test_h.mean() * 100
                win_rate_pred_h = y_pred_h.mean() * 100
                
                models['high_confidence'] = {
                    'model': model_high_conf,
                    'accuracy': accuracy_h,
                    'win_rate_test': win_rate_test_h,
                    'win_rate_pred': win_rate_pred_h,
                    'feature_names': feature_names,
                    'samples': len(X_train_h)
                }
                
                print(f"   ✅ Accuracy: {accuracy_h:.4f}")
                print(f"   📊 Win Rate Test: {win_rate_test_h:.1f}%")
                print(f"   🎯 Win Rate Pred: {win_rate_pred_h:.1f}%")
            else:
                print("   ⚠️  Insuficientes datos de alta confidence")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # 3. Modelo por régimen trending
        print("\n🌊 Entrenando modelo para régimen trending...")
        try:
            trending_mask = df['market_regime'] == 'trending'
            if trending_mask.sum() > 100:
                X_trend = X[trending_mask]
                y_trend = y[trending_mask]
                
                X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
                    X_trend, y_trend, test_size=0.2, random_state=42, stratify=y_trend
                )
                
                model_trending = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                )
                
                model_trending.fit(X_train_t, y_train_t)
                
                y_pred_t = model_trending.predict(X_test_t)
                accuracy_t = accuracy_score(y_test_t, y_pred_t)
                win_rate_test_t = y_test_t.mean() * 100
                win_rate_pred_t = y_pred_t.mean() * 100
                
                models['trending'] = {
                    'model': model_trending,
                    'accuracy': accuracy_t,
                    'win_rate_test': win_rate_test_t,
                    'win_rate_pred': win_rate_pred_t,
                    'feature_names': feature_names,
                    'samples': len(X_train_t)
                }
                
                print(f"   ✅ Accuracy: {accuracy_t:.4f}")
                print(f"   📊 Win Rate Test: {win_rate_test_t:.1f}%")
                print(f"   🎯 Win Rate Pred: {win_rate_pred_t:.1f}%")
            else:
                print("   ⚠️  Insuficientes datos de trending")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print(f"\n✅ Modelos entrenados: {len(models)}")
        return models
    
    def save_models(self, models, output_dir="robust_retrained_models"):
        """Guardar modelos y metadatos"""
        print(f"\n💾 GUARDANDO MODELOS ROBUSTOS")
        print("=" * 50)
        
        os.makedirs(output_dir, exist_ok=True)
        
        saved_count = 0
        for model_name, model_info in models.items():
            filename = f"{model_name}_robust.pkl"
            filepath = os.path.join(output_dir, filename)
            
            save_data = {
                'model': model_info['model'],
                'metadata': {
                    'accuracy': model_info['accuracy'],
                    'win_rate_test': model_info['win_rate_test'],
                    'win_rate_pred': model_info['win_rate_pred'],
                    'samples': model_info['samples'],
                    'retrained_date': datetime.now().isoformat(),
                    'model_type': 'robust_xgboost',
                    'dataset': model_name
                },
                'feature_names': model_info['feature_names']
            }
            
            joblib.dump(save_data, filepath)
            saved_count += 1
            print(f"   ✅ {filename}")
        
        # Resumen
        summary = {
            'retraining_date': datetime.now().isoformat(),
            'total_models': saved_count,
            'models': list(models.keys()),
            'data_period': {
                'start': str(self.df['stream_timestamp'].min()),
                'end': str(self.df['stream_timestamp'].max()),
                'total_trades': len(self.df)
            },
            'improvements': {
                'data_cleaning': 'Applied',
                'robust_features': 'Applied',
                'confidence_filtering': 'Available',
                'regime_optimization': 'Available'
            }
        }
        
        summary_path = os.path.join(output_dir, "robust_retraining_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ robust_retraining_summary.json")
        print(f"\n✅ Guardados {saved_count} modelos en {output_dir}/")
        
        return output_dir
    
    def run_robust_retraining(self):
        """Ejecutar pipeline robusto completo"""
        print("🚀 PIPELINE ROBUSTO DE RE-ENTRENAMIENTO")
        print("=" * 60)
        print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # 1. Conectar y cargar datos
            if not self.connect_and_load_data():
                return False
            
            # 2. Limpiar y crear features
            X, y, feature_names, df_clean = self.clean_and_engineer_features()
            
            if len(X) < 100:
                print("❌ Insuficientes datos después de limpieza")
                return False
            
            # 3. Entrenar modelos
            models = self.train_robust_models(X, y, feature_names, df_clean)
            
            if not models:
                print("❌ No se entrenaron modelos")
                return False
            
            # 4. Guardar modelos
            output_dir = self.save_models(models)
            
            print(f"\n🎉 RE-ENTRENAMIENTO ROBUSTO COMPLETADO")
            print("=" * 50)
            print(f"📁 Modelos guardados en: {output_dir}")
            print(f"🤖 Modelos entrenados: {len(models)}")
            
            # Mostrar mejores modelos
            if models:
                best_model = max(models.items(), key=lambda x: x[1]['win_rate_test'])
                print(f"\n🏆 MEJOR MODELO: {best_model[0]}")
                print(f"   📊 Win Rate: {best_model[1]['win_rate_test']:.1f}%")
                print(f"   🎯 Accuracy: {best_model[1]['accuracy']:.4f}")
                print(f"   📈 Samples: {best_model[1]['samples']:,}")
            
            return True
            
        finally:
            if self.conn:
                self.conn.close()
                print("✅ Conexión cerrada")

def main():
    """Función principal"""
    retrainer = RobustXGBoostRetrainer()
    
    print("🎯 INICIANDO RE-ENTRENAMIENTO ROBUSTO")
    print("Este proceso va a:")
    print("• Limpiar datos problemáticos (infinitos, NaN)")
    print("• Crear features robustas y validadas")
    print("• Entrenar modelos especializados")
    print("• Manejar casos edge de manera segura")
    
    success = retrainer.run_robust_retraining()
    
    if success:
        print("\n💡 PRÓXIMOS PASOS:")
        print("1. ✅ Modelos robustos creados")
        print("2. 🔄 Probar modelos localmente")
        print("3. 📊 Comparar con modelos actuales")
        print("4. 🚀 Desplegar mejores modelos")
    else:
        print("\n❌ Re-entrenamiento falló")

if __name__ == "__main__":
    main()
