"""
Multi-Timeframe XGBoost Predictor
==================================
Sistema universal que opera en múltiples timeframes con un solo modelo
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MultiTimeframePredictor:
    """
    Predictor universal que ajusta automáticamente por timeframe
    """
    
    # Configuración de timeframes y sus multiplicadores
    TIMEFRAME_CONFIG = {
        'M1': {
            'minutes': 1,
            'sl_base': 10,      # SL base en pips
            'tp_base': 30,      # TP base en pips
            'lookback_multiplier': 1.0,
            'volatility_window': 60,  # 1 hora en M1
        },
        'M5': {
            'minutes': 5,
            'sl_base': 20,
            'tp_base': 60,
            'lookback_multiplier': 0.8,
            'volatility_window': 60,  # 5 horas en M5
        },
        'M15': {
            'minutes': 15,
            'sl_base': 30,
            'tp_base': 100,
            'lookback_multiplier': 0.6,
            'volatility_window': 40,  # 10 horas en M15
        },
        'M30': {
            'minutes': 30,
            'sl_base': 40,
            'tp_base': 120,
            'lookback_multiplier': 0.5,
            'volatility_window': 48,  # 24 horas en M30
        },
        'H1': {
            'minutes': 60,
            'sl_base': 50,
            'tp_base': 150,
            'lookback_multiplier': 0.4,
            'volatility_window': 24,  # 24 horas en H1
        },
        'H4': {
            'minutes': 240,
            'sl_base': 80,
            'tp_base': 250,
            'lookback_multiplier': 0.3,
            'volatility_window': 30,  # 5 días en H4
        },
        'D1': {
            'minutes': 1440,
            'sl_base': 150,
            'tp_base': 450,
            'lookback_multiplier': 0.2,
            'volatility_window': 20,  # 20 días en D1
        }
    }
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def load_multi_timeframe_data(self, symbol, data_files):
        """
        Carga datos de múltiples timeframes
        
        Args:
            symbol: Símbolo a procesar
            data_files: Dict con paths a archivos CSV por timeframe
                       {'M1': 'path/to/M1.csv', 'M5': 'path/to/M5.csv', ...}
        """
        all_data = {}
        
        for timeframe, filepath in data_files.items():
            print(f"📊 Cargando {symbol} {timeframe}...")
            
            # Leer archivo con encoding UTF-16 LE
            with open(filepath, 'rb') as f:
                content = f.read()
            
            text = content.decode('utf-16-le')
            lines = text.strip().split('\n')
            header = lines[0].split(';')
            data = []
            
            for line in lines[1:]:
                values = line.split(';')
                if len(values) == 6:
                    data.append(values)
            
            df = pd.DataFrame(data, columns=header)
            
            # Convertir tipos
            df['Time'] = pd.to_datetime(df['Time'], format='%Y.%m.%d %H:%M')
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['Symbol'] = symbol
            df['Timeframe'] = timeframe
            df.set_index('Time', inplace=True)
            
            all_data[timeframe] = df
            print(f"✅ {timeframe}: {len(df)} velas cargadas")
        
        return all_data
    
    def calculate_normalized_features(self, df, timeframe='M15'):
        """
        Calcula características normalizadas independientes del timeframe
        """
        features = pd.DataFrame(index=df.index)
        config = self.TIMEFRAME_CONFIG.get(timeframe, self.TIMEFRAME_CONFIG['M15'])
        
        # 1. Características de precio normalizadas
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volatilidad normalizada por timeframe
        volatility_window = config['volatility_window']
        features['volatility_normalized'] = features['returns'].rolling(volatility_window).std()
        
        # 3. ATR normalizado
        atr = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        features['atr_normalized'] = atr / df['Close'] * 100  # Como % del precio
        
        # 4. RSI (independiente del timeframe)
        features['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        features['rsi_slope'] = features['rsi'].diff()
        
        # 5. MACD normalizado
        macd = ta.trend.MACD(df['Close'])
        features['macd_normalized'] = macd.macd_diff() / df['Close'] * 100
        
        # 6. Bollinger Bands normalizados
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        bb_width = bb.bollinger_hband() - bb.bollinger_lband()
        features['bb_width_normalized'] = bb_width / df['Close'] * 100
        features['bb_position'] = (df['Close'] - bb.bollinger_lband()) / bb_width
        
        # 7. Moving Average Crossovers normalizados
        ma_fast = df['Close'].rolling(10).mean()
        ma_slow = df['Close'].rolling(30).mean()
        features['ma_cross_normalized'] = (ma_fast - ma_slow) / ma_slow * 100
        
        # 8. Volume features (si disponible)
        if df['Volume'].sum() > 0:
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            features['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        
        # 9. Patrones de velas normalizados
        candle_range = df['High'] - df['Low']
        features['body_ratio'] = abs(df['Close'] - df['Open']) / (candle_range + 0.0001)
        features['upper_shadow_ratio'] = (df['High'] - df[['Close', 'Open']].max(axis=1)) / (candle_range + 0.0001)
        features['lower_shadow_ratio'] = (df[['Close', 'Open']].min(axis=1) - df['Low']) / (candle_range + 0.0001)
        
        # 10. Momentum normalizado
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # 11. Características de estructura de mercado
        features['higher_high'] = (df['High'] > df['High'].shift(1).rolling(20).max()).astype(int)
        features['lower_low'] = (df['Low'] < df['Low'].shift(1).rolling(20).min()).astype(int)
        
        # 12. Hurst exponent (tendencia vs reversión)
        features['hurst'] = df['Close'].rolling(50).apply(self.calculate_hurst_simple)
        
        # 13. Autocorrelación
        features['autocorr'] = df['Close'].rolling(30).apply(lambda x: x.autocorr(lag=1) if len(x) > 1 else 0)
        
        # 14. Timeframe como feature categórica
        features['timeframe_minutes'] = config['minutes']
        features['timeframe_category'] = self.encode_timeframe(timeframe)
        
        # 15. Regime detection features
        features['trend_strength'] = abs(features['ma_cross_normalized'])
        features['volatility_regime'] = features['volatility_normalized'].rolling(20).mean()
        
        return features
    
    def calculate_hurst_simple(self, ts):
        """Calcula el exponente de Hurst simplificado"""
        try:
            if len(ts) < 20:
                return 0.5
            
            # Método simplificado R/S
            mean = np.mean(ts)
            std = np.std(ts)
            if std == 0:
                return 0.5
                
            ts_norm = (ts - mean) / std
            cumsum = np.cumsum(ts_norm - np.mean(ts_norm))
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(ts)
            
            if S == 0:
                return 0.5
                
            return np.log(R/S) / np.log(len(ts)) if R/S > 0 else 0.5
        except:
            return 0.5
    
    def encode_timeframe(self, timeframe):
        """Codifica el timeframe como valor numérico"""
        timeframe_map = {
            'M1': 1, 'M5': 2, 'M15': 3, 'M30': 4,
            'H1': 5, 'H4': 6, 'D1': 7
        }
        return timeframe_map.get(timeframe, 3)
    
    def adjust_targets_by_timeframe(self, base_sl, base_tp, timeframe):
        """
        Ajusta SL/TP según el timeframe
        """
        config = self.TIMEFRAME_CONFIG.get(timeframe, self.TIMEFRAME_CONFIG['M15'])
        
        # Ajustar basado en la configuración del timeframe
        sl = base_sl * (config['sl_base'] / 30)  # Normalizado a M15 como base
        tp = base_tp * (config['tp_base'] / 100)
        
        # Aplicar límites por timeframe
        if timeframe == 'M1':
            sl = min(sl, 20)
            tp = min(tp, 60)
        elif timeframe == 'M5':
            sl = min(sl, 35)
            tp = min(tp, 100)
        elif timeframe in ['H4', 'D1']:
            sl = max(sl, 50)
            tp = max(tp, 150)
        
        return sl, tp
    
    def create_training_dataset(self, all_timeframes_data):
        """
        Crea un dataset unificado de todos los timeframes
        """
        all_features = []
        all_targets = []
        
        for timeframe, df in all_timeframes_data.items():
            print(f"🔧 Procesando features para {timeframe}...")
            
            # Calcular features normalizadas
            features = self.calculate_normalized_features(df, timeframe)
            
            # Crear targets sintéticos basados en movimientos futuros
            config = self.TIMEFRAME_CONFIG[timeframe]
            
            # Calcular el movimiento futuro en N velas
            future_periods = min(10, len(df) // 100)  # Adaptar al timeframe
            future_high = df['High'].shift(-future_periods).rolling(future_periods).max()
            future_low = df['Low'].shift(-future_periods).rolling(future_periods).min()
            
            # Targets basados en el rango futuro
            sl_target = abs(df['Close'] - future_low) / df['Close'] * 10000  # En pips
            tp_target = abs(future_high - df['Close']) / df['Close'] * 10000
            
            # Ajustar por timeframe
            sl_target = sl_target * config['sl_base'] / 30
            tp_target = tp_target * config['tp_base'] / 100
            
            # Agregar targets al dataset
            features['sl_target'] = sl_target
            features['tp_target'] = tp_target
            
            # Filtrar NaN y valores extremos
            features = features.dropna()
            features = features[(features['sl_target'] > 5) & (features['sl_target'] < 500)]
            features = features[(features['tp_target'] > 10) & (features['tp_target'] < 1000)]
            
            all_features.append(features)
            
            print(f"✅ {timeframe}: {len(features)} muestras generadas")
        
        # Combinar todos los timeframes
        combined_df = pd.concat(all_features, ignore_index=True)
        
        # Separar features y targets
        target_columns = ['sl_target', 'tp_target']
        feature_columns = [col for col in combined_df.columns if col not in target_columns]
        
        self.feature_columns = feature_columns
        
        X = combined_df[feature_columns].fillna(0)
        y_sl = combined_df['sl_target']
        y_tp = combined_df['tp_target']
        
        print(f"\n✅ Dataset combinado: {len(X)} muestras totales")
        print(f"📊 Características: {len(feature_columns)}")
        
        return X, y_sl, y_tp
    
    def train_universal_model(self, X, y_sl, y_tp):
        """
        Entrena el modelo universal multi-timeframe
        """
        print("\n🤖 Entrenando modelo universal multi-timeframe...")
        
        # División train/test
        X_train, X_test, y_sl_train, y_sl_test, y_tp_train, y_tp_test = train_test_split(
            X, y_sl, y_tp, test_size=0.2, random_state=42
        )
        
        # Escalar features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Modelo para SL
        print("📈 Entrenando modelo SL...")
        sl_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        sl_model.fit(
            X_train_scaled, y_sl_train,
            eval_set=[(X_test_scaled, y_sl_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Modelo para TP
        print("📈 Entrenando modelo TP...")
        tp_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        tp_model.fit(
            X_train_scaled, y_tp_train,
            eval_set=[(X_test_scaled, y_tp_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Modelo de detección de régimen
        print("📈 Entrenando detector de régimen...")
        
        # Crear labels de régimen basados en volatilidad y tendencia
        volatility_percentile = X_train['volatility_normalized'].quantile([0.33, 0.67])
        trend_percentile = X_train['trend_strength'].quantile([0.33, 0.67])
        
        regime_labels = pd.Series(index=X_train.index, data=1)  # Default: ranging
        
        # Trending: alta tendencia, baja volatilidad
        trending_mask = (X_train['trend_strength'] > trend_percentile.iloc[1]) & \
                       (X_train['volatility_normalized'] < volatility_percentile.iloc[0])
        regime_labels[trending_mask] = 0
        
        # Volatile: alta volatilidad
        volatile_mask = X_train['volatility_normalized'] > volatility_percentile.iloc[1]
        regime_labels[volatile_mask] = 2
        
        regime_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softmax',
            num_class=3,
            random_state=42
        )
        regime_model.fit(X_train_scaled, regime_labels)
        
        # Guardar modelos
        self.models['sl'] = sl_model
        self.models['tp'] = tp_model
        self.models['regime'] = regime_model
        self.scalers['universal'] = scaler
        
        # Evaluación
        sl_mae = np.mean(np.abs(sl_model.predict(X_test_scaled) - y_sl_test))
        tp_mae = np.mean(np.abs(tp_model.predict(X_test_scaled) - y_tp_test))
        
        print(f"\n✅ Modelos entrenados exitosamente:")
        print(f"   SL MAE: {sl_mae:.2f} pips")
        print(f"   TP MAE: {tp_mae:.2f} pips")
        
        # Feature importance
        feature_importance = {}
        for name, importance in zip(self.feature_columns, sl_model.feature_importances_):
            feature_importance[name] = importance
        
        # Top 10 features más importantes
        print("\n🎯 Top 10 Features más importantes:")
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"   {i}. {feature}: {importance:.4f}")
        
        return sl_model, tp_model, regime_model
    
    def predict(self, features_dict, timeframe='M15', symbol='BTCUSD'):
        """
        Realiza predicción para cualquier timeframe
        """
        # Preparar features con normalización
        features_df = pd.DataFrame([features_dict])
        
        # Agregar información del timeframe
        config = self.TIMEFRAME_CONFIG.get(timeframe, self.TIMEFRAME_CONFIG['M15'])
        features_df['timeframe_minutes'] = config['minutes']
        features_df['timeframe_category'] = self.encode_timeframe(timeframe)
        
        # Asegurar que todas las columnas estén presentes
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Seleccionar solo las columnas de entrenamiento
        features_df = features_df[self.feature_columns]
        
        # Escalar
        features_scaled = self.scalers['universal'].transform(features_df.fillna(0))
        
        # Predecir régimen
        regime_pred = self.models['regime'].predict(features_scaled)[0]
        regime_proba = self.models['regime'].predict_proba(features_scaled)[0]
        
        regime_map = {0: 'trending', 1: 'ranging', 2: 'volatile'}
        regime = regime_map[regime_pred]
        confidence = max(regime_proba) * 100
        
        # Predecir SL/TP base
        sl_base = self.models['sl'].predict(features_scaled)[0]
        tp_base = self.models['tp'].predict(features_scaled)[0]
        
        # Ajustar por timeframe
        sl_final, tp_final = self.adjust_targets_by_timeframe(sl_base, tp_base, timeframe)
        
        # Aplicar restricciones finales
        sl_final = max(10, min(300, sl_final))
        tp_final = max(sl_final * 1.5, min(1000, tp_final))
        
        # Convertir a precio según el símbolo
        if symbol == 'BTCUSD':
            pip_value = 1.0
        elif symbol in ['XAUUSD', 'GOLD']:
            pip_value = 0.1
        else:  # Forex
            pip_value = 0.0001
        
        return {
            'timeframe': timeframe,
            'regime': regime,
            'confidence': round(confidence, 2),
            'sl_pips': round(sl_final, 2),
            'tp_pips': round(tp_final, 2),
            'sl_price': round(sl_final * pip_value, 5),
            'tp_price': round(tp_final * pip_value, 5),
            'risk_reward_ratio': round(tp_final / sl_final, 2),
            'model_version': 'universal_v1'
        }

# Función de entrenamiento completo
def train_multi_timeframe_system(symbol='BTCUSD', data_files=None):
    """
    Entrena el sistema multi-timeframe completo
    """
    print("="*60)
    print("🚀 ENTRENAMIENTO MULTI-TIMEFRAME UNIVERSAL")
    print("="*60)
    print(f"Símbolo: {symbol}")
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Archivos de datos por defecto
    if data_files is None:
        data_files = {
            'M1': f'data/{symbol}_M1_last20000.csv',
            'M5': f'data/{symbol}_M5_last20000.csv',
            'M15': f'data/{symbol}_M15_last20000.csv',
            'H1': f'data/{symbol}_H1_last20000.csv',
            'H4': f'data/{symbol}_H4_last20000.csv',
        }
    
    # Inicializar predictor
    predictor = MultiTimeframePredictor()
    
    # Cargar datos
    print("\n📊 Cargando datos multi-timeframe...")
    all_data = predictor.load_multi_timeframe_data(symbol, data_files)
    
    # Crear dataset de entrenamiento
    print("\n🔧 Creando dataset unificado...")
    X, y_sl, y_tp = predictor.create_training_dataset(all_data)
    
    # Entrenar modelo universal
    predictor.train_universal_model(X, y_sl, y_tp)
    
    # Guardar modelo
    import pickle
    print("\n💾 Guardando modelo universal...")
    
    model_data = {
        'models': predictor.models,
        'scalers': predictor.scalers,
        'feature_columns': predictor.feature_columns,
        'timeframe_config': predictor.TIMEFRAME_CONFIG,
        'metadata': {
            'version': 'universal_v1',
            'symbol': symbol,
            'timeframes': list(all_data.keys()),
            'trained_at': datetime.now().isoformat(),
            'total_samples': len(X)
        }
    }
    
    output_file = f'{symbol}_universal_model.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Modelo guardado en {output_file}")
    
    # Prueba de predicción
    print("\n🧪 Prueba de predicción por timeframe:")
    print("-"*40)
    
    test_features = {
        'returns': 0.001,
        'volatility_normalized': 0.015,
        'atr_normalized': 1.2,
        'rsi': 55,
        'rsi_slope': 2,
        'macd_normalized': 0.05,
        'bb_width_normalized': 2.5,
        'bb_position': 0.6,
        'ma_cross_normalized': 0.5,
        'volume_ratio': 1.2,
        'body_ratio': 0.65,
        'momentum_5': 0.01,
        'trend_strength': 0.8,
        'volatility_regime': 0.02,
        'hurst': 0.55,
        'autocorr': 0.3
    }
    
    for tf in ['M1', 'M5', 'M15', 'H1', 'H4']:
        pred = predictor.predict(test_features, timeframe=tf, symbol=symbol)
        print(f"\n{tf}: SL={pred['sl_pips']} pips, TP={pred['tp_pips']} pips, "
              f"R:R={pred['risk_reward_ratio']}, Régimen={pred['regime']}")
    
    print("\n" + "="*60)
    print("✅ SISTEMA MULTI-TIMEFRAME ENTRENADO EXITOSAMENTE")
    print("="*60)
    
    return predictor

if __name__ == "__main__":
    # Ejemplo de uso
    predictor = train_multi_timeframe_system('BTCUSD')
