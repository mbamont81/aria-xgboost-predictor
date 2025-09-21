"""
FastAPI Server - Sistema Universal Multi-Timeframe
===================================================
Servidor actualizado que soporta múltiples timeframes y símbolos
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import uvicorn
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import logging
from cachetools import TTLCache
import json
import hashlib

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialización de FastAPI
app = FastAPI(
    title="ARIA XGBoost Universal Predictor",
    description="Sistema universal multi-timeframe para predicción SL/TP",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache con TTL de 60 segundos
prediction_cache = TTLCache(maxsize=1000, ttl=60)

# Variables globales para modelos
UNIVERSAL_MODELS = {}
UNIVERSAL_SCALERS = {}
UNIVERSAL_METADATA = {}
SYSTEM_INFO = {}

# Configuración de timeframes
TIMEFRAME_CONFIG = {
    'M1': {'minutes': 1, 'sl_base': 10, 'tp_base': 30},
    'M5': {'minutes': 5, 'sl_base': 20, 'tp_base': 60},
    'M15': {'minutes': 15, 'sl_base': 30, 'tp_base': 100},
    'M30': {'minutes': 30, 'sl_base': 40, 'tp_base': 120},
    'H1': {'minutes': 60, 'sl_base': 50, 'tp_base': 150},
    'H4': {'minutes': 240, 'sl_base': 80, 'tp_base': 250},
    'D1': {'minutes': 1440, 'sl_base': 150, 'tp_base': 450}
}

# Modelos de datos
class UniversalPredictionRequest(BaseModel):
    symbol: str = Field(default="BTCUSD", description="Trading symbol")
    timeframe: str = Field(default="M15", description="Timeframe (M1, M5, M15, H1, H4, D1)")
    
    # Características básicas requeridas
    atr_percentile_100: float = Field(description="ATR percentile")
    rsi_std_20: float = Field(description="RSI standard deviation")
    price_acceleration: float = Field(description="Price acceleration")
    candle_body_ratio_mean: float = Field(description="Mean candle body ratio")
    breakout_frequency: float = Field(description="Breakout frequency")
    volume_imbalance: float = Field(description="Volume imbalance")
    rolling_autocorr_20: float = Field(description="Rolling autocorrelation")
    hurst_exponent_50: float = Field(description="Hurst exponent")
    
    # Campos opcionales adicionales
    current_price: Optional[float] = Field(default=None, description="Current price")
    spread: Optional[float] = Field(default=0, description="Current spread")
    account_balance: Optional[float] = Field(default=None, description="Account balance for risk calculation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "BTCUSD",
                "timeframe": "M15",
                "atr_percentile_100": 85.5,
                "rsi_std_20": 12.3,
                "price_acceleration": 0.0025,
                "candle_body_ratio_mean": 0.65,
                "breakout_frequency": 0.15,
                "volume_imbalance": 0.35,
                "rolling_autocorr_20": 0.25,
                "hurst_exponent_50": 0.55
            }
        }

class UniversalPredictionResponse(BaseModel):
    symbol: str
    timeframe: str
    sl_pips: float
    tp_pips: float
    sl_price: float
    tp_price: float
    regime: str
    confidence: float
    risk_reward_ratio: float
    cache_hit: bool
    model_version: str
    timestamp: datetime
    
    # Información adicional
    timeframe_adjusted: bool = Field(default=True, description="SL/TP adjusted for timeframe")
    recommended_lot_size: Optional[float] = None

# Funciones auxiliares
def get_cache_key(request: UniversalPredictionRequest) -> str:
    """Genera una clave de cache única"""
    data = request.dict()
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(data_str.encode()).hexdigest()

def encode_timeframe(timeframe: str) -> int:
    """Codifica el timeframe como valor numérico"""
    timeframe_map = {
        'M1': 1, 'M5': 2, 'M15': 3, 'M30': 4,
        'H1': 5, 'H4': 6, 'D1': 7
    }
    return timeframe_map.get(timeframe.upper(), 3)

def calculate_extended_features_universal(base_features: dict, timeframe: str) -> dict:
    """
    Extiende las características para el modelo universal
    """
    features = {}
    
    # Características normalizadas básicas
    features['returns'] = base_features.get('price_acceleration', 0)
    features['log_returns'] = np.log(1 + features['returns']) if features['returns'] > -1 else 0
    
    # Volatilidad normalizada
    features['volatility_normalized'] = base_features.get('rsi_std_20', 0) / 20
    
    # ATR normalizado (como % estimado)
    features['atr_normalized'] = base_features.get('atr_percentile_100', 50) / 100 * 2
    
    # RSI y derivados
    rsi_base = 50 + base_features.get('rsi_std_20', 0)
    features['rsi'] = max(0, min(100, rsi_base))
    features['rsi_slope'] = base_features.get('rsi_std_20', 0) / 10
    
    # MACD normalizado
    features['macd_normalized'] = base_features.get('price_acceleration', 0) * 100
    
    # Bollinger Bands
    features['bb_width_normalized'] = abs(base_features.get('rsi_std_20', 10)) / 5
    features['bb_position'] = 0.5 + base_features.get('price_acceleration', 0) * 10
    
    # Moving Average Cross
    features['ma_cross_normalized'] = base_features.get('price_acceleration', 0) * 50
    
    # Volume
    features['volume_ratio'] = 1 + base_features.get('volume_imbalance', 0)
    features['volume_trend'] = features['volume_ratio'] * 0.9
    
    # Patrones de velas
    features['body_ratio'] = base_features.get('candle_body_ratio_mean', 0.5)
    features['upper_shadow_ratio'] = 0.2
    features['lower_shadow_ratio'] = 0.2
    
    # Momentum
    features['momentum_5'] = base_features.get('price_acceleration', 0) * 5
    features['momentum_10'] = base_features.get('price_acceleration', 0) * 10
    features['momentum_20'] = base_features.get('price_acceleration', 0) * 20
    
    # Estructura de mercado
    features['higher_high'] = 1 if base_features.get('breakout_frequency', 0) > 0.5 else 0
    features['lower_low'] = 1 if base_features.get('breakout_frequency', 0) < -0.5 else 0
    
    # Hurst y autocorrelación
    features['hurst'] = base_features.get('hurst_exponent_50', 0.5)
    features['autocorr'] = base_features.get('rolling_autocorr_20', 0)
    
    # Información del timeframe
    config = TIMEFRAME_CONFIG.get(timeframe.upper(), TIMEFRAME_CONFIG['M15'])
    features['timeframe_minutes'] = config['minutes']
    features['timeframe_category'] = encode_timeframe(timeframe)
    
    # Regime features
    features['trend_strength'] = abs(features['ma_cross_normalized'])
    features['volatility_regime'] = features['volatility_normalized']
    
    return features

def adjust_targets_by_timeframe(sl_base: float, tp_base: float, timeframe: str) -> tuple:
    """Ajusta SL/TP según el timeframe"""
    config = TIMEFRAME_CONFIG.get(timeframe.upper(), TIMEFRAME_CONFIG['M15'])
    
    # Factores de ajuste basados en el timeframe
    sl_factor = config['sl_base'] / 30  # Normalizado a M15
    tp_factor = config['tp_base'] / 100
    
    sl_adjusted = sl_base * sl_factor
    tp_adjusted = tp_base * tp_factor
    
    # Aplicar límites específicos por timeframe
    if timeframe.upper() == 'M1':
        sl_adjusted = min(sl_adjusted, 25)
        tp_adjusted = min(tp_adjusted, 75)
    elif timeframe.upper() == 'M5':
        sl_adjusted = min(sl_adjusted, 40)
        tp_adjusted = min(tp_adjusted, 120)
    elif timeframe.upper() in ['H4', 'D1']:
        sl_adjusted = max(sl_adjusted, 60)
        tp_adjusted = max(tp_adjusted, 180)
    
    # Asegurar ratio mínimo
    if tp_adjusted < sl_adjusted * 1.5:
        tp_adjusted = sl_adjusted * 2
    
    return round(sl_adjusted, 2), round(tp_adjusted, 2)

def predict_universal(symbol: str, timeframe: str, features: dict) -> dict:
    """
    Realiza predicción usando el modelo universal
    """
    symbol = symbol.upper()
    timeframe = timeframe.upper()
    
    # Verificar si tenemos modelo para el símbolo
    if symbol not in UNIVERSAL_MODELS:
        # Intentar con símbolo genérico si existe
        if 'GENERIC' in UNIVERSAL_MODELS:
            symbol = 'GENERIC'
        else:
            # Usar el primer símbolo disponible como fallback
            if UNIVERSAL_MODELS:
                symbol = list(UNIVERSAL_MODELS.keys())[0]
            else:
                raise ValueError("No hay modelos cargados")
    
    # Obtener modelos y metadata del símbolo
    models = UNIVERSAL_MODELS[symbol]
    scalers = UNIVERSAL_SCALERS[symbol]
    metadata = UNIVERSAL_METADATA[symbol]
    
    # Preparar features DataFrame
    features_df = pd.DataFrame([features])
    
    # Asegurar que todas las columnas necesarias estén presentes
    for col in metadata['feature_columns']:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Seleccionar solo las columnas de entrenamiento
    features_df = features_df[metadata['feature_columns']]
    
    # Escalar features
    features_scaled = scalers['universal'].transform(features_df.fillna(0))
    
    # Predecir régimen
    regime_pred = models['regime'].predict(features_scaled)[0]
    regime_proba = models['regime'].predict_proba(features_scaled)[0]
    
    regime_map = {0: 'trending', 1: 'ranging', 2: 'volatile'}
    regime = regime_map.get(regime_pred, 'volatile')
    confidence = max(regime_proba) * 100
    
    # Predecir SL/TP base
    sl_base = models['sl'].predict(features_scaled)[0]
    tp_base = models['tp'].predict(features_scaled)[0]
    
    # Ajustar por timeframe
    sl_final, tp_final = adjust_targets_by_timeframe(sl_base, tp_base, timeframe)
    
    # Aplicar restricciones finales
    sl_final = max(10, min(500, sl_final))
    tp_final = max(sl_final * 1.5, min(1500, tp_final))
    
    # Calcular valores en precio según el símbolo
    if 'BTC' in symbol or 'BTCUSD' in symbol:
        pip_value = 1.0
    elif 'XAU' in symbol or 'GOLD' in symbol:
        pip_value = 0.1
    else:  # Forex
        pip_value = 0.0001
    
    return {
        'symbol': symbol,
        'timeframe': timeframe,
        'sl_pips': sl_final,
        'tp_pips': tp_final,
        'sl_price': round(sl_final * pip_value, 5),
        'tp_price': round(tp_final * pip_value, 5),
        'regime': regime,
        'confidence': round(confidence, 2),
        'risk_reward_ratio': round(tp_final / sl_final, 2)
    }

# Endpoints
@app.on_event("startup")
async def startup_event():
    """Carga los modelos universales al iniciar"""
    global UNIVERSAL_MODELS, UNIVERSAL_SCALERS, UNIVERSAL_METADATA, SYSTEM_INFO
    
    logger.info("🚀 Iniciando servidor Universal Multi-Timeframe...")
    
    model_path = os.getenv('MODEL_PATH', 'xgboost_universal_models.pkl')
    
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            UNIVERSAL_MODELS = data.get('models', {})
            UNIVERSAL_SCALERS = data.get('scalers', {})
            UNIVERSAL_METADATA = data.get('metadata', {})
            SYSTEM_INFO = data.get('system_info', {})
            
            logger.info(f"✅ Modelos cargados para símbolos: {list(UNIVERSAL_MODELS.keys())}")
            logger.info(f"📊 Versión del sistema: {SYSTEM_INFO.get('version', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
    else:
        logger.warning(f"⚠️ No se encontró archivo de modelos en {model_path}")
    
    logger.info("✅ Servidor iniciado correctamente")

@app.get("/")
async def root():
    """Endpoint raíz con información del sistema"""
    return {
        "service": "ARIA XGBoost Universal Predictor",
        "version": "3.0.0",
        "status": "operational",
        "system_info": SYSTEM_INFO,
        "models_loaded": len(UNIVERSAL_MODELS),
        "symbols_available": list(UNIVERSAL_MODELS.keys()),
        "timeframes_supported": list(TIMEFRAME_CONFIG.keys()),
        "cache_size": len(prediction_cache)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(UNIVERSAL_MODELS) > 0,
        "symbols": list(UNIVERSAL_MODELS.keys()),
        "cache_entries": len(prediction_cache),
        "timestamp": datetime.utcnow()
    }

@app.post("/predict/universal", response_model=UniversalPredictionResponse)
async def predict_universal_endpoint(request: UniversalPredictionRequest):
    """
    Endpoint principal de predicción universal multi-timeframe
    """
    # Verificar cache
    cache_key = get_cache_key(request)
    
    if cache_key in prediction_cache:
        cached_result = prediction_cache[cache_key]
        cached_result['cache_hit'] = True
        logger.info(f"Cache hit para {request.symbol} {request.timeframe}")
        return UniversalPredictionResponse(**cached_result)
    
    try:
        # Preparar features base
        base_features = {
            'atr_percentile_100': request.atr_percentile_100,
            'rsi_std_20': request.rsi_std_20,
            'price_acceleration': request.price_acceleration,
            'candle_body_ratio_mean': request.candle_body_ratio_mean,
            'breakout_frequency': request.breakout_frequency,
            'volume_imbalance': request.volume_imbalance,
            'rolling_autocorr_20': request.rolling_autocorr_20,
            'hurst_exponent_50': request.hurst_exponent_50,
        }
        
        # Extender features para el modelo universal
        features = calculate_extended_features_universal(base_features, request.timeframe)
        
        # Realizar predicción
        prediction = predict_universal(request.symbol, request.timeframe, features)
        
        # Calcular lot size recomendado si se proporciona balance
        recommended_lot = None
        if request.account_balance and request.account_balance > 0:
            # 1% de riesgo por operación
            risk_amount = request.account_balance * 0.01
            sl_distance = prediction['sl_pips']
            if sl_distance > 0:
                recommended_lot = round(risk_amount / (sl_distance * 10), 2)
        
        # Preparar respuesta
        response_data = {
            'symbol': prediction['symbol'],
            'timeframe': prediction['timeframe'],
            'sl_pips': prediction['sl_pips'],
            'tp_pips': prediction['tp_pips'],
            'sl_price': prediction['sl_price'],
            'tp_price': prediction['tp_price'],
            'regime': prediction['regime'],
            'confidence': prediction['confidence'],
            'risk_reward_ratio': prediction['risk_reward_ratio'],
            'cache_hit': False,
            'model_version': SYSTEM_INFO.get('version', 'unknown'),
            'timestamp': datetime.utcnow(),
            'timeframe_adjusted': True,
            'recommended_lot_size': recommended_lot
        }
        
        # Guardar en cache
        prediction_cache[cache_key] = response_data
        
        logger.info(f"Predicción para {request.symbol} {request.timeframe}: "
                   f"Régimen={prediction['regime']}, "
                   f"SL={prediction['sl_pips']}, TP={prediction['tp_pips']}")
        
        return UniversalPredictionResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/timeframes")
async def get_supported_timeframes():
    """Retorna los timeframes soportados con sus configuraciones"""
    return {
        "timeframes": TIMEFRAME_CONFIG,
        "description": "Configuración de SL/TP base por timeframe"
    }

@app.get("/symbols")
async def get_available_symbols():
    """Retorna los símbolos disponibles"""
    symbols_info = {}
    
    for symbol in UNIVERSAL_MODELS.keys():
        if symbol in UNIVERSAL_METADATA:
            symbols_info[symbol] = {
                "timeframes_trained": UNIVERSAL_METADATA[symbol].get('timeframes', []),
                "total_samples": UNIVERSAL_METADATA[symbol].get('total_samples', 0),
                "trained_at": UNIVERSAL_METADATA[symbol].get('trained_at', 'unknown')
            }
    
    return {
        "symbols": list(UNIVERSAL_MODELS.keys()),
        "details": symbols_info,
        "total": len(UNIVERSAL_MODELS)
    }

@app.post("/predict")
async def predict_legacy(request: UniversalPredictionRequest):
    """
    Endpoint legacy para compatibilidad hacia atrás
    Redirige al endpoint universal
    """
    return await predict_universal_endpoint(request)

@app.get("/model-info/{symbol}")
async def get_model_info(symbol: str):
    """Obtiene información detallada de un modelo específico"""
    symbol = symbol.upper()
    
    if symbol not in UNIVERSAL_MODELS:
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado para {symbol}")
    
    metadata = UNIVERSAL_METADATA.get(symbol, {})
    
    return {
        "symbol": symbol,
        "metadata": metadata,
        "models": list(UNIVERSAL_MODELS[symbol].keys()),
        "feature_count": len(metadata.get('feature_columns', []))
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)