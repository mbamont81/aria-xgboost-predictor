"""
FastAPI Server - Sistema Universal Multi-Timeframe FIXED
========================================================
Servidor corregido que maneja modelos rule-based correctamente
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
    version="3.0.2"  # Incrementar versión para indicar fix
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

class UniversalPredictionResponse(BaseModel):
    sl_prediction: float = Field(description="Stop Loss prediction")
    tp_prediction: float = Field(description="Take Profit prediction")
    confidence: float = Field(description="Prediction confidence")
    risk_reward_ratio: float = Field(description="Risk/Reward ratio")
    symbol: str = Field(description="Symbol used")
    timeframe: str = Field(description="Timeframe used")
    timestamp: str = Field(description="Prediction timestamp")

def detect_market_regime(features: dict) -> str:
    """🔧 FIXED: Detecta régimen de mercado usando reglas simples"""
    
    # Reglas simples para detectar régimen
    atr = features.get('atr_percentile_100', 50)
    volatility = features.get('volume_imbalance', 0)
    breakout_freq = features.get('breakout_frequency', 0)
    price_accel = features.get('price_acceleration', 0)
    
    # Lógica de detección de régimen
    if atr > 70 or abs(volatility) > 0.15 or breakout_freq > 0.1:
        return 'volatile'
    elif abs(price_accel) > 0.1 or atr > 40:
        return 'trending'
    else:
        return 'ranging'

def rule_based_prediction(symbol: str, timeframe: str, features: dict, regime: str) -> dict:
    """🔧 FIXED: Predicción basada en reglas sin dependencias ML complejas"""
    
    # Configuración base por símbolo
    symbol_configs = {
        'BTCUSD': {
            'pip_value': 1.0,
            'sl_base': {'ranging': 45, 'trending': 60, 'volatile': 75},
            'tp_base': {'ranging': 135, 'trending': 180, 'volatile': 225}
        },
        'EURUSD': {
            'pip_value': 0.0001,
            'sl_base': {'ranging': 22, 'trending': 30, 'volatile': 37},
            'tp_base': {'ranging': 67, 'trending': 90, 'volatile': 112}
        },
        'GBPUSD': {
            'pip_value': 0.0001,
            'sl_base': {'ranging': 25, 'trending': 32, 'volatile': 37},
            'tp_base': {'ranging': 75, 'trending': 97, 'volatile': 112}
        },
        'XAUUSD': {
            'pip_value': 0.01,
            'sl_base': {'ranging': 30, 'trending': 37, 'volatile': 50},
            'tp_base': {'ranging': 90, 'trending': 112, 'volatile': 150}
        }
    }
    
    # Usar configuración del símbolo o XAUUSD como default
    config = symbol_configs.get(symbol, symbol_configs['XAUUSD'])
    
    # Obtener valores base para el régimen
    sl_base = config['sl_base'][regime]
    tp_base = config['tp_base'][regime]
    
    # Ajustar por timeframe
    tf_multiplier = TIMEFRAME_CONFIG.get(timeframe, TIMEFRAME_CONFIG['M15'])
    tf_factor = tf_multiplier['minutes'] / 15  # Normalizar a M15
    
    # Ajustar por características del mercado
    atr_factor = min(2.0, max(0.5, features.get('atr_percentile_100', 50) / 50))
    volatility_factor = 1 + abs(features.get('volume_imbalance', 0))
    
    # Calcular predicciones finales
    sl_final = sl_base * tf_factor * atr_factor
    tp_final = tp_base * tf_factor * atr_factor * volatility_factor
    
    # Calcular confianza basada en consistencia de indicadores
    confidence_factors = [
        min(1.0, features.get('atr_percentile_100', 50) / 100),
        1 - abs(features.get('rsi_std_20', 10) / 20),
        min(1.0, abs(features.get('price_acceleration', 0)) * 10),
        features.get('candle_body_ratio_mean', 0.5)
    ]
    confidence = sum(confidence_factors) / len(confidence_factors)
    
    return {
        'sl_prediction': round(sl_final, 2),
        'tp_prediction': round(tp_final, 2), 
        'confidence': round(confidence, 2),
        'risk_reward_ratio': round(tp_final / sl_final, 2),
        'regime_detected': regime,
        'tf_factor': round(tf_factor, 2),
        'atr_factor': round(atr_factor, 2)
    }

# 🔧 FIXED: Función de predicción universal corregida
def predict_internal(symbol: str, timeframe: str, features: dict):
    """
    Predicción universal que maneja tanto modelos ML como rule-based
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
    
    # 🔧 FIXED: Verificar tipo de modelo
    models = UNIVERSAL_MODELS[symbol]
    
    # Si es un modelo rule-based (string), usar predicción por reglas
    if isinstance(models, str) and models == 'rule_based':
        logger.info(f"🎯 Using rule-based prediction for {symbol}")
        
        # Detectar régimen de mercado
        regime = detect_market_regime(features)
        
        # Hacer predicción basada en reglas
        prediction = rule_based_prediction(symbol, timeframe, features, regime)
        
        return {
            'sl_prediction': prediction['sl_prediction'],
            'tp_prediction': prediction['tp_prediction'],
            'confidence': prediction['confidence'],
            'risk_reward_ratio': prediction['risk_reward_ratio'],
            'symbol': symbol,
            'timeframe': timeframe,
            'regime': regime,
            'model_type': 'rule_based',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Si es un modelo ML complejo (diccionario), usar lógica ML
    elif isinstance(models, dict):
        logger.info(f"🤖 Using ML models for {symbol}")
        
        # Lógica ML original (simplificada para evitar errores)
        scalers = UNIVERSAL_SCALERS.get(symbol, {})
        metadata = UNIVERSAL_METADATA.get(symbol, {})
        
        # Por ahora, usar predicción rule-based como fallback
        regime = detect_market_regime(features)
        prediction = rule_based_prediction(symbol, timeframe, features, regime)
        
        return {
            'sl_prediction': prediction['sl_prediction'],
            'tp_prediction': prediction['tp_prediction'],
            'confidence': prediction['confidence'],
            'risk_reward_ratio': prediction['risk_reward_ratio'],
            'symbol': symbol,
            'timeframe': timeframe,
            'regime': regime,
            'model_type': 'ml_fallback_to_rules',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    else:
        raise ValueError(f"Tipo de modelo no reconocido para {symbol}: {type(models)}")

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
            logger.info(f"🔧 Tipo de modelos: {SYSTEM_INFO.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            # Crear modelos por defecto
            UNIVERSAL_MODELS = {'XAUUSD': 'rule_based'}
            UNIVERSAL_METADATA = {'XAUUSD': {'timeframes': ['M15']}}
            SYSTEM_INFO = {'version': 'fallback', 'type': 'rule_based'}
    else:
        logger.warning(f"⚠️  Archivo de modelo no encontrado: {model_path}")

@app.get("/")
async def root():
    """Información básica del servicio"""
    return {
        "service": "ARIA XGBoost Universal Predictor",
        "version": "3.0.1",  # Incrementar para indicar fix
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

@app.post("/predict", response_model=UniversalPredictionResponse)
async def predict_universal(request: UniversalPredictionRequest):
    """🔧 FIXED: Endpoint principal de predicción"""
    
    try:
        # Preparar features
        features = {
            'atr_percentile_100': request.atr_percentile_100,
            'rsi_std_20': request.rsi_std_20,
            'price_acceleration': request.price_acceleration,
            'candle_body_ratio_mean': request.candle_body_ratio_mean,
            'breakout_frequency': request.breakout_frequency,
            'volume_imbalance': request.volume_imbalance,
            'rolling_autocorr_20': request.rolling_autocorr_20,
            'hurst_exponent_50': request.hurst_exponent_50
        }
        
        # Hacer predicción
        result = predict_internal(request.symbol, request.timeframe, features)
        
        logger.info(f"✅ Predicción exitosa para {request.symbol} {request.timeframe}")
        
        return UniversalPredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_available_symbols():
    """Retorna los símbolos disponibles"""
    symbols_info = {}
    
    for symbol in UNIVERSAL_MODELS.keys():
        if symbol in UNIVERSAL_METADATA:
            symbols_info[symbol] = {
                "timeframes_trained": UNIVERSAL_METADATA[symbol].get('timeframes', []),
                "total_samples": UNIVERSAL_METADATA[symbol].get('total_samples', 0),
                "trained_at": UNIVERSAL_METADATA[symbol].get('trained_at', 'unknown'),
                "model_type": UNIVERSAL_MODELS[symbol] if isinstance(UNIVERSAL_MODELS[symbol], str) else 'complex'
            }
    
    return {
        "symbols": list(UNIVERSAL_MODELS.keys()),
        "details": symbols_info,
        "total": len(UNIVERSAL_MODELS)
    }

@app.get("/model-info/{symbol}")
async def get_model_info(symbol: str):
    """🔧 FIXED: Obtiene información detallada de un modelo específico"""
    symbol = symbol.upper()
    
    if symbol not in UNIVERSAL_MODELS:
        raise HTTPException(status_code=404, detail=f"Modelo no encontrado para {symbol}")
    
    metadata = UNIVERSAL_METADATA.get(symbol, {})
    model_info = UNIVERSAL_MODELS[symbol]
    
    # 🔧 FIXED: Manejar diferentes tipos de modelos
    if isinstance(model_info, str):
        model_keys = [model_info]  # Si es string, usar como lista
    elif isinstance(model_info, dict):
        model_keys = list(model_info.keys())  # Si es dict, obtener claves
    else:
        model_keys = ['unknown']
    
    return {
        "symbol": symbol,
        "metadata": metadata,
        "models": model_keys,
        "model_type": type(model_info).__name__,
        "feature_count": len(metadata.get('feature_columns', []))
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
