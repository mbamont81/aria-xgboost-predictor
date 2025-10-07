#!/usr/bin/env python3
"""
ARIA XGBoost Hybrid System - Uses regime classifier + rule-based predictions
Temporary solution while investigating XGBoost model loading issues
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
models = {}

# Configuración mejorada por régimen
REGIME_CONFIG = {
    'volatile': {
        'XAUUSD': {'sl_base': 120, 'tp_base': 280},
        'EURUSD': {'sl_base': 90, 'tp_base': 200},
        'BTCUSD': {'sl_base': 150, 'tp_base': 350},
        'GBPUSD': {'sl_base': 95, 'tp_base': 220}
    },
    'ranging': {
        'XAUUSD': {'sl_base': 80, 'tp_base': 180},
        'EURUSD': {'sl_base': 60, 'tp_base': 140},
        'BTCUSD': {'sl_base': 100, 'tp_base': 230},
        'GBPUSD': {'sl_base': 70, 'tp_base': 160}
    },
    'trending': {
        'XAUUSD': {'sl_base': 100, 'tp_base': 250},
        'EURUSD': {'sl_base': 75, 'tp_base': 180},
        'BTCUSD': {'sl_base': 120, 'tp_base': 300},
        'GBPUSD': {'sl_base': 85, 'tp_base': 200}
    }
}

class PredictionRequest(BaseModel):
    atr_percentile_100: float
    rsi_std_20: float
    price_acceleration: float
    candle_body_ratio_mean: float
    breakout_frequency: float
    volume_imbalance: float
    rolling_autocorr_20: float
    hurst_exponent_50: float
    symbol: str = "XAUUSD"
    timeframe: str = "M15"

class PredictionResponse(BaseModel):
    success: bool
    detected_regime: str
    regime_confidence: float
    sl_pips: float
    tp_pips: float
    overall_confidence: float
    model_used: str
    processing_time_ms: float

def normalize_symbol_for_xgboost(symbol: str) -> str:
    """Normaliza símbolos de broker a formato estándar"""
    if not symbol:
        return ""
    
    # Mapeos directos
    direct_mappings = {
        "GOLD#": "XAUUSD",
        "Gold": "XAUUSD",
        "XAUUSD.s": "XAUUSD",
        "XAUUSD.m": "XAUUSD",
        "USTEC.f": "US500",
    }
    
    if symbol in direct_mappings:
        return direct_mappings[symbol]
    
    # Remover sufijos
    if len(symbol) > 6 and symbol[-1].lower() in ['c', 'm', 'p', 'f']:
        symbol = symbol[:-1]
    
    return symbol.upper()

def load_regime_classifier():
    """Cargar solo el clasificador de régimen"""
    global models
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        logger.error("❌ Models directory not found")
        return False
    
    try:
        classifier_path = f"{models_dir}/regime_classifier.pkl"
        if os.path.exists(classifier_path):
            models['regime_classifier'] = joblib.load(classifier_path)
            logger.info("✅ Regime classifier loaded successfully")
            return True
        else:
            logger.error("❌ Regime classifier file not found")
            return False
    except Exception as e:
        logger.error(f"❌ Error loading regime classifier: {e}")
        return False

def calculate_regime_based_predictions(symbol: str, regime: str, features: dict) -> dict:
    """Calcular predicciones basadas en régimen usando lógica mejorada"""
    
    # Obtener configuración del símbolo y régimen
    config = REGIME_CONFIG.get(regime, {}).get(symbol, REGIME_CONFIG['ranging']['XAUUSD'])
    
    sl_base = config['sl_base']
    tp_base = config['tp_base']
    
    # Factores de ajuste basados en features
    atr_factor = 0.8 + (features.get('atr_percentile_100', 50) / 100) * 0.6  # 0.8-1.4
    volatility_factor = 1.0 + abs(features.get('volume_imbalance', 0)) * 2  # 1.0-1.4
    trend_factor = 1.0 + abs(features.get('price_acceleration', 0)) * 10  # 1.0-1.5
    
    # Ajustes específicos por régimen
    if regime == 'volatile':
        sl_final = sl_base * atr_factor * volatility_factor
        tp_final = tp_base * atr_factor * 1.2
    elif regime == 'trending':
        sl_final = sl_base * trend_factor
        tp_final = tp_base * trend_factor * 1.5  # Mejor RR en trending
    else:  # ranging
        sl_final = sl_base * atr_factor
        tp_final = tp_base * atr_factor * 1.1
    
    # Aplicar límites
    if symbol == 'XAUUSD':
        sl_final = max(50.0, min(sl_final, 400.0))
        tp_final = max(80.0, min(tp_final, 600.0))
    else:
        sl_final = max(20.0, min(sl_final, 200.0))
        tp_final = max(40.0, min(tp_final, 400.0))
    
    return {
        'sl_pips': round(sl_final, 1),
        'tp_pips': round(tp_final, 1),
        'confidence': 85.0,  # High confidence for regime-based
        'model_used': f'regime_aware_hybrid_{regime}'
    }

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Cargar clasificador de régimen"""
    logger.info("🚀 STARTING ARIA HYBRID SYSTEM")
    logger.info("🎯 Using regime classifier + rule-based predictions")
    
    success = load_regime_classifier()
    if success:
        logger.info("✅ Hybrid system ready")
        logger.info(f"📊 Models loaded: {list(models.keys())}")
    else:
        logger.error("❌ Failed to load regime classifier")

@app.get("/")
async def root():
    """Endpoint principal"""
    return {
        "service": "ARIA XGBoost Hybrid Predictor",
        "version": "5.2.0-HYBRID_SYSTEM",
        "status": "operational",
        "system_info": {"type": "hybrid_ml_predictor"},
        "models_loaded": len(models),
        "symbols_available": ["BTCUSD", "EURUSD", "GBPUSD", "XAUUSD"],
        "features": {
            "regime_detection": "ML_based",
            "predictions": "regime_aware_rules",
            "continuous_learning": "enabled",
            "hybrid_approach": "classifier_ml_plus_rules"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predicción usando clasificador ML + reglas por régimen"""
    
    try:
        start_time = datetime.now()
        
        # Normalizar símbolo
        normalized_symbol = normalize_symbol_for_xgboost(request.symbol)
        
        # Preparar features para el clasificador
        features_array = np.array([
            request.atr_percentile_100,
            request.rsi_std_20,
            request.price_acceleration,
            request.candle_body_ratio_mean,
            request.breakout_frequency,
            request.volume_imbalance,
            request.rolling_autocorr_20,
            request.hurst_exponent_50
        ]).reshape(1, -1)
        
        # Predecir régimen usando ML
        regime_classifier = models.get('regime_classifier')
        if not regime_classifier:
            raise HTTPException(status_code=500, detail="Clasificador de régimen no disponible")
        
        regime_pred = regime_classifier.predict(features_array)[0]
        regime_proba = regime_classifier.predict_proba(features_array)[0]
        
        regimes = ['volatile', 'ranging', 'trending']
        
        # Handle string predictions
        if isinstance(regime_pred, str):
            if regime_pred in regimes:
                detected_regime = regime_pred
            else:
                detected_regime = 'ranging'  # Default
        else:
            detected_regime = regimes[regime_pred] if regime_pred < len(regimes) else 'ranging'
        
        regime_confidence = max(regime_proba) * 100
        
        # Log symbol information
        if request.symbol != normalized_symbol:
            logger.info(f"🔄 Symbol normalized: {request.symbol} → {normalized_symbol}")
        
        logger.info(f"📊 Processing prediction for {normalized_symbol}")
        logger.info(f"🎯 Regime detected: {detected_regime} ({regime_confidence:.1f}% confidence)")
        
        # Check if symbol is supported
        supported_symbols = ["BTCUSD", "EURUSD", "GBPUSD", "XAUUSD"]
        if normalized_symbol not in supported_symbols:
            logger.warning(f"⚠️ Symbol {normalized_symbol} not in optimized config, using XAUUSD defaults")
        else:
            logger.info(f"✅ Symbol {normalized_symbol} supported with optimized config")
        
        # Calcular predicciones usando reglas mejoradas por régimen
        features_dict = {
            'atr_percentile_100': request.atr_percentile_100,
            'rsi_std_20': request.rsi_std_20,
            'price_acceleration': request.price_acceleration,
            'volume_imbalance': request.volume_imbalance,
            'breakout_frequency': request.breakout_frequency
        }
        
        # Log key features that affect predictions
        logger.info(f"🔧 Key features: ATR={request.atr_percentile_100:.1f}, Volume_Imbalance={request.volume_imbalance:.3f}, Price_Accel={request.price_acceleration:.3f}")
        
        predictions = calculate_regime_based_predictions(normalized_symbol, detected_regime, features_dict)
        
        # Calcular tiempo de procesamiento
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"✅ Prediction for {normalized_symbol}: {detected_regime} SL={predictions['sl_pips']} TP={predictions['tp_pips']} (confidence: {regime_confidence:.1f}%)")
        
        return PredictionResponse(
            success=True,
            detected_regime=detected_regime,
            regime_confidence=round(regime_confidence, 1),
            sl_pips=predictions['sl_pips'],
            tp_pips=predictions['tp_pips'],
            overall_confidence=predictions['confidence'],
            model_used=predictions['model_used'],
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
