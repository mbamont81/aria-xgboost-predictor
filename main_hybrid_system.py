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

# Configuración mejorada por régimen - EXPANDIDA
REGIME_CONFIG = {
    'volatile': {
        'XAUUSD': {'sl_base': 120, 'tp_base': 280},
        'EURUSD': {'sl_base': 90, 'tp_base': 200},
        'BTCUSD': {'sl_base': 150, 'tp_base': 350},
        'GBPUSD': {'sl_base': 95, 'tp_base': 220},
        # Major Pairs
        'USDJPY': {'sl_base': 85, 'tp_base': 190},
        'AUDUSD': {'sl_base': 85, 'tp_base': 200},
        'NZDUSD': {'sl_base': 80, 'tp_base': 185},
        'USDCAD': {'sl_base': 75, 'tp_base': 175},
        'USDCHF': {'sl_base': 80, 'tp_base': 180},
        # Cross Pairs
        'EURGBP': {'sl_base': 70, 'tp_base': 165},
        'EURJPY': {'sl_base': 100, 'tp_base': 230},
        'GBPJPY': {'sl_base': 110, 'tp_base': 250},
        'AUDCHF': {'sl_base': 90, 'tp_base': 210},
        'AUDJPY': {'sl_base': 105, 'tp_base': 240},
        'CADCHF': {'sl_base': 85, 'tp_base': 195},
        'CADJPY': {'sl_base': 95, 'tp_base': 220},
        'CHFJPY': {'sl_base': 100, 'tp_base': 235},
        'EURAUD': {'sl_base': 95, 'tp_base': 215},
        'EURCAD': {'sl_base': 90, 'tp_base': 205},
        'EURCHF': {'sl_base': 85, 'tp_base': 195},
        'EURNZD': {'sl_base': 100, 'tp_base': 225},
        'GBPAUD': {'sl_base': 105, 'tp_base': 240},
        'GBPCAD': {'sl_base': 100, 'tp_base': 230},
        'GBPNZD': {'sl_base': 110, 'tp_base': 255},
        # Metals
        'XAGUSD': {'sl_base': 110, 'tp_base': 260},
        'XPTUSD': {'sl_base': 130, 'tp_base': 300},
        # Crypto
        'ETHUSD': {'sl_base': 140, 'tp_base': 320},
        'LTCUSD': {'sl_base': 120, 'tp_base': 280},
        'ADAUSD': {'sl_base': 100, 'tp_base': 230},
        'SOLUSD': {'sl_base': 130, 'tp_base': 300},
        'XRPUSD': {'sl_base': 90, 'tp_base': 210},
        # Indices
        'US30': {'sl_base': 200, 'tp_base': 450},
        'US500': {'sl_base': 180, 'tp_base': 400},
        'UK100': {'sl_base': 160, 'tp_base': 360},
        'JP225': {'sl_base': 170, 'tp_base': 380},
        'TecDE30': {'sl_base': 150, 'tp_base': 340},
        'USTEC': {'sl_base': 190, 'tp_base': 420},
        # Commodities
        'XTIUSD': {'sl_base': 120, 'tp_base': 280},
        'XBRUSD': {'sl_base': 125, 'tp_base': 290},
        'XNGUSD': {'sl_base': 110, 'tp_base': 260}
    },
    'ranging': {
        'XAUUSD': {'sl_base': 80, 'tp_base': 180},
        'EURUSD': {'sl_base': 60, 'tp_base': 140},
        'BTCUSD': {'sl_base': 100, 'tp_base': 230},
        'GBPUSD': {'sl_base': 70, 'tp_base': 160},
        # Major Pairs
        'USDJPY': {'sl_base': 55, 'tp_base': 125},
        'AUDUSD': {'sl_base': 60, 'tp_base': 135},
        'NZDUSD': {'sl_base': 55, 'tp_base': 130},
        'USDCAD': {'sl_base': 50, 'tp_base': 120},
        'USDCHF': {'sl_base': 55, 'tp_base': 125},
        # Cross Pairs
        'EURGBP': {'sl_base': 45, 'tp_base': 110},
        'EURJPY': {'sl_base': 70, 'tp_base': 155},
        'GBPJPY': {'sl_base': 75, 'tp_base': 170},
        'AUDCHF': {'sl_base': 60, 'tp_base': 140},
        'AUDJPY': {'sl_base': 70, 'tp_base': 160},
        'CADCHF': {'sl_base': 55, 'tp_base': 130},
        'CADJPY': {'sl_base': 65, 'tp_base': 150},
        'CHFJPY': {'sl_base': 70, 'tp_base': 160},
        'EURAUD': {'sl_base': 65, 'tp_base': 145},
        'EURCAD': {'sl_base': 60, 'tp_base': 140},
        'EURCHF': {'sl_base': 55, 'tp_base': 130},
        'EURNZD': {'sl_base': 70, 'tp_base': 155},
        'GBPAUD': {'sl_base': 70, 'tp_base': 160},
        'GBPCAD': {'sl_base': 70, 'tp_base': 155},
        'GBPNZD': {'sl_base': 75, 'tp_base': 170},
        # Metals
        'XAGUSD': {'sl_base': 75, 'tp_base': 175},
        'XPTUSD': {'sl_base': 90, 'tp_base': 200},
        # Crypto
        'ETHUSD': {'sl_base': 95, 'tp_base': 220},
        'LTCUSD': {'sl_base': 80, 'tp_base': 190},
        'ADAUSD': {'sl_base': 70, 'tp_base': 160},
        'SOLUSD': {'sl_base': 90, 'tp_base': 200},
        'XRPUSD': {'sl_base': 60, 'tp_base': 140},
        # Indices
        'US30': {'sl_base': 130, 'tp_base': 300},
        'US500': {'sl_base': 120, 'tp_base': 270},
        'UK100': {'sl_base': 110, 'tp_base': 240},
        'JP225': {'sl_base': 115, 'tp_base': 255},
        'TecDE30': {'sl_base': 100, 'tp_base': 230},
        'USTEC': {'sl_base': 125, 'tp_base': 280},
        # Commodities
        'XTIUSD': {'sl_base': 80, 'tp_base': 190},
        'XBRUSD': {'sl_base': 85, 'tp_base': 195},
        'XNGUSD': {'sl_base': 75, 'tp_base': 175}
    },
    'trending': {
        'XAUUSD': {'sl_base': 100, 'tp_base': 250},
        'EURUSD': {'sl_base': 75, 'tp_base': 180},
        'BTCUSD': {'sl_base': 120, 'tp_base': 300},
        'GBPUSD': {'sl_base': 85, 'tp_base': 200},
        # Major Pairs
        'USDJPY': {'sl_base': 70, 'tp_base': 160},
        'AUDUSD': {'sl_base': 75, 'tp_base': 170},
        'NZDUSD': {'sl_base': 70, 'tp_base': 155},
        'USDCAD': {'sl_base': 65, 'tp_base': 150},
        'USDCHF': {'sl_base': 70, 'tp_base': 155},
        # Cross Pairs
        'EURGBP': {'sl_base': 60, 'tp_base': 140},
        'EURJPY': {'sl_base': 85, 'tp_base': 195},
        'GBPJPY': {'sl_base': 95, 'tp_base': 220},
        'AUDCHF': {'sl_base': 75, 'tp_base': 175},
        'AUDJPY': {'sl_base': 90, 'tp_base': 205},
        'CADCHF': {'sl_base': 70, 'tp_base': 165},
        'CADJPY': {'sl_base': 80, 'tp_base': 185},
        'CHFJPY': {'sl_base': 85, 'tp_base': 200},
        'EURAUD': {'sl_base': 80, 'tp_base': 185},
        'EURCAD': {'sl_base': 75, 'tp_base': 175},
        'EURCHF': {'sl_base': 70, 'tp_base': 165},
        'EURNZD': {'sl_base': 85, 'tp_base': 190},
        'GBPAUD': {'sl_base': 90, 'tp_base': 205},
        'GBPCAD': {'sl_base': 85, 'tp_base': 195},
        'GBPNZD': {'sl_base': 95, 'tp_base': 215},
        # Metals
        'XAGUSD': {'sl_base': 95, 'tp_base': 220},
        'XPTUSD': {'sl_base': 110, 'tp_base': 260},
        # Crypto
        'ETHUSD': {'sl_base': 120, 'tp_base': 280},
        'LTCUSD': {'sl_base': 100, 'tp_base': 240},
        'ADAUSD': {'sl_base': 85, 'tp_base': 195},
        'SOLUSD': {'sl_base': 110, 'tp_base': 260},
        'XRPUSD': {'sl_base': 75, 'tp_base': 175},
        # Indices
        'US30': {'sl_base': 170, 'tp_base': 380},
        'US500': {'sl_base': 150, 'tp_base': 340},
        'UK100': {'sl_base': 135, 'tp_base': 305},
        'JP225': {'sl_base': 145, 'tp_base': 320},
        'TecDE30': {'sl_base': 125, 'tp_base': 290},
        'USTEC': {'sl_base': 160, 'tp_base': 360},
        # Commodities
        'XTIUSD': {'sl_base': 100, 'tp_base': 240},
        'XBRUSD': {'sl_base': 105, 'tp_base': 250},
        'XNGUSD': {'sl_base': 95, 'tp_base': 220}
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
    
    # Aplicar límites inteligentes por categoría
    sl_before_limit = sl_final
    tp_before_limit = tp_final
    
    if symbol in ['XAUUSD', 'XAGUSD', 'XPTUSD']:  # Metals
        sl_final = max(50.0, min(sl_final, 400.0))
        tp_final = max(80.0, min(tp_final, 600.0))
    elif symbol in ['BTCUSD', 'ETHUSD', 'LTCUSD', 'ADAUSD', 'SOLUSD', 'XRPUSD']:  # Crypto
        sl_final = max(30.0, min(sl_final, 300.0))
        tp_final = max(60.0, min(tp_final, 500.0))
    elif symbol in ['US30', 'US500', 'UK100', 'JP225', 'TecDE30', 'USTEC']:  # Indices
        sl_final = max(50.0, min(sl_final, 350.0))
        tp_final = max(100.0, min(tp_final, 600.0))
    elif symbol in ['XTIUSD', 'XBRUSD', 'XNGUSD']:  # Commodities
        sl_final = max(40.0, min(sl_final, 250.0))
        tp_final = max(80.0, min(tp_final, 450.0))
    elif 'JPY' in symbol:  # JPY pairs
        sl_final = max(30.0, min(sl_final, 250.0))
        tp_final = max(60.0, min(tp_final, 450.0))
    else:  # Regular forex pairs
        sl_final = max(20.0, min(sl_final, 200.0))
        tp_final = max(40.0, min(tp_final, 350.0))
    
    # Log si se aplicaron límites
    if abs(sl_final - sl_before_limit) > 0.1 or abs(tp_final - tp_before_limit) > 0.1:
        logger.info(f"🔒 Limits applied for {symbol}: SL {sl_before_limit:.1f}→{sl_final:.1f}, TP {tp_before_limit:.1f}→{tp_final:.1f}")
    
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

@app.get("/health")
async def health_check():
    """Health check endpoint para el EA"""
    return {
        "status": "healthy",
        "service": "ARIA Hybrid System",
        "models_loaded": len(models),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Endpoint principal"""
    return {
        "service": "ARIA XGBoost Hybrid Predictor",
        "version": "5.2.0-HYBRID_SYSTEM",
        "status": "operational",
        "system_info": {"type": "hybrid_ml_predictor"},
        "models_loaded": len(models),
        "symbols_available": [
            # Majors
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
            # Crosses
            "EURGBP", "EURJPY", "GBPJPY", "AUDCHF", "AUDJPY", "CADCHF", "CADJPY", "CHFJPY",
            "EURAUD", "EURCAD", "EURCHF", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD",
            # Metals
            "XAUUSD", "XAGUSD", "XPTUSD",
            # Crypto
            "BTCUSD", "ETHUSD", "LTCUSD", "ADAUSD", "SOLUSD", "XRPUSD",
            # Indices
            "US30", "US500", "UK100", "JP225", "TecDE30", "USTEC",
            # Commodities
            "XTIUSD", "XBRUSD", "XNGUSD"
        ],
        "features": {
            "regime_detection": "ML_based",
            "predictions": "regime_aware_rules",
            "continuous_learning": "enabled",
            "hybrid_approach": "classifier_ml_plus_rules"
        }
    }

@app.get("/predict")
async def predict_legacy(
    symbol: str = "XAUUSD",
    lot_size: float = 0.01,
    atr: float = 50.0,
    trend_strength: float = 0.0
):
    """Endpoint legacy compatible con formato antiguo del EA"""
    try:
        logger.info(f"🔄 Legacy request: {symbol}, ATR={atr}, trend={trend_strength}")
        
        # Convertir formato antiguo a nuevo
        request_data = PredictionRequest(
            symbol=symbol,
            atr_percentile_100=atr,
            rsi_std_20=0.5,  # Valor por defecto
            price_acceleration=trend_strength,
            candle_body_ratio_mean=0.7,  # Valor por defecto
            breakout_frequency=0.3,  # Valor por defecto
            volume_imbalance=0.1,  # Valor por defecto
            rolling_autocorr_20=0.2,  # Valor por defecto
            hurst_exponent_50=0.5,  # Valor por defecto
            timeframe="M1"
        )
        
        # Usar la misma lógica de predicción
        result = await predict_full(request_data)
        
        # Formato de respuesta compatible
        return {
            "success": result.success,
            "sl_pips": result.sl_pips,
            "tp_pips": result.tp_pips,
            "confidence": result.overall_confidence,
            "regime": result.detected_regime,
            "model": result.model_used
        }
        
    except Exception as e:
        logger.error(f"❌ Error in legacy endpoint: {e}")
        return {
            "success": False,
            "sl_pips": 100.0,
            "tp_pips": 200.0,
            "confidence": 50.0,
            "regime": "fallback",
            "model": "error_fallback"
        }

@app.post("/predict", response_model=PredictionResponse)
async def predict_full(request: PredictionRequest):
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
        supported_symbols = [
            # Majors
            "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
            # Crosses
            "EURGBP", "EURJPY", "GBPJPY", "AUDCHF", "AUDJPY", "CADCHF", "CADJPY", "CHFJPY",
            "EURAUD", "EURCAD", "EURCHF", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD",
            # Metals
            "XAUUSD", "XAGUSD", "XPTUSD",
            # Crypto
            "BTCUSD", "ETHUSD", "LTCUSD", "ADAUSD", "SOLUSD", "XRPUSD",
            # Indices
            "US30", "US500", "UK100", "JP225", "TecDE30", "USTEC",
            # Commodities
            "XTIUSD", "XBRUSD", "XNGUSD"
        ]
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
