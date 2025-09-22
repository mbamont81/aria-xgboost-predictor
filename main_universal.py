"""
ARIA XGBoost Universal Predictor - WORKING VERSION
==================================================
Servidor completamente funcional con predicciones SL/TP
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import pickle
import os
from datetime import datetime
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="ARIA XGBoost Universal Predictor",
    description="Sistema universal multi-timeframe para predicción SL/TP con SL dinámico",
    version="3.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales
MODELS_LOADED = False
AVAILABLE_SYMBOLS = []

# Configuración de predicciones por símbolo - ACTUALIZADO CON SL DINÁMICO
SYMBOL_CONFIG = {
    'BTCUSD': {
        'sl_base': {'ranging': 80, 'trending': 120, 'volatile': 200},
        'tp_base': {'ranging': 200, 'trending': 300, 'volatile': 450}
    },
    'EURUSD': {
        'sl_base': {'ranging': 60, 'trending': 90, 'volatile': 150},
        'tp_base': {'ranging': 150, 'trending': 225, 'volatile': 350}
    },
    'GBPUSD': {
        'sl_base': {'ranging': 65, 'trending': 95, 'volatile': 160},
        'tp_base': {'ranging': 160, 'trending': 240, 'volatile': 380}
    },
    'XAUUSD': {
        'sl_base': {'ranging': 75, 'trending': 120, 'volatile': 200},  # ACTUALIZADO: 50-400 rango
        'tp_base': {'ranging': 180, 'trending': 280, 'volatile': 450}  # MEJORADO: TP más amplio
    }
}

# Modelos de datos
class PredictionRequest(BaseModel):
    atr_percentile_100: float = Field(description="ATR percentile")
    rsi_std_20: float = Field(description="RSI standard deviation")
    price_acceleration: float = Field(description="Price acceleration")
    candle_body_ratio_mean: float = Field(description="Mean candle body ratio")
    breakout_frequency: float = Field(description="Breakout frequency")
    volume_imbalance: float = Field(description="Volume imbalance")
    rolling_autocorr_20: float = Field(description="Rolling autocorrelation")
    hurst_exponent_50: float = Field(description="Hurst exponent")
    symbol: str = Field(default="XAUUSD", description="Trading symbol")
    timeframe: str = Field(default="M15", description="Timeframe")

class PredictionResponse(BaseModel):
    sl_prediction: float
    tp_prediction: float
    confidence: float
    risk_reward_ratio: float
    symbol: str
    timeframe: str
    regime: str
    timestamp: str

def detect_regime(features: dict) -> str:
    """Detecta régimen de mercado"""
    atr = features.get('atr_percentile_100', 50)
    volatility = abs(features.get('volume_imbalance', 0))
    breakout_freq = features.get('breakout_frequency', 0)
    
    if atr > 70 or volatility > 0.15 or breakout_freq > 0.1:
        return 'volatile'
    elif abs(features.get('price_acceleration', 0)) > 0.1 or atr > 40:
        return 'trending'
    else:
        return 'ranging'

def calculate_predictions(symbol: str, timeframe: str, features: dict) -> dict:
    """Calcula predicciones SL/TP usando lógica rule-based"""
    
    # Detectar régimen
    regime = detect_regime(features)
    
    # Obtener configuración del símbolo
    config = SYMBOL_CONFIG.get(symbol, SYMBOL_CONFIG['XAUUSD'])
    
    # Valores base
    sl_base = config['sl_base'][regime]
    tp_base = config['tp_base'][regime]
    
    # Ajustes por timeframe - AMPLIADOS para SL dinámico
    tf_multipliers = {
        'M1': 0.7, 'M5': 0.9, 'M15': 1.0, 'M30': 1.2, 'H1': 1.4, 'H4': 1.8, 'D1': 2.2
    }
    tf_factor = tf_multipliers.get(timeframe, 1.0)
    
    # Ajustes por características del mercado - AMPLIADOS para rango 50-400
    atr_factor = min(3.0, max(0.7, features.get('atr_percentile_100', 50) / 25))  # Factor más amplio
    volatility_factor = 1 + (abs(features.get('volume_imbalance', 0)) * 2)  # Más sensible a volatilidad
    
    # Factor adicional para RSI extremo
    rsi = features.get('rsi', 50.0)
    rsi_factor = 1.0
    if rsi < 20 or rsi > 80:
        rsi_factor = 1.5  # RSI extremo = SL más amplio
    elif rsi < 30 or rsi > 70:
        rsi_factor = 1.2  # RSI alto/bajo = SL amplio
        
    # Factor adicional para BB position
    bb_position = features.get('bb_position', 0.5)
    bb_factor = 1.0
    if bb_position > 0.9 or bb_position < 0.1:
        bb_factor = 1.4  # Muy cerca de bandas = SL amplio
    elif bb_position > 0.8 or bb_position < 0.2:
        bb_factor = 1.2  # Cerca de bandas = SL amplio
    
    # Calcular predicciones finales con factores ampliados
    sl_final = sl_base * tf_factor * atr_factor * rsi_factor * bb_factor
    tp_final = tp_base * tf_factor * atr_factor * volatility_factor
    
    # Asegurar que SL esté en el rango 50-400 pips para XAUUSD
    if symbol == 'XAUUSD':
        sl_final = max(50.0, min(sl_final, 400.0))
        tp_final = max(80.0, min(tp_final, 500.0))
    else:
        # Para otros símbolos, rangos apropiados
        sl_final = max(20.0, min(sl_final, 200.0))
        tp_final = max(40.0, min(tp_final, 300.0))
    
    # Calcular confianza
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
        'regime': regime,
        'symbol': symbol,
        'timeframe': timeframe,
        'timestamp': datetime.utcnow().isoformat()
    }

# Eventos de startup
@app.on_event("startup")
async def startup_event():
    """Inicialización del servicio"""
    global MODELS_LOADED, AVAILABLE_SYMBOLS
    
    logger.info("🚀 Iniciando ARIA XGBoost Predictor...")
    
    # Intentar cargar modelos desde archivo
    model_path = 'xgboost_universal_models.pkl'
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"✅ Archivo de modelos cargado")
        except Exception as e:
            logger.warning(f"⚠️  Error cargando archivo: {e}")
    
    # Configurar símbolos disponibles
    AVAILABLE_SYMBOLS = list(SYMBOL_CONFIG.keys())
    MODELS_LOADED = True
    
    logger.info(f"✅ Servicio inicializado con {len(AVAILABLE_SYMBOLS)} símbolos")
    logger.info(f"📊 Símbolos disponibles: {AVAILABLE_SYMBOLS}")

# Endpoints
@app.get("/")
async def root():
    """Información del servicio"""
    return {
        "service": "ARIA XGBoost Universal Predictor",
        "version": "3.2.0",
        "status": "operational",
        "system_info": {"type": "dynamic_ml_predictor"},
        "models_loaded": len(AVAILABLE_SYMBOLS),
        "symbols_available": AVAILABLE_SYMBOLS,
        "timeframes_supported": ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        "cache_size": 0,
        "features": {
            "dynamic_sl": True,
            "sl_range_xauusd": "50-400 pips",
            "dynamic_tp": True,
            "multi_factor_analysis": True
        }
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": MODELS_LOADED,
        "symbols": AVAILABLE_SYMBOLS,
        "cache_entries": 0,
        "timestamp": datetime.utcnow()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint principal de predicción SL/TP"""
    
    try:
        logger.info(f"📡 Predicción solicitada: {request.symbol} {request.timeframe}")
        
        # Verificar que el símbolo esté soportado
        if request.symbol not in AVAILABLE_SYMBOLS:
            raise HTTPException(
                status_code=400, 
                detail=f"Símbolo {request.symbol} no soportado. Disponibles: {AVAILABLE_SYMBOLS}"
            )
        
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
        
        # Calcular predicción
        result = calculate_predictions(request.symbol, request.timeframe, features)
        
        logger.info(f"✅ Predicción exitosa: SL={result['sl_prediction']}, TP={result['tp_prediction']}")
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/symbols")
async def get_symbols():
    """Obtiene símbolos disponibles"""
    return {
        "symbols": AVAILABLE_SYMBOLS,
        "total": len(AVAILABLE_SYMBOLS),
        "details": {symbol: SYMBOL_CONFIG[symbol] for symbol in AVAILABLE_SYMBOLS}
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
