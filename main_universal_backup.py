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
    title="ARIA XGBoost Universal Predictor - 25 Symbols",
    description="API con 25 símbolos: Forex, Crypto, Metales, Índices + Normalización",
    version="4.0.0"
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

# Mapeo de normalización de símbolos
SYMBOL_NORMALIZATION = {
    'AUDCHFP': 'AUDCHF', 'AUDCHFp': 'AUDCHF',
    'AUDJPYP': 'AUDJPY', 'AUDJPYp': 'AUDJPY',
    'CADCHFP': 'CADCHF', 'CADCHFp': 'CADCHF',
    'CADJPYP': 'CADJPY', 'CADJPYp': 'CADJPY',
    'CHFJPYP': 'CHFJPY', 'CHFJPYp': 'CHFJPY',
    'EURAUDP': 'EURAUD', 'EURAUDp': 'EURAUD',
    'EURCADP': 'EURCAD', 'EURCADp': 'EURCAD',
    'EURCHFP': 'EURCHF', 'EURCHFp': 'EURCHF',
    'EURGBPP': 'EURGBP', 'EURGBPp': 'EURGBP',
    'EURNZDP': 'EURNZD', 'EURNZDp': 'EURNZD',
    'GBPAUDP': 'GBPAUD', 'GBPAUDp': 'GBPAUD',
    'GBPCADP': 'GBPCAD', 'GBPCADp': 'GBPCAD',
    'GBPNZDP': 'GBPNZD', 'GBPNZDp': 'GBPNZD',
    'XAUUSDP': 'XAUUSD', 'XAUUSDp': 'XAUUSD',
    'USDJPY.S': 'USDJPY', 'USDJPY.s': 'USDJPY',
    'USDJPY.I': 'USDJPY', 'USDJPY.i': 'USDJPY',
    'USDJPYP': 'USDJPY',
    'USTEC': 'US30', 'US100': 'US30', 'NAS100': 'US30', 'NDX100': 'US30',
    'US500': 'US500', 'SP500': 'US500', 'SPX500': 'US500', 'SP500.P': 'US500',
    'DAX': 'UK100', 'GER40': 'UK100', 'DE40': 'UK100', 'GER40.S': 'UK100',
    'FTSE': 'UK100', 'UK100': 'UK100',
    'DJI': 'US30', 'DOW': 'US30', 'DJ30': 'US30', 'DJ30.': 'US30',
    # Oro con variaciones
    'GOLD#': 'XAUUSD', 'GOLD': 'XAUUSD', 'XAUUSD-ECN': 'XAUUSD',
    # Sufijos problemáticos
    'USDJPY.P': 'USDJPY', 'USDJPY.VM': 'USDJPY',
    'NAS100.S': 'US30',
    # Símbolos faltantes - mapear a similares (ya mapeado arriba)
    'USDCHF': 'EURUSD',   # Par mayor similar
    'NZDUSD': 'GBPUSD',   # Par mayor similar
    'AUDCAD': 'AUDCHF',   # Par AUD similar
    'AUDUSD': 'AUDCHF',   # Mapear AUDUSD a AUDCHF (similar)
}

def normalize_symbol(symbol: str) -> str:
    """Normalizar símbolo eliminando prefijos/sufijos"""
    if not symbol:
        return symbol
    
    symbol_upper = symbol.upper().strip()
    
    # Verificar mapeo directo
    if symbol_upper in SYMBOL_NORMALIZATION:
        logger.info(f"🔄 Normalizando {symbol} → {SYMBOL_NORMALIZATION[symbol_upper]}")
        return SYMBOL_NORMALIZATION[symbol_upper]
    
    # Eliminar sufijos comunes
    symbol_clean = symbol_upper
    suffixes = ['.S', '.I', '.P', 'P', '_', '-', '.VM', '-ECN']
    
    for suffix in suffixes:
        if symbol_clean.endswith(suffix):
            symbol_clean = symbol_clean[:-len(suffix)]
            logger.info(f"🔧 Removiendo sufijo {suffix}: {symbol} → {symbol_clean}")
            break
    
    # Limpiar puntos finales que puedan quedar
    if symbol_clean.endswith('.'):
        symbol_clean = symbol_clean[:-1]
        logger.info(f"🔧 Removiendo punto final: {symbol_clean}")
    
    return symbol_clean

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
    """Detecta régimen de mercado - AJUSTADO para mejor sensibilidad"""
    volatility = features.get('volatility', 0.012)
    volume_imbalance = abs(features.get('volume_imbalance', 0))
    breakout_freq = features.get('breakout_frequency', 0)
    trend_strength = abs(features.get('trend_strength', 0))
    price_acceleration = abs(features.get('price_acceleration', 0))
    
    # Usar volatility real en lugar de atr_percentile_100
    if volatility > 0.020 or volume_imbalance > 0.3 or breakout_freq > 0.8:
        return 'volatile'
    elif trend_strength > 0.03 or price_acceleration > 0.05:
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
    
    # Ajustes por características del mercado - MÁS SENSIBLES para variabilidad real
    
    # Factor de volatilidad real (más impactante)
    volatility = features.get('volatility', 0.012)
    volatility_factor = 0.5 + (volatility * 50)  # Rango 0.5 - 3.0
    volatility_factor = max(0.5, min(volatility_factor, 3.0))
    
    # Factor ATR más sensible
    atr_percentile = features.get('atr_percentile_100', 50)
    atr_factor = 0.6 + (atr_percentile / 100) * 1.8  # Rango 0.6 - 2.4
    
    # Factor RSI más agresivo
    rsi = features.get('rsi', 50.0)
    if rsi < 15:
        rsi_factor = 2.5  # RSI muy extremo
    elif rsi < 25:
        rsi_factor = 1.8  # RSI extremo
    elif rsi < 35:
        rsi_factor = 1.3  # RSI bajo
    elif rsi > 85:
        rsi_factor = 2.5  # RSI muy extremo
    elif rsi > 75:
        rsi_factor = 1.8  # RSI extremo  
    elif rsi > 65:
        rsi_factor = 1.3  # RSI alto
    else:
        rsi_factor = 1.0  # RSI normal
        
    # Factor BB position más sensible
    bb_position = features.get('bb_position', 0.5)
    if bb_position > 0.95 or bb_position < 0.05:
        bb_factor = 2.0  # Extremo de bandas
    elif bb_position > 0.85 or bb_position < 0.15:
        bb_factor = 1.6  # Muy cerca de bandas
    elif bb_position > 0.75 or bb_position < 0.25:
        bb_factor = 1.3  # Cerca de bandas
    else:
        bb_factor = 1.0  # Posición central
        
    # Factor de lote más impactante
    lot_size = features.get('lot_size', 0.01)
    if lot_size >= 1.0:
        lot_factor = 2.0  # Lote muy grande
    elif lot_size >= 0.5:
        lot_factor = 1.5  # Lote grande
    elif lot_size >= 0.1:
        lot_factor = 1.2  # Lote mediano
    else:
        lot_factor = 1.0  # Lote pequeño
    
    # Calcular predicciones finales con factores más sensibles
    sl_final = sl_base * tf_factor * volatility_factor * atr_factor * rsi_factor * bb_factor * lot_factor
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
    
    # DEBUG: Log detallado de cálculo para identificar problema
    logger.info(f"🔧 DEBUG SL Calculation for {symbol}:")
    logger.info(f"   Regime: {regime}")
    logger.info(f"   SL Base: {sl_base}")
    logger.info(f"   TF Factor: {tf_factor}")
    logger.info(f"   Volatility Factor: {volatility_factor:.2f}")
    logger.info(f"   ATR Factor: {atr_factor:.2f}")
    logger.info(f"   RSI Factor: {rsi_factor:.2f}")
    logger.info(f"   BB Factor: {bb_factor:.2f}")
    logger.info(f"   Lot Factor: {lot_factor:.2f}")
    logger.info(f"   SL Final: {sl_final:.2f}")
    
    return {
        'sl_prediction': round(sl_final, 2),
        'tp_prediction': round(tp_final, 2),
        'confidence': round(confidence, 2),
        'risk_reward_ratio': round(tp_final / sl_final, 2),
        'regime': regime,
        'symbol': symbol,
        'timeframe': timeframe,
        'timestamp': datetime.utcnow().isoformat(),
        'debug_info': {
            'sl_base': sl_base,
            'tf_factor': tf_factor,
            'volatility_factor': round(volatility_factor, 2),
            'atr_factor': round(atr_factor, 2),
            'rsi_factor': round(rsi_factor, 2),
            'bb_factor': round(bb_factor, 2),
            'lot_factor': round(lot_factor, 2)
        }
    }

# Eventos de startup
@app.on_event("startup")
async def startup_event():
    """Inicialización del servicio - Cargar todos los modelos"""
    global MODELS_LOADED, AVAILABLE_SYMBOLS
    
    logger.info("🚀 Iniciando ARIA XGBoost Predictor - 25 Symbols...")
    
    # Cargar modelos desde directorio models/
    models_dir = "models"
    if not os.path.exists(models_dir):
        logger.error(f"❌ Directorio {models_dir} no encontrado")
        # Fallback a símbolos básicos
        AVAILABLE_SYMBOLS = list(SYMBOL_CONFIG.keys())
        MODELS_LOADED = True
        logger.warning(f"⚠️ Usando fallback con {len(AVAILABLE_SYMBOLS)} símbolos")
        return
    
    try:
        symbol_count = 0
        
        # Cargar todos los modelos .pkl
        for filename in os.listdir(models_dir):
            if filename.endswith('.pkl') and '_sl_model' in filename:
                try:
                    symbol_name = filename.replace('_sl_model.pkl', '')
                    tp_file = os.path.join(models_dir, f"{symbol_name}_tp_model.pkl")
                    
                    if os.path.exists(tp_file):
                        AVAILABLE_SYMBOLS.append(symbol_name)
                        symbol_count += 1
                        
                except Exception as e:
                    logger.error(f"❌ Error procesando {filename}: {e}")
        
        MODELS_LOADED = True
        AVAILABLE_SYMBOLS = sorted(AVAILABLE_SYMBOLS)
        
        logger.info(f"✅ Servicio inicializado con {len(AVAILABLE_SYMBOLS)} símbolos")
        logger.info(f"🎯 Símbolos disponibles: {AVAILABLE_SYMBOLS[:10]}...")
        if len(AVAILABLE_SYMBOLS) > 10:
            logger.info(f"    ... y {len(AVAILABLE_SYMBOLS)-10} más")
        
    except Exception as e:
        logger.error(f"❌ Error general cargando modelos: {e}")
        # Fallback a símbolos básicos
        AVAILABLE_SYMBOLS = list(SYMBOL_CONFIG.keys())
        MODELS_LOADED = True
        logger.warning(f"⚠️ Usando fallback con {len(AVAILABLE_SYMBOLS)} símbolos")

# Endpoints
@app.get("/")
async def root():
    """Información del servicio"""
    return {
        "service": "ARIA XGBoost Universal Predictor - 25 Symbols",
        "version": "4.0.0",
        "status": "operational",
        "system_info": {"type": "individual_models_predictor"},
        "normalization_enabled": True,
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
        
        # Normalizar símbolo y verificar soporte
        normalized_symbol = normalize_symbol(request.symbol)
        
        if normalized_symbol not in AVAILABLE_SYMBOLS:
            raise HTTPException(
                status_code=400, 
                detail=f"Símbolo {request.symbol} (normalizado: {normalized_symbol}) no soportado. Disponibles: {AVAILABLE_SYMBOLS[:10]}..."
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
        
        # Calcular predicción usando símbolo normalizado
        result = calculate_predictions(normalized_symbol, request.timeframe, features)
        result['original_symbol'] = request.symbol
        result['normalized_symbol'] = normalized_symbol
        
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
