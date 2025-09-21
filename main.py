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

# Inicializar FastAPI
app = FastAPI(
    title="Aria Regime-Aware XGBoost API",
    description="API para predicciones XGBoost especializadas por régimen de mercado",
    version="1.0.0"
)

# Estructura de request
class PredictionRequest(BaseModel):
    atr_percentile_100: float
    rsi_std_20: float
    price_acceleration: float
    candle_body_ratio_mean: float
    breakout_frequency: float
    volume_imbalance: float
    rolling_autocorr_20: float
    hurst_exponent_50: float
    symbol: str = "XAUUSD"  # Opcional

# Estructura de response
class PredictionResponse(BaseModel):
    success: bool
    detected_regime: str
    regime_confidence: float
    regime_weights: dict
    sl_pips: float
    tp_pips: float
    overall_confidence: float
    model_used: str
    processing_time_ms: float
    debug_info: dict

# Variables globales para modelos
models = {}
feature_names = [
    "atr_percentile_100", "rsi_std_20", "price_acceleration",
    "candle_body_ratio_mean", "breakout_frequency", "volume_imbalance", 
    "rolling_autocorr_20", "hurst_exponent_50"
]

def load_models():
    """Cargar todos los modelos al iniciar la aplicación"""
    global models
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        logger.error(f"Directorio {models_dir} no encontrado")
        return False
    
    try:
        # Cargar clasificador de regímenes
        models['regime_classifier'] = joblib.load(f"{models_dir}/regime_classifier.pkl")
        logger.info("Clasificador de regímenes cargado")
        
        # Cargar modelos especializados
        regimes = ['volatile', 'ranging', 'trending']
        for regime in regimes:
            sl_path = f"{models_dir}/xgb_{regime}_sl.pkl"
            tp_path = f"{models_dir}/xgb_{regime}_tp.pkl"
            
            if os.path.exists(sl_path) and os.path.exists(tp_path):
                models[f'{regime}_sl'] = joblib.load(sl_path)
                models[f'{regime}_tp'] = joblib.load(tp_path)
                logger.info(f"Modelos {regime} cargados")
            else:
                logger.warning(f"Modelos {regime} no encontrados")
        
        logger.info(f"Total modelos cargados: {len(models)}")
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelos: {e}")
        return False

def validate_features(features: np.ndarray) -> bool:
    """Validar que las features estén en rangos razonables"""
    try:
        # Verificar que no hay NaN o infinitos
        if np.isnan(features).any() or np.isinf(features).any():
            return False
        
        # Verificar rangos básicos
        if features[0] < 0 or features[0] > 100:  # atr_percentile_100
            return False
        if features[1] < 0 or features[1] > 50:   # rsi_std_20
            return False
        
        return True
    except:
        return False

@app.on_event("startup")
async def startup_event():
    """Cargar modelos al iniciar la aplicación"""
    success = load_models()
    if not success:
        logger.error("Error al cargar modelos - API podría no funcionar correctamente")

@app.get("/")
async def root():
    """Endpoint de prueba"""
    return {
        "message": "Aria Regime-Aware XGBoost API",
        "status": "active",
        "models_loaded": len(models),
        "available_endpoints": ["/predict", "/health", "/models-info"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models-info")
async def models_info():
    """Información sobre modelos cargados"""
    return {
        "models_loaded": list(models.keys()),
        "feature_names": feature_names,
        "supported_regimes": ["volatile", "ranging", "trending"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_regime_sltp(request: PredictionRequest):
    """Endpoint principal para predicciones regime-aware"""
    start_time = datetime.now()
    
    try:
        # Verificar que los modelos estén cargados
        if not models:
            raise HTTPException(status_code=500, detail="Modelos no cargados")
        
        # Preparar features
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
        
        # Validar features
        if not validate_features(features_array[0]):
            raise HTTPException(status_code=400, detail="Features fuera de rango válido")
        
        # 1. Predecir régimen
        regime_classifier = models.get('regime_classifier')
        if not regime_classifier:
            raise HTTPException(status_code=500, detail="Clasificador de regímenes no disponible")
        
        regime_probs = regime_classifier.predict_proba(features_array)[0]
        regime_classes = regime_classifier.classes_
        
        # Crear diccionario de weights
        regime_weights = {}
        for i, regime_class in enumerate(regime_classes):
            regime_weights[regime_class] = float(regime_probs[i])
        
        # Régimen detectado y confianza
        detected_regime = regime_classes[np.argmax(regime_probs)]
        regime_confidence = float(np.max(regime_probs))
        
        # 2. Predecir SL/TP con modelo especializado
        sl_model = models.get(f'{detected_regime}_sl')
        tp_model = models.get(f'{detected_regime}_tp')
        
        if not sl_model or not tp_model:
            raise HTTPException(status_code=500, detail=f"Modelos para régimen {detected_regime} no disponibles")
        
        # Predicciones individuales
        sl_pred = float(sl_model.predict(features_array)[0])
        tp_pred = float(tp_model.predict(features_array)[0])
        
        # 3. Weighted ensemble (opcional - por ahora usamos modelo detectado)
        # En el futuro se puede implementar ensemble ponderado
        
        # 4. Post-procesamiento y validación
        sl_pips = max(5.0, min(sl_pred, 200.0))  # Límites razonables
        tp_pips = max(10.0, min(tp_pred, 500.0))
        
        # Asegurar ratio mínimo TP/SL
        if tp_pips / sl_pips < 1.2:
            tp_pips = sl_pips * 1.5
        
        # Calcular tiempo de procesamiento
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # 5. Preparar response
        response = PredictionResponse(
            success=True,
            detected_regime=detected_regime,
            regime_confidence=round(regime_confidence * 100, 2),
            regime_weights={k: round(v, 3) for k, v in regime_weights.items()},
            sl_pips=round(sl_pips, 1),
            tp_pips=round(tp_pips, 1),
            overall_confidence=round(regime_confidence * 100, 2),
            model_used=f"regime_aware_xgboost_{detected_regime}",
            processing_time_ms=round(processing_time, 2),
            debug_info={
                "raw_sl_prediction": round(sl_pred, 2),
                "raw_tp_prediction": round(tp_pred, 2),
                "symbol": request.symbol,
                "feature_validation": "passed",
                "timestamp": end_time.isoformat()
            }
        )
        
        # Log para monitoring
        logger.info(f"Prediction: {detected_regime} (conf={regime_confidence:.2f}) SL={sl_pips} TP={tp_pips}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(requests: list[PredictionRequest]):
    """Endpoint para predicciones en lote"""
    results = []
    
    for request in requests:
        try:
            result = await predict_regime_sltp(request)
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e),
                "symbol": request.symbol
            })
    
    return {"results": results, "total_processed": len(results)}

# Endpoint de fallback para compatibilidad con EA actual
@app.post("/xgboost/predict_sltp")
async def predict_sltp_legacy(request: PredictionRequest):
    """Endpoint legacy para compatibilidad"""
    return await predict_regime_sltp(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)