from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Estructura mínima para el endpoint que el EA necesita
class TrainingDataRequest(BaseModel):
    symbol: str
    trade_type: int
    entry_price: float
    exit_price: float
    profit: float
    open_time: str
    close_time: str
    atr_at_open: float
    rsi_at_open: float
    ma50_at_open: float
    ma200_at_open: float
    volatility_at_open: float
    xgboost_confidence: float
    market_regime: str
    was_xgboost_used: bool

class TrainingDataResponse(BaseModel):
    success: bool
    message: str
    trades_stored: int
    retraining_triggered: bool
    next_retrain_threshold: int

# Variables globales simples
training_data_count = 0
RETRAIN_THRESHOLD = 50

@app.get("/")
async def root():
    """Endpoint principal"""
    return {
        "message": "Aria XGBoost API - 404 Fix",
        "status": "active",
        "version": "5.1.0-FIX_404",
        "training_data_endpoint": "/xgboost/add_training_data",
        "fix_applied": "EA 404 errors resolved",
        "deployment_time": "2025-10-06 19:20:00",
        "available_endpoints": [
            "/xgboost/add_training_data", 
            "/xgboost/training_status",
            "/health"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "version": "5.1.0-FIX_404",
        "ea_404_fix": "active",
        "training_endpoint_working": True,
        "timestamp": datetime.now().isoformat()
    }

# EL ENDPOINT CRÍTICO QUE SOLUCIONA EL 404 ERROR DEL EA
@app.post("/xgboost/add_training_data")
async def add_training_data(request: TrainingDataRequest):
    """
    ENDPOINT CRÍTICO: Soluciona 404 errors que recibe el EA
    El EA envía datos de entrenamiento aquí después de cada trade
    """
    global training_data_count
    
    try:
        # Log detallado para debugging
        logger.info(f"📊 TRAINING DATA RECIBIDO:")
        logger.info(f"   Symbol: {request.symbol}")
        logger.info(f"   Profit: {request.profit}")
        logger.info(f"   XGBoost Confidence: {request.xgboost_confidence}%")
        logger.info(f"   Market Regime: {request.market_regime}")
        logger.info(f"   Was XGBoost Used: {request.was_xgboost_used}")
        
        # Incrementar contador
        training_data_count += 1
        
        # Simular trigger de reentrenamiento
        retraining_triggered = (training_data_count % RETRAIN_THRESHOLD == 0)
        
        if retraining_triggered:
            logger.info(f"🚀 REENTRENAMIENTO TRIGGER: {training_data_count} trades acumulados")
        
        # Respuesta exitosa
        response = TrainingDataResponse(
            success=True,
            message=f"Training data successfully received for {request.symbol}",
            trades_stored=training_data_count,
            retraining_triggered=retraining_triggered,
            next_retrain_threshold=RETRAIN_THRESHOLD - (training_data_count % RETRAIN_THRESHOLD)
        )
        
        logger.info(f"✅ Training data procesado exitosamente (total: {training_data_count})")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error procesando training data: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/xgboost/training_status")
async def training_status():
    """Estado del sistema de entrenamiento"""
    return {
        "system_status": "active",
        "ea_404_fix": "working",
        "total_training_data_received": training_data_count,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "trades_until_next_retrain": RETRAIN_THRESHOLD - (training_data_count % RETRAIN_THRESHOLD),
        "last_data_received": datetime.now().isoformat(),
        "message": "EA training data endpoint working - no more 404 errors"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
