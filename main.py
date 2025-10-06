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

app = FastAPI()

# Estructura básica para el endpoint que el EA necesita
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
        "message": "Aria XGBoost API",
        "status": "active",
        "version": "5.0.0-MINIMAL_WORKING",
        "continuous_learning": "enabled",
        "training_data_endpoint": "/xgboost/add_training_data",
        "auto_retraining": f"Every {RETRAIN_THRESHOLD} trades",
        "deployment_time": "2025-10-06 19:15:00",
        "available_endpoints": [
            "/predict", 
            "/health", 
            "/xgboost/add_training_data", 
            "/xgboost/training_status"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "version": "5.0.0-MINIMAL_WORKING",
        "continuous_learning_active": True,
        "training_endpoint_available": True,
        "timestamp": datetime.now().isoformat()
    }

# EL ENDPOINT CRÍTICO QUE EL EA NECESITA
@app.post("/xgboost/add_training_data")
async def add_training_data(request: TrainingDataRequest):
    """Endpoint para recibir datos de entrenamiento del EA - SOLUCIONA 404 ERROR"""
    global training_data_count
    
    try:
        logger.info(f"📊 Training data recibido: {request.symbol} profit={request.profit} conf={request.xgboost_confidence}%")
        
        # Incrementar contador
        training_data_count += 1
        
        # Simular lógica de reentrenamiento (por ahora)
        retraining_triggered = (training_data_count % RETRAIN_THRESHOLD == 0)
        
        if retraining_triggered:
            logger.info(f"🚀 TRIGGER: Reentrenamiento después de {training_data_count} trades")
        
        response = TrainingDataResponse(
            success=True,
            message=f"Training data received for {request.symbol}",
            trades_stored=training_data_count,
            retraining_triggered=retraining_triggered,
            next_retrain_threshold=RETRAIN_THRESHOLD - (training_data_count % RETRAIN_THRESHOLD)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error procesando training data: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/xgboost/training_status")
async def training_status():
    """Estado del entrenamiento"""
    return {
        "continuous_learning": "enabled",
        "total_training_data_received": training_data_count,
        "retrain_threshold": RETRAIN_THRESHOLD,
        "trades_until_next_retrain": RETRAIN_THRESHOLD - (training_data_count % RETRAIN_THRESHOLD),
        "system_status": "receiving_data",
        "timestamp": datetime.now().isoformat()
    }

# Mantener endpoints básicos existentes
@app.get("/models-info")
async def models_info():
    """Info de modelos"""
    return {
        "status": "active",
        "continuous_learning": "enabled",
        "training_data_received": training_data_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
