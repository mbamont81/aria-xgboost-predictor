
//+------------------------------------------------------------------+
//| SISTEMA DE CALIBRACIÓN DE CONFIDENCE BASADO EN DATOS REALES     |
//+------------------------------------------------------------------+

// Función para calibrar confidence basado en análisis de streamed_trades
double CalibrateXGBoostConfidence(double rawConfidence, string symbol, string marketRegime)
{
    double calibrationFactor = 1.0;
    
    // Factores de calibración basados en análisis de 3,662 trades reales

    // XAUUSD - Basado en análisis real
    if(symbol == "XAUUSD")
    {
        if(marketRegime == "trending") calibrationFactor = 0.90; // Avg conf 93.7% vs 64.4% win rate
    }

    // EURUSD - Basado en análisis real
    if(symbol == "EURUSD")
    {
    }

    // Gold - Basado en análisis real
    if(symbol == "Gold")
    {
        if(marketRegime == "trending") calibrationFactor = 0.80; // Avg conf 93.1% vs 28.6% win rate
    }

    // XAUUSD.s - Basado en análisis real
    if(symbol == "XAUUSD.s")
    {
        if(marketRegime == "trending") calibrationFactor = 1.10; // Avg conf 95.0% vs 75.9% win rate
    }

    // AUDUSD - Basado en análisis real
    if(symbol == "AUDUSD")
    {
    }

    // EURGBP - Basado en análisis real
    if(symbol == "EURGBP")
    {
    }

    // USDCHF - Basado en análisis real
    if(symbol == "USDCHF")
    {
    }

    // CHFJPY - Basado en análisis real
    if(symbol == "CHFJPY")
    {
    }

    // NZDUSD - Basado en análisis real
    if(symbol == "NZDUSD")
    {
    }

    // BTCUSD - Basado en análisis real
    if(symbol == "BTCUSD")
    {
        if(marketRegime == "trending") calibrationFactor = 0.80; // Avg conf 95.0% vs 50.0% win rate
    }

    // EURUSDb - Basado en análisis real
    if(symbol == "EURUSDb")
    {
    }

    // USDCAD - Basado en análisis real
    if(symbol == "USDCAD")
    {
    }

    // GBPUSD - Basado en análisis real
    if(symbol == "GBPUSD")
    {
    }

    // USDCNH - Basado en análisis real
    if(symbol == "USDCNH")
    {
    }

    // GOLD# - Basado en análisis real
    if(symbol == "GOLD#")
    {
        if(marketRegime == "trending") calibrationFactor = 0.80; // Avg conf 95.0% vs 45.5% win rate
    }

    // GBPAUDb - Basado en análisis real
    if(symbol == "GBPAUDb")
    {
    }

    // USTEC - Basado en análisis real
    if(symbol == "USTEC")
    {
    }

    // XAUUSD.p - Basado en análisis real
    if(symbol == "XAUUSD.p")
    {
    }

    // USDJPY.s - Basado en análisis real
    if(symbol == "USDJPY.s")
    {
    }

    
    // Aplicar calibración
    double calibratedConfidence = rawConfidence * calibrationFactor;
    
    // Log para debugging
    if(calibrationFactor != 1.0)
    {
        Print("🎯 Confidence calibrada: ", rawConfidence, "% → ", calibratedConfidence, 
              "% (", symbol, " ", marketRegime, " factor: ", calibrationFactor, ")");
    }
    
    return MathMin(calibratedConfidence, 100.0); // Cap a 100%
}

// USAR EN TU EA:
// double calibratedConf = CalibrateXGBoostConfidence(xgbResponse.confidence, _Symbol, xgbResponse.marketRegime);
// if(calibratedConf >= tu_threshold_actual) { /* usar XGBoost */ }
