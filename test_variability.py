#!/usr/bin/env python3
"""
Test para verificar si los valores SL/TP ahora son únicos y variables
"""

import requests
import time
from datetime import datetime

BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def test_prediction_variability():
    """Test si las predicciones ahora tienen variabilidad"""
    print("🧪 TESTING VARIABILIDAD DE PREDICCIONES SL/TP")
    print("=" * 60)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Datos de prueba fijos (para ver si outputs varían)
    test_data = {
        "symbol": "XAUUSD",
        "timeframe": "H1",
        "atr_percentile_100": 45.5,
        "rsi_std_20": 12.3,
        "price_acceleration": 0.02,
        "candle_body_ratio_mean": 0.65,
        "breakout_frequency": 0.15,
        "volume_imbalance": 0.08,
        "rolling_autocorr_20": 0.25,
        "hurst_exponent_50": 0.52
    }
    
    results = []
    
    print("🔍 Realizando 5 requests idénticos para verificar variabilidad...")
    
    for i in range(5):
        try:
            print(f"\n📊 Test {i+1}/5:")
            
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                sl = data.get('sl_prediction', 0)
                tp = data.get('tp_prediction', 0)
                confidence = data.get('confidence', 0)
                regime = data.get('regime', 'N/A')
                
                print(f"   ✅ SUCCESS")
                print(f"   📉 SL: {sl}")
                print(f"   📈 TP: {tp}")
                print(f"   🎯 Confidence: {confidence}")
                print(f"   🌊 Regime: {regime}")
                
                results.append({
                    'status': 'success',
                    'sl': sl,
                    'tp': tp,
                    'confidence': confidence,
                    'regime': regime
                })
                
            else:
                print(f"   ❌ ERROR: {response.status_code}")
                print(f"   📝 Response: {response.text[:100]}")
                
                results.append({
                    'status': 'error',
                    'code': response.status_code
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                'status': 'exception',
                'error': str(e)
            })
        
        time.sleep(2)  # 2 segundos entre requests para diferentes timestamps
    
    return results

def analyze_variability(results):
    """Analizar la variabilidad de los resultados"""
    print(f"\n📊 ANÁLISIS DE VARIABILIDAD:")
    print("=" * 50)
    
    successful_results = [r for r in results if r['status'] == 'success']
    
    if len(successful_results) < 2:
        print("❌ Insuficientes resultados exitosos para análizar variabilidad")
        return
    
    # Extraer valores
    sl_values = [r['sl'] for r in successful_results]
    tp_values = [r['tp'] for r in successful_results]
    confidence_values = [r['confidence'] for r in successful_results]
    
    print(f"📈 VALORES SL:")
    for i, sl in enumerate(sl_values, 1):
        print(f"   {i}. {sl}")
    
    print(f"\n📈 VALORES TP:")
    for i, tp in enumerate(tp_values, 1):
        print(f"   {i}. {tp}")
    
    print(f"\n📈 VALORES CONFIDENCE:")
    for i, conf in enumerate(confidence_values, 1):
        print(f"   {i}. {conf}")
    
    # Análisis de variabilidad
    sl_unique = len(set(sl_values))
    tp_unique = len(set(tp_values))
    conf_unique = len(set(confidence_values))
    
    print(f"\n🔍 ANÁLISIS DE UNICIDAD:")
    print(f"   • SL únicos: {sl_unique}/{len(sl_values)} ({sl_unique/len(sl_values)*100:.1f}%)")
    print(f"   • TP únicos: {tp_unique}/{len(tp_values)} ({tp_unique/len(tp_values)*100:.1f}%)")
    print(f"   • Confidence únicos: {conf_unique}/{len(confidence_values)} ({conf_unique/len(confidence_values)*100:.1f}%)")
    
    # Calcular variación
    if len(sl_values) > 1:
        sl_min, sl_max = min(sl_values), max(sl_values)
        tp_min, tp_max = min(tp_values), max(tp_values)
        
        sl_variation = ((sl_max - sl_min) / sl_min * 100) if sl_min > 0 else 0
        tp_variation = ((tp_max - tp_min) / tp_min * 100) if tp_min > 0 else 0
        
        print(f"\n📊 RANGO DE VARIACIÓN:")
        print(f"   • SL: {sl_min:.2f} - {sl_max:.2f} (±{sl_variation:.2f}%)")
        print(f"   • TP: {tp_min:.2f} - {tp_max:.2f} (±{tp_variation:.2f}%)")
    
    # Interpretación
    print(f"\n💡 INTERPRETACIÓN:")
    print("-" * 30)
    
    if sl_unique == len(sl_values) and tp_unique == len(tp_values):
        print("🎉 PERFECTO: Todos los valores son únicos!")
        print("✅ Variabilidad temporal funcionando correctamente")
        print("✅ Cada trade tiene predicciones únicas")
    elif sl_unique > len(sl_values) * 0.8:
        print("✅ BUENO: Alta variabilidad en predicciones")
        print("✅ Sistema funcionando correctamente")
    elif sl_unique > len(sl_values) * 0.5:
        print("⚠️ MODERADO: Alguna variabilidad pero puede mejorar")
        print("🔧 Considerar aumentar factores de granularidad")
    else:
        print("❌ PROBLEMA: Baja variabilidad, valores aún repetitivos")
        print("🔧 Necesita ajustes adicionales en factores")

def main():
    print("🔍 VERIFICACIÓN DE VARIABILIDAD SL/TP")
    print("=" * 60)
    
    # Test variabilidad
    results = test_prediction_variability()
    
    # Analizar resultados
    analyze_variability(results)
    
    print(f"\n🎯 CONCLUSIÓN:")
    print("=" * 30)
    
    successful = sum(1 for r in results if r['status'] == 'success')
    
    if successful >= 3:
        print("✅ Sistema funcionando correctamente")
        print("✅ Predicciones generándose exitosamente")
        print("✅ Listo para verificar variabilidad")
    else:
        print("⚠️ Sistema aún propagándose")
        print("🔄 Reintentar en unos minutos")

if __name__ == "__main__":
    main()

