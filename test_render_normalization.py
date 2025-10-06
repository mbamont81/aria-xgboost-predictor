#!/usr/bin/env python3
"""
Script para probar la normalización de símbolos en el servidor Render
Verifica que la funcionalidad de normalización esté funcionando correctamente
"""

import requests
import json
import time
from datetime import datetime

# URL base del servicio en Render
BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def test_service_status():
    """Probar el estado del servicio"""
    print("🔍 Probando estado del servicio...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Servicio activo: {data.get('message', 'N/A')}")
            print(f"📊 Modelos cargados: {data.get('models_loaded', 'N/A')}")
            print(f"🔄 Normalización: {data.get('symbol_normalization', 'N/A')}")
            print(f"🛠️ Endpoints disponibles: {data.get('available_endpoints', [])}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error conectando al servicio: {e}")
        return False

def test_symbol_normalization():
    """Probar el endpoint de normalización de símbolos"""
    print("\n🔍 Probando normalización de símbolos...")
    
    # Símbolos problemáticos que deberían normalizarse
    test_symbols = [
        "TECDE30",      # → GER30
        "USTEC",        # → NAS100
        "JP225",        # → JPN225
        "XTIUSD",       # → OILUSD
        "GOLD#",        # → XAUUSD
        "EURUSD.M",     # → EURUSD
        "GBPUSD_",      # → GBPUSD
        "USDJPYC",      # → USDJPY
        "AUDUSD",       # → AUDUSD (sin cambio)
        "NZDUSD"        # → NZDUSD (sin cambio)
    ]
    
    results = []
    
    for symbol in test_symbols:
        try:
            response = requests.get(f"{BASE_URL}/normalize-symbol/{symbol}", timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    original = data.get('original_symbol')
                    normalized = data.get('normalized_symbol')
                    changed = data.get('symbol_changed')
                    
                    status = "🔄" if changed else "✅"
                    print(f"{status} {original} → {normalized}")
                    
                    results.append({
                        'symbol': symbol,
                        'success': True,
                        'original': original,
                        'normalized': normalized,
                        'changed': changed
                    })
                else:
                    print(f"❌ Error normalizando {symbol}: {data.get('error', 'Unknown')}")
                    results.append({'symbol': symbol, 'success': False, 'error': data.get('error')})
            else:
                print(f"❌ Error HTTP {response.status_code} para {symbol}: {response.text}")
                results.append({'symbol': symbol, 'success': False, 'error': f"HTTP {response.status_code}"})
                
        except Exception as e:
            print(f"❌ Error probando {symbol}: {e}")
            results.append({'symbol': symbol, 'success': False, 'error': str(e)})
    
    return results

def test_prediction_with_normalization():
    """Probar predicciones con normalización automática"""
    print("\n🔍 Probando predicciones con normalización automática...")
    
    # Datos de prueba con símbolos problemáticos
    test_cases = [
        {
            "symbol": "TECDE30",  # Debería normalizarse a GER30
            "timeframe": "H1",
            "features": {
                "rsi": 45.5,
                "bb_position": 0.3,
                "macd_signal": 0.1,
                "atr_normalized": 0.02,
                "volume_ratio": 1.2,
                "price_change_1h": 0.005,
                "volatility": 0.015,
                "trend_strength": 0.6,
                "support_distance": 0.01,
                "resistance_distance": 0.02
            }
        },
        {
            "symbol": "GOLD#",  # Debería normalizarse a XAUUSD
            "timeframe": "H4",
            "features": {
                "rsi": 65.2,
                "bb_position": 0.8,
                "macd_signal": -0.2,
                "atr_normalized": 0.03,
                "volume_ratio": 0.9,
                "price_change_1h": -0.002,
                "volatility": 0.025,
                "trend_strength": 0.4,
                "support_distance": 0.015,
                "resistance_distance": 0.008
            }
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Caso de prueba {i}: {test_case['symbol']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_case,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                debug_info = data.get('debug_info', {})
                
                original_symbol = debug_info.get('original_symbol', 'N/A')
                normalized_symbol = debug_info.get('normalized_symbol', 'N/A')
                symbol_changed = debug_info.get('symbol_changed', False)
                
                print(f"✅ Predicción exitosa")
                print(f"   Original: {original_symbol}")
                print(f"   Normalizado: {normalized_symbol}")
                print(f"   Cambió: {'Sí' if symbol_changed else 'No'}")
                print(f"   SL: {data.get('stop_loss', 'N/A')}")
                print(f"   TP: {data.get('take_profit', 'N/A')}")
                print(f"   Régimen: {data.get('regime', 'N/A')}")
                
                results.append({
                    'test_case': i,
                    'symbol': test_case['symbol'],
                    'success': True,
                    'original_symbol': original_symbol,
                    'normalized_symbol': normalized_symbol,
                    'symbol_changed': symbol_changed,
                    'prediction': {
                        'stop_loss': data.get('stop_loss'),
                        'take_profit': data.get('take_profit'),
                        'regime': data.get('regime')
                    }
                })
                
            else:
                print(f"❌ Error HTTP {response.status_code}: {response.text}")
                results.append({
                    'test_case': i,
                    'symbol': test_case['symbol'],
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}"
                })
                
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            results.append({
                'test_case': i,
                'symbol': test_case['symbol'],
                'success': False,
                'error': str(e)
            })
    
    return results

def generate_report(normalization_results, prediction_results):
    """Generar reporte final de las pruebas"""
    print("\n" + "="*60)
    print("📋 REPORTE FINAL DE PRUEBAS")
    print("="*60)
    
    # Resumen de normalización
    norm_success = sum(1 for r in normalization_results if r.get('success', False))
    norm_total = len(normalization_results)
    norm_changed = sum(1 for r in normalization_results if r.get('changed', False))
    
    print(f"\n🔄 NORMALIZACIÓN DE SÍMBOLOS:")
    print(f"   ✅ Exitosas: {norm_success}/{norm_total}")
    print(f"   🔄 Normalizadas: {norm_changed}")
    print(f"   ❌ Fallidas: {norm_total - norm_success}")
    
    # Resumen de predicciones
    pred_success = sum(1 for r in prediction_results if r.get('success', False))
    pred_total = len(prediction_results)
    pred_normalized = sum(1 for r in prediction_results if r.get('symbol_changed', False))
    
    print(f"\n📊 PREDICCIONES CON NORMALIZACIÓN:")
    print(f"   ✅ Exitosas: {pred_success}/{pred_total}")
    print(f"   🔄 Con normalización: {pred_normalized}")
    print(f"   ❌ Fallidas: {pred_total - pred_success}")
    
    # Estado general
    overall_success = (norm_success == norm_total) and (pred_success == pred_total)
    status_emoji = "✅" if overall_success else "⚠️"
    
    print(f"\n{status_emoji} ESTADO GENERAL: {'TODAS LAS PRUEBAS EXITOSAS' if overall_success else 'ALGUNAS PRUEBAS FALLARON'}")
    
    print(f"\n⏰ Timestamp: {datetime.now().isoformat()}")
    print("="*60)

def main():
    """Función principal"""
    print("🚀 INICIANDO PRUEBAS DE NORMALIZACIÓN EN RENDER")
    print("="*60)
    
    # Esperar un poco para asegurar que el deployment esté activo
    print("⏳ Esperando 30 segundos para asegurar deployment...")
    time.sleep(30)
    
    # Probar estado del servicio
    if not test_service_status():
        print("❌ El servicio no está disponible. Abortando pruebas.")
        return
    
    # Probar normalización de símbolos
    normalization_results = test_symbol_normalization()
    
    # Probar predicciones con normalización
    prediction_results = test_prediction_with_normalization()
    
    # Generar reporte final
    generate_report(normalization_results, prediction_results)

if __name__ == "__main__":
    main()
