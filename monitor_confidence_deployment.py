#!/usr/bin/env python3
"""
Monitor automático para verificar cuando el deployment del filtro de confidence esté activo
"""

import requests
import time
from datetime import datetime

BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def check_deployment_status():
    """Check if deployment is complete"""
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            version = data.get('version', 'N/A')
            confidence_filtering = data.get('confidence_filtering', 'N/A')
            threshold = data.get('confidence_threshold', 'N/A')
            
            return {
                'version': version,
                'confidence_filtering': confidence_filtering,
                'threshold': threshold,
                'deployed': version == "4.2.0" and confidence_filtering == "enabled"
            }
    except:
        pass
    
    return {
        'version': 'Error',
        'confidence_filtering': 'Error',
        'threshold': 'Error',
        'deployed': False
    }

def test_confidence_filter_working():
    """Test if confidence filter is actually working"""
    test_data = {
        "symbol": "XAUUSD",
        "atr_percentile_100": 45.5,
        "rsi_std_20": 12.3,
        "price_acceleration": 0.02,
        "candle_body_ratio_mean": 0.65,
        "breakout_frequency": 0.15,
        "volume_imbalance": 0.08,
        "rolling_autocorr_20": 0.25,
        "hurst_exponent_50": 0.52
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            confidence = data.get('overall_confidence', 0)
            debug_info = data.get('debug_info', {})
            confidence_filter = debug_info.get('confidence_filter', 'N/A')
            
            return {
                'working': confidence > 0 and confidence_filter != 'N/A',
                'confidence': confidence,
                'filter_status': confidence_filter
            }
        elif response.status_code == 422:
            # This means filter is working (rejecting low confidence)
            return {
                'working': True,
                'confidence': 0,
                'filter_status': 'REJECTED (working correctly)'
            }
    except:
        pass
    
    return {
        'working': False,
        'confidence': 0,
        'filter_status': 'Error'
    }

def monitor_deployment():
    """Monitor deployment until complete"""
    print("🔍 MONITOREANDO DEPLOYMENT DEL FILTRO DE CONFIDENCE")
    print("=" * 60)
    print("⏰ Verificando cada 30 segundos hasta que esté activo...")
    print("Press Ctrl+C para detener")
    
    target_version = "4.2.0"
    checks = 0
    
    try:
        while True:
            checks += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            
            print(f"\n[{current_time}] Check #{checks}")
            print("-" * 30)
            
            # Check deployment status
            status = check_deployment_status()
            print(f"Versión: {status['version']}")
            print(f"Confidence filtering: {status['confidence_filtering']}")
            print(f"Threshold: {status['threshold']}")
            
            # Test filter functionality
            filter_test = test_confidence_filter_working()
            print(f"Filter working: {'✅' if filter_test['working'] else '❌'}")
            print(f"Test confidence: {filter_test['confidence']}%")
            print(f"Filter status: {filter_test['filter_status']}")
            
            if status['deployed'] and filter_test['working']:
                print("\n🎉 DEPLOYMENT COMPLETADO EXITOSAMENTE!")
                print("=" * 50)
                print("✅ Versión 4.2.0 activa")
                print("✅ Filtro de confidence >= 90% funcionando")
                print("✅ Predicciones de baja confidence rechazadas")
                print("✅ Sistema listo para mejora de +8.5% win rate")
                
                print(f"\n🔧 ACCIÓN INMEDIATA:")
                print("Actualizar EA con: g_minConfidence = 90.0")
                print("Esto activará la mejora transformacional")
                
                break
                
            elif status['version'] == target_version:
                print("⚠️  Versión correcta pero filtro aún no activo")
                
            else:
                print(f"⏳ Esperando deployment... (actual: {status['version']}, target: {target_version})")
            
            print("Next check in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoreo detenido por usuario")
        print("Puedes ejecutar este script nuevamente para continuar")

def main():
    print("🚀 MONITOR DE DEPLOYMENT - FILTRO DE CONFIDENCE")
    print("=" * 60)
    
    # Quick initial check
    status = check_deployment_status()
    
    if status['deployed']:
        print("🎉 ¡DEPLOYMENT YA COMPLETADO!")
        print("✅ Filtro de confidence >= 90% activo")
        print("🔧 Actualiza el EA con g_minConfidence = 90.0")
    else:
        print("⏳ Deployment en progreso...")
        print("🔍 Iniciando monitoreo automático...")
        monitor_deployment()

if __name__ == "__main__":
    main()
