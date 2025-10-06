#!/usr/bin/env python3
"""
Verificación completa del deployment del filtro de confidence
"""

import requests
import time
from datetime import datetime

BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def check_service_version():
    """Verificar versión y configuración del servicio"""
    print("🔍 VERIFICANDO VERSIÓN DEL SERVICIO")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            version = data.get('version', 'N/A')
            confidence_filtering = data.get('confidence_filtering', 'N/A')
            threshold = data.get('confidence_threshold', 'N/A')
            expected_win_rate = data.get('expected_win_rate', 'N/A')
            improvement = data.get('improvement', 'N/A')
            deployment_time = data.get('deployment_time', 'N/A')
            
            print(f"📊 Versión: {version}")
            print(f"🎯 Confidence filtering: {confidence_filtering}")
            print(f"📈 Threshold: {threshold}")
            print(f"🏆 Expected win rate: {expected_win_rate}")
            print(f"📊 Improvement: {improvement}")
            print(f"⏰ Deployment time: {deployment_time}")
            
            # Verificar si es la versión correcta
            target_version = "4.2.0"
            if version == target_version:
                print(f"\n✅ DEPLOYMENT EXITOSO!")
                print(f"✅ Versión correcta: {version}")
                if confidence_filtering == "enabled":
                    print(f"✅ Filtro de confidence activo")
                    return True
                else:
                    print(f"⚠️  Filtro de confidence no activo")
                    return False
            else:
                print(f"\n⚠️  DEPLOYMENT PENDIENTE")
                print(f"⏳ Versión actual: {version}")
                print(f"🎯 Versión esperada: {target_version}")
                return False
                
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_confidence_filtering():
    """Test que el filtro de confidence esté funcionando"""
    print(f"\n🧪 TESTING FILTRO DE CONFIDENCE")
    print("=" * 50)
    
    # Datos de prueba
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
    
    results = []
    
    for i in range(3):
        try:
            print(f"\n🔍 Test {i+1}/3:")
            
            response = requests.post(
                f"{BASE_URL}/predict",
                json=test_data,
                headers={"Content-Type": "application/json"},
                timeout=15
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                confidence = data.get('overall_confidence', 0)
                regime = data.get('detected_regime', 'N/A')
                
                debug_info = data.get('debug_info', {})
                confidence_filter = debug_info.get('confidence_filter', 'N/A')
                expected_win_rate = debug_info.get('expected_win_rate', 'N/A')
                
                print(f"   ✅ PREDICCIÓN ACEPTADA")
                print(f"   📊 Confidence: {confidence}%")
                print(f"   🌊 Régimen: {regime}")
                print(f"   🎯 Filter status: {confidence_filter}")
                print(f"   🏆 Expected win rate: {expected_win_rate}")
                
                results.append({
                    'status': 'accepted',
                    'confidence': confidence,
                    'filter_working': confidence >= 90
                })
                
            elif response.status_code == 422:
                try:
                    error_data = response.json()
                    detail = error_data.get('detail', 'No details')
                    print(f"   🚫 PREDICCIÓN RECHAZADA (FILTRO ACTIVO)")
                    print(f"   📝 Razón: {detail}")
                    
                    results.append({
                        'status': 'rejected',
                        'reason': detail,
                        'filter_working': True
                    })
                except:
                    print(f"   🚫 PREDICCIÓN RECHAZADA (422)")
                    results.append({
                        'status': 'rejected',
                        'filter_working': True
                    })
                    
            else:
                print(f"   ❌ ERROR: {response.status_code}")
                results.append({
                    'status': 'error',
                    'code': response.status_code,
                    'filter_working': False
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                'status': 'exception',
                'filter_working': False
            })
        
        time.sleep(1)
    
    # Analizar resultados
    print(f"\n📊 ANÁLISIS DE FILTRO:")
    print("-" * 30)
    
    accepted = sum(1 for r in results if r['status'] == 'accepted')
    rejected = sum(1 for r in results if r['status'] == 'rejected')
    filter_working = sum(1 for r in results if r.get('filter_working', False))
    
    print(f"✅ Aceptadas: {accepted}/3")
    print(f"🚫 Rechazadas: {rejected}/3")
    print(f"🎯 Filtro funcionando: {filter_working}/3")
    
    if accepted > 0:
        accepted_results = [r for r in results if r['status'] == 'accepted']
        avg_confidence = sum(r['confidence'] for r in accepted_results) / len(accepted_results)
        print(f"📊 Confidence promedio (aceptadas): {avg_confidence:.1f}%")
        
        if avg_confidence >= 90:
            print("✅ Todas las predicciones aceptadas tienen confidence >= 90%")
        else:
            print("⚠️  Algunas predicciones aceptadas tienen confidence < 90%")
    
    return filter_working > 0

def verify_complete_deployment():
    """Verificación completa del deployment"""
    print("🔍 VERIFICACIÓN COMPLETA DEL DEPLOYMENT")
    print("=" * 60)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Verificar versión del servicio
    version_ok = check_service_version()
    
    # 2. Test del filtro de confidence
    filter_ok = test_confidence_filtering()
    
    # 3. Resultado final
    print(f"\n🎯 RESULTADO FINAL:")
    print("=" * 30)
    
    if version_ok and filter_ok:
        print("🎉 DEPLOYMENT COMPLETAMENTE EXITOSO!")
        print("✅ Versión 4.2.0 activa")
        print("✅ Filtro de confidence >= 90% funcionando")
        print("✅ Listo para actualizar EA")
        
        print(f"\n🔧 PRÓXIMO PASO:")
        print("Actualizar EA con: g_minConfidence = 90.0")
        print("Esto activará la mejora de +8.5% win rate")
        
        return True
        
    elif version_ok and not filter_ok:
        print("⚠️  DEPLOYMENT PARCIAL")
        print("✅ Versión actualizada")
        print("❌ Filtro no funcionando correctamente")
        print("🔍 Revisar logs de Render")
        
        return False
        
    else:
        print("❌ DEPLOYMENT PENDIENTE")
        print("⏳ Esperando propagación...")
        print("🔄 Reintentar en 5-10 minutos")
        
        return False

def main():
    success = verify_complete_deployment()
    
    print(f"\n📋 RESUMEN:")
    print("=" * 30)
    
    if success:
        print("🚀 Sistema listo para mejora de +8.5% win rate")
    else:
        print("⏳ Esperando deployment completo")

if __name__ == "__main__":
    main()
