#!/usr/bin/env python3
"""
Verificación de emergencia del estado del servicio después del rollback
"""

import requests
from datetime import datetime

def emergency_service_check():
    print("🚨 VERIFICACIÓN DE EMERGENCIA POST-ROLLBACK")
    print("=" * 60)
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # 1. Check service status
        print("1. 🔍 Verificando servicio principal...")
        r = requests.get("https://aria-xgboost-predictor.onrender.com/", timeout=10)
        print(f"   Status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"   ✅ Servicio respondiendo")
            print(f"   Message: {data.get('message', 'N/A')}")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Models loaded: {data.get('models_loaded', 'N/A')}")
        else:
            print(f"   ❌ Servicio no responde correctamente")
            return False
        
        # 2. Test critical /predict endpoint
        print("\n2. 🧪 Testing endpoint /predict...")
        test_data = {
            "symbol": "XAUUSD",
            "atr_percentile_100": 50.0,
            "rsi_std_20": 10.0,
            "price_acceleration": 0.01,
            "candle_body_ratio_mean": 0.6,
            "breakout_frequency": 0.1,
            "volume_imbalance": 0.05,
            "rolling_autocorr_20": 0.2,
            "hurst_exponent_50": 0.5
        }
        
        r2 = requests.post(
            "https://aria-xgboost-predictor.onrender.com/predict",
            json=test_data,
            timeout=15
        )
        
        print(f"   Status: {r2.status_code}")
        
        if r2.status_code == 200:
            print("   ✅ SERVICIO RESTAURADO EXITOSAMENTE!")
            print("   ✅ Endpoint /predict funcionando")
            return True
        elif r2.status_code == 404:
            print("   ❌ Servicio aún roto - /predict devuelve 404")
            return False
        else:
            print(f"   ⚠️  Servicio responde pero con error {r2.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🚨 VERIFICACIÓN DE EMERGENCIA")
    print("=" * 50)
    
    service_ok = emergency_service_check()
    
    print(f"\n📊 RESULTADO:")
    print("=" * 30)
    
    if service_ok:
        print("🎉 SERVICIO RESTAURADO!")
        print("✅ Rollback exitoso")
        print("✅ Usuarios pueden usar predicciones")
        print("✅ Servicio estable")
        
        print(f"\n💡 PRÓXIMOS PASOS:")
        print("1. Servicio restaurado y funcional")
        print("2. Agregar endpoint de training de forma incremental")
        print("3. Testing exhaustivo antes de deployment")
    else:
        print("❌ SERVICIO AÚN ROTO")
        print("🚨 Necesita intervención adicional")
        print("🔍 Revisar logs de Render para más detalles")

if __name__ == "__main__":
    main()
