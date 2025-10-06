#!/usr/bin/env python3
"""
Verificación rápida del deployment
"""

import requests
from datetime import datetime

def quick_check():
    print("🔍 VERIFICACIÓN RÁPIDA DEL DEPLOYMENT")
    print("=" * 50)
    print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        response = requests.get("https://aria-xgboost-predictor.onrender.com/", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            version = data.get('version', 'N/A')
            confidence_filtering = data.get('confidence_filtering', 'N/A')
            threshold = data.get('confidence_threshold', 'N/A')
            
            print(f"📊 Versión: {version}")
            print(f"🎯 Confidence filtering: {confidence_filtering}")
            print(f"📈 Threshold: {threshold}")
            
            if version == "4.2.0" and confidence_filtering == "enabled":
                print("\n🎉 ¡DEPLOYMENT EXITOSO!")
                print("✅ Filtro de confidence >= 90% activo")
                print("🔧 Listo para actualizar EA")
                return True
            else:
                print(f"\n⏳ Deployment aún propagándose...")
                print(f"Expected: v4.2.0 con confidence_filtering=enabled")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = quick_check()
    
    if success:
        print("\n🚀 PRÓXIMO PASO:")
        print("Actualizar EA: g_minConfidence = 90.0")
    else:
        print("\n⏳ Esperar 5 minutos más y verificar nuevamente")

quick_check()
