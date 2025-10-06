#!/usr/bin/env python3
"""
Monitor para verificar cuando la normalización esté activa
"""

import requests
import time
from datetime import datetime

def check_service_version():
    """Check current service version"""
    try:
        response = requests.get("https://aria-xgboost-predictor.onrender.com/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            version = data.get('version', 'N/A')
            deployment_time = data.get('deployment_time', 'N/A')
            normalization_active = data.get('normalization_active', False)
            return version, deployment_time, normalization_active
    except:
        pass
    return None, None, False

def test_normalization():
    """Test if normalization is working"""
    try:
        prediction_data = {
            "symbol": "XAUUSD.m",
            "atr_percentile_100": 45.5,
            "rsi_std_20": 12.3,
            "price_acceleration": 0.02,
            "candle_body_ratio_mean": 0.65,
            "breakout_frequency": 0.15,
            "volume_imbalance": 0.08,
            "rolling_autocorr_20": 0.25,
            "hurst_exponent_50": 0.52
        }
        
        response = requests.post(
            "https://aria-xgboost-predictor.onrender.com/predict",
            json=prediction_data,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        
        return response.status_code == 200
    except:
        return False

def monitor_deployment():
    """Monitor deployment until normalization is working"""
    print("🔍 MONITORING DEPLOYMENT STATUS")
    print("=" * 50)
    print("Checking every 30 seconds until normalization is active...")
    print("Press Ctrl+C to stop monitoring")
    
    target_version = "4.1.0"
    checks = 0
    
    try:
        while True:
            checks += 1
            current_time = datetime.now().strftime('%H:%M:%S')
            
            print(f"\n[{current_time}] Check #{checks}")
            print("-" * 30)
            
            # Check service version
            version, deployment_time, normalization_flag = check_service_version()
            print(f"Service version: {version}")
            print(f"Deployment time: {deployment_time}")
            print(f"Normalization flag: {normalization_flag}")
            
            # Test normalization
            normalization_works = test_normalization()
            print(f"XAUUSD.m test: {'✅ SUCCESS' if normalization_works else '❌ FAILED'}")
            
            if version == target_version and normalization_works:
                print("\n🎉 SUCCESS! NORMALIZATION IS NOW ACTIVE!")
                print("=" * 50)
                print("✅ Service updated to version 4.1.0")
                print("✅ XAUUSD.m normalization working")
                print("✅ EA fallback messages should stop now")
                break
            elif version == target_version:
                print(f"⚠️  Version updated but normalization still not working")
            else:
                print(f"⏳ Waiting for deployment... (current: {version}, target: {target_version})")
            
            print(f"Next check in 30 seconds...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
        print("You can run this script again to continue monitoring")

if __name__ == "__main__":
    monitor_deployment()
