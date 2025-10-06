#!/usr/bin/env python3
"""
Test directo con XAUUSD para verificar si el servicio está respondiendo correctamente
"""

import requests
import json

def test_xauusd_prediction():
    """Test prediction with XAUUSD using complete valid data"""
    
    # Complete prediction request matching the EA's format
    prediction_data = {
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
    
    print("🧪 TESTING XAUUSD PREDICTION")
    print("=" * 50)
    print(f"Request data: {json.dumps(prediction_data, indent=2)}")
    
    try:
        response = requests.post(
            "https://aria-xgboost-predictor.onrender.com/predict",
            json=prediction_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            print(f"Detected regime: {data.get('detected_regime', 'N/A')}")
            print(f"SL pips: {data.get('sl_pips', 'N/A')}")
            print(f"TP pips: {data.get('tp_pips', 'N/A')}")
            print(f"Overall confidence: {data.get('overall_confidence', 'N/A')}")
            print(f"Processing time: {data.get('processing_time_ms', 'N/A')} ms")
            
            debug_info = data.get('debug_info', {})
            if debug_info:
                print(f"\nDebug info:")
                for key, value in debug_info.items():
                    print(f"  {key}: {value}")
            
            return True
            
        else:
            print(f"❌ ERROR {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

def test_xauusd_with_suffix():
    """Test prediction with XAUUSD.m to see if normalization works"""
    
    prediction_data = {
        "symbol": "XAUUSD.m",  # This should normalize to XAUUSD
        "atr_percentile_100": 45.5,
        "rsi_std_20": 12.3,
        "price_acceleration": 0.02,
        "candle_body_ratio_mean": 0.65,
        "breakout_frequency": 0.15,
        "volume_imbalance": 0.08,
        "rolling_autocorr_20": 0.25,
        "hurst_exponent_50": 0.52
    }
    
    print("\n🧪 TESTING XAUUSD.m NORMALIZATION")
    print("=" * 50)
    
    try:
        response = requests.post(
            "https://aria-xgboost-predictor.onrender.com/predict",
            json=prediction_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS! Normalization is working!")
            
            debug_info = data.get('debug_info', {})
            original = debug_info.get('original_symbol', 'N/A')
            normalized = debug_info.get('normalized_symbol', 'N/A')
            print(f"Original: {original}")
            print(f"Normalized: {normalized}")
            
            return True
            
        elif response.status_code == 500:
            print("❌ 500 ERROR - Normalization not working")
            print(f"Response: {response.text}")
            return False
            
        else:
            print(f"❌ ERROR {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TESTING RENDER SERVICE FUNCTIONALITY")
    print("=" * 60)
    
    # Test 1: Direct XAUUSD (should work)
    xauusd_works = test_xauusd_prediction()
    
    # Test 2: XAUUSD.m normalization (should work if normalization is active)
    normalization_works = test_xauusd_with_suffix()
    
    print("\n📊 SUMMARY:")
    print("=" * 30)
    print(f"XAUUSD direct: {'✅ Working' if xauusd_works else '❌ Failed'}")
    print(f"XAUUSD.m normalization: {'✅ Working' if normalization_works else '❌ Failed'}")
    
    if xauusd_works and normalization_works:
        print("\n🎉 Both tests passed! Normalization is working.")
        print("The EA fallback messages should stop now.")
    elif xauusd_works and not normalization_works:
        print("\n⚠️  Service works but normalization is not active yet.")
        print("This explains the EA fallback messages.")
        print("Need to wait for deployment or investigate further.")
    else:
        print("\n❌ Service has issues. This explains the EA fallback messages.")
        print("Need to investigate service problems.")
