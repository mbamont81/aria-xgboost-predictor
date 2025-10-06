#!/usr/bin/env python3
"""
Test prediction with automatic normalization
"""

import requests
import json
from datetime import datetime

BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def test_prediction_with_normalization():
    """Test prediction with a symbol that needs normalization"""
    print("🔍 Testing prediction with GOLD# (should normalize to XAUUSD)...")
    
    # Test data with GOLD# symbol
    test_data = {
        "symbol": "GOLD#",
        "timeframe": "M15",
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
            "resistance_distance": 0.008,
            "atr_percentile_100": 75.5,
            "rsi_std_20": 12.3,
            "price_acceleration": 0.001,
            "candle_body_ratio_mean": 0.65,
            "breakout_frequency": 0.15,
            "volume_imbalance": 0.05,
            "rolling_autocorr_20": 0.25,
            "hurst_exponent_50": 0.55
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📄 Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            debug_info = data.get('debug_info', {})
            
            print("✅ Prediction successful!")
            print(f"   Original symbol: {debug_info.get('original_symbol', 'N/A')}")
            print(f"   Normalized symbol: {debug_info.get('normalized_symbol', 'N/A')}")
            print(f"   Symbol changed: {debug_info.get('symbol_changed', 'N/A')}")
            print(f"   Stop Loss: {data.get('stop_loss', 'N/A')}")
            print(f"   Take Profit: {data.get('take_profit', 'N/A')}")
            print(f"   Regime: {data.get('regime', 'N/A')}")
            
            # Check if normalization worked
            if debug_info.get('original_symbol') == 'GOLD#' and debug_info.get('normalized_symbol') == 'XAUUSD':
                print("🎉 SERVER-SIDE NORMALIZATION CONFIRMED!")
                return True
            else:
                print("⚠️ Normalization info not found in response")
                return False
                
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 TESTING SERVER-SIDE NORMALIZATION")
    print("=" * 50)
    print(f"⏰ Time: {datetime.now().isoformat()}")
    
    success = test_prediction_with_normalization()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ SERVER-SIDE NORMALIZATION IS WORKING!")
        print("🔄 Double protection layer is active")
    else:
        print("❌ Could not confirm server-side normalization")
    print("=" * 50)

if __name__ == "__main__":
    main()
