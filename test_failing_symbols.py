#!/usr/bin/env python3
"""
Test script to verify that the improved normalization fixes the failing symbols
identified in the Render logs analysis.
"""

import requests
import json
import time
from datetime import datetime

# Render service URL
BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def test_symbol_normalization():
    """Test the normalization endpoint with the failing symbols"""
    print("🧪 TESTING SYMBOL NORMALIZATION")
    print("=" * 60)
    
    # Symbols that were failing based on log analysis
    failing_symbols = [
        "XAUUSD.m",     # Should normalize to XAUUSD
        "EURJPYc",      # Should normalize to EURJPY  
        "USTEC.f",      # Should normalize to NAS100
        "EURNZDc",      # Should normalize to EURNZD
        "USDJPYm",      # Should normalize to USDJPY
        "EURCHFc",      # Should normalize to EURCHF
        "BTCUSDc",      # Should normalize to BTCUSD
        "EURGBPc",      # Should normalize to EURGBP
        "AUDCADc",      # Should normalize to AUDCHF (known mapping)
        "EURJPY",       # Should remain EURJPY (no model issue)
        "EURNZD"        # Should remain EURNZD (no model issue)
    ]
    
    results = []
    
    for symbol in failing_symbols:
        try:
            print(f"\n🔍 Testing: {symbol}")
            response = requests.get(f"{BASE_URL}/normalize-symbol/{symbol}")
            
            if response.status_code == 200:
                data = response.json()
                normalized = data.get('normalized_symbol', 'N/A')
                print(f"   ✅ {symbol} → {normalized}")
                results.append({
                    'original': symbol,
                    'normalized': normalized,
                    'status': 'success'
                })
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                results.append({
                    'original': symbol,
                    'normalized': None,
                    'status': f'error_{response.status_code}'
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                'original': symbol,
                'normalized': None,
                'status': f'exception: {e}'
            })
            
        time.sleep(0.5)  # Small delay between requests
    
    return results

def test_prediction_with_failing_symbols():
    """Test prediction endpoint with the failing symbols to see if they work now"""
    print("\n\n🎯 TESTING PREDICTIONS WITH FAILING SYMBOLS")
    print("=" * 60)
    
    # Test a few key symbols that should now work
    test_symbols = [
        "XAUUSD.m",     # Should normalize to XAUUSD and predict
        "USTEC.f",      # Should normalize to NAS100 and predict  
        "BTCUSDc",      # Should normalize to BTCUSD and predict
        "AUDCADc"       # Should normalize to AUDCHF and predict
    ]
    
    # Minimal prediction request (will get 422 but should show normalization in logs)
    prediction_data = {
        "symbol": "",  # Will be filled per symbol
        "timeframe": "M5",
        "atr_percentile_100": 50.0,
        "rsi_14": 50.0,
        "bb_position": 0.5,
        "macd_signal": 0.0,
        "stoch_k": 50.0,
        "williams_r": -50.0,
        "cci_14": 0.0,
        "momentum_10": 1.0,
        "roc_10": 0.0,
        "adx_14": 25.0,
        "aroon_up": 50.0,
        "aroon_down": 50.0,
        "psar_signal": 0.0,
        "obv_trend": 0.0,
        "vwap_distance": 0.0,
        "pivot_r1_distance": 0.0,
        "pivot_s1_distance": 0.0,
        "fib_236_distance": 0.0,
        "fib_618_distance": 0.0,
        "ma_20_slope": 0.0,
        "ma_50_slope": 0.0,
        "volume_sma_ratio": 1.0,
        "price_ma20_ratio": 1.0,
        "high_low_ratio": 1.0,
        "close_open_ratio": 1.0
    }
    
    results = []
    
    for symbol in test_symbols:
        try:
            print(f"\n🎯 Testing prediction: {symbol}")
            prediction_data["symbol"] = symbol
            
            response = requests.post(
                f"{BASE_URL}/predict",
                json=prediction_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                debug_info = data.get('debug_info', {})
                original_symbol = debug_info.get('original_symbol', 'N/A')
                normalized_symbol = debug_info.get('normalized_symbol', 'N/A')
                print(f"   ✅ Prediction successful!")
                print(f"   📊 {original_symbol} → {normalized_symbol}")
                results.append({
                    'symbol': symbol,
                    'status': 'prediction_success',
                    'normalized': normalized_symbol
                })
            elif response.status_code == 422:
                # Expected due to minimal data, but check if normalization happened
                print(f"   ⚠️  422 (expected - minimal data)")
                print(f"   🔍 Normalization should still be logged on server")
                results.append({
                    'symbol': symbol,
                    'status': 'validation_error_expected',
                    'normalized': 'check_logs'
                })
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                results.append({
                    'symbol': symbol,
                    'status': f'error_{response.status_code}',
                    'normalized': None
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                'symbol': symbol,
                'status': f'exception: {e}',
                'normalized': None
            })
            
        time.sleep(1)  # Delay between prediction requests
    
    return results

def generate_test_report(norm_results, pred_results):
    """Generate a comprehensive test report"""
    print("\n\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    # Normalization results
    print("\n🔧 NORMALIZATION RESULTS:")
    print("-" * 40)
    successful_normalizations = 0
    for result in norm_results:
        if result['status'] == 'success':
            successful_normalizations += 1
            print(f"✅ {result['original']:12} → {result['normalized']}")
        else:
            print(f"❌ {result['original']:12} → {result['status']}")
    
    print(f"\n📈 Normalization Success Rate: {successful_normalizations}/{len(norm_results)} ({successful_normalizations/len(norm_results)*100:.1f}%)")
    
    # Prediction results  
    print("\n🎯 PREDICTION RESULTS:")
    print("-" * 40)
    for result in pred_results:
        status_icon = "✅" if "success" in result['status'] else "⚠️" if "expected" in result['status'] else "❌"
        print(f"{status_icon} {result['symbol']:12} → {result['status']}")
    
    # Expected fixes verification
    print("\n🎯 EXPECTED FIXES VERIFICATION:")
    print("-" * 40)
    expected_mappings = {
        "XAUUSD.m": "XAUUSD",
        "EURJPYc": "EURJPY", 
        "USTEC.f": "NAS100",
        "EURNZDc": "EURNZD",
        "USDJPYm": "USDJPY",
        "EURCHFc": "EURCHF",
        "BTCUSDc": "BTCUSD",
        "EURGBPc": "EURGBP",
        "AUDCADc": "AUDCHF"
    }
    
    fixes_working = 0
    for result in norm_results:
        original = result['original']
        if original in expected_mappings:
            expected = expected_mappings[original]
            actual = result['normalized']
            if actual == expected:
                fixes_working += 1
                print(f"✅ {original:12} → {actual} (as expected)")
            else:
                print(f"❌ {original:12} → {actual} (expected {expected})")
    
    print(f"\n📊 Expected Fixes Working: {fixes_working}/{len(expected_mappings)} ({fixes_working/len(expected_mappings)*100:.1f}%)")
    
    # Next steps
    print("\n🔮 NEXT STEPS:")
    print("-" * 40)
    print("1. Monitor Render logs for reduced 500 errors")
    print("2. Check if EURJPY, EURNZD need model files")
    print("3. Verify real trading requests show improved success rate")
    
    return {
        'normalization_success_rate': successful_normalizations/len(norm_results)*100,
        'expected_fixes_working': fixes_working/len(expected_mappings)*100,
        'total_symbols_tested': len(norm_results)
    }

def main():
    print(f"🚀 TESTING IMPROVED NORMALIZATION")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test normalization endpoint
    norm_results = test_symbol_normalization()
    
    # Test prediction endpoint  
    pred_results = test_prediction_with_failing_symbols()
    
    # Generate report
    summary = generate_test_report(norm_results, pred_results)
    
    print(f"\n✨ Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return summary

if __name__ == "__main__":
    main()
