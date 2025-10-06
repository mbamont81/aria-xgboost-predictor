#!/usr/bin/env python3
"""
Test script focused on prediction endpoint to verify normalization is working internally.
Since /normalize-symbol endpoint returns 404, we'll test via /predict with complete data.
"""

import requests
import json
import time
from datetime import datetime

# Render service URL
BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def test_predictions_with_complete_data():
    """Test prediction endpoint with complete valid data for failing symbols"""
    print("🎯 TESTING PREDICTIONS WITH COMPLETE DATA")
    print("=" * 60)
    
    # Test symbols that should now work after normalization improvements
    test_cases = [
        {
            "original": "XAUUSD.m",
            "expected_normalized": "XAUUSD",
            "should_work": True
        },
        {
            "original": "USTEC.f", 
            "expected_normalized": "NAS100",
            "should_work": False  # NAS100 not in available symbols
        },
        {
            "original": "BTCUSDc",
            "expected_normalized": "BTCUSD", 
            "should_work": True
        },
        {
            "original": "AUDCADc",
            "expected_normalized": "AUDCHF",
            "should_work": True
        },
        {
            "original": "EURJPYc",
            "expected_normalized": "EURJPY",
            "should_work": False  # EURJPY not in available symbols
        }
    ]
    
    # Complete prediction request with all required fields
    base_prediction_data = {
        "symbol": "",  # Will be filled per test case
        "timeframe": "M5",
        "atr_percentile_100": 45.5,
        "rsi_14": 52.3,
        "bb_position": 0.6,
        "macd_signal": 0.1,
        "stoch_k": 48.7,
        "williams_r": -45.2,
        "cci_14": 12.5,
        "momentum_10": 1.02,
        "roc_10": 0.8,
        "adx_14": 28.3,
        "aroon_up": 55.0,
        "aroon_down": 35.0,
        "psar_signal": 1.0,
        "obv_trend": 0.2,
        "vwap_distance": 0.15,
        "pivot_r1_distance": 0.05,
        "pivot_s1_distance": -0.03,
        "fib_236_distance": 0.02,
        "fib_618_distance": -0.01,
        "ma_20_slope": 0.3,
        "ma_50_slope": 0.1,
        "volume_sma_ratio": 1.2,
        "price_ma20_ratio": 1.01,
        "high_low_ratio": 1.15,
        "close_open_ratio": 1.003
    }
    
    results = []
    
    for test_case in test_cases:
        symbol = test_case["original"]
        expected = test_case["expected_normalized"]
        should_work = test_case["should_work"]
        
        try:
            print(f"\n🔍 Testing: {symbol}")
            print(f"   Expected normalization: {symbol} → {expected}")
            print(f"   Should work: {'✅' if should_work else '❌'} (model {'available' if should_work else 'not available'})")
            
            prediction_data = base_prediction_data.copy()
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
                
                print(f"   ✅ Prediction SUCCESS!")
                print(f"   📊 Normalization: {original_symbol} → {normalized_symbol}")
                
                # Check if normalization worked as expected
                normalization_correct = normalized_symbol == expected
                print(f"   🎯 Normalization correct: {'✅' if normalization_correct else '❌'}")
                
                results.append({
                    'symbol': symbol,
                    'status': 'success',
                    'original_symbol': original_symbol,
                    'normalized_symbol': normalized_symbol,
                    'expected_normalized': expected,
                    'normalization_correct': normalization_correct,
                    'should_work': should_work
                })
                
            elif response.status_code == 500:
                print(f"   ❌ 500 ERROR - Model not found or normalization failed")
                try:
                    error_data = response.json()
                    print(f"   📝 Error details: {error_data}")
                except:
                    print(f"   📝 Error text: {response.text}")
                
                results.append({
                    'symbol': symbol,
                    'status': 'error_500',
                    'original_symbol': symbol,
                    'normalized_symbol': 'unknown',
                    'expected_normalized': expected,
                    'normalization_correct': False,
                    'should_work': should_work
                })
                
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                results.append({
                    'symbol': symbol,
                    'status': f'error_{response.status_code}',
                    'original_symbol': symbol,
                    'normalized_symbol': 'unknown',
                    'expected_normalized': expected,
                    'normalization_correct': False,
                    'should_work': should_work
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                'symbol': symbol,
                'status': f'exception: {e}',
                'original_symbol': symbol,
                'normalized_symbol': 'unknown',
                'expected_normalized': expected,
                'normalization_correct': False,
                'should_work': should_work
            })
            
        time.sleep(2)  # Delay between requests
    
    return results

def analyze_results(results):
    """Analyze the test results and provide insights"""
    print("\n\n📊 DETAILED RESULTS ANALYSIS")
    print("=" * 60)
    
    successful_predictions = 0
    correct_normalizations = 0
    expected_successes = 0
    unexpected_failures = 0
    
    for result in results:
        symbol = result['symbol']
        status = result['status']
        normalized = result['normalized_symbol']
        expected = result['expected_normalized']
        should_work = result['should_work']
        normalization_correct = result['normalization_correct']
        
        print(f"\n🔍 {symbol}:")
        print(f"   Status: {status}")
        print(f"   Normalized: {normalized}")
        print(f"   Expected: {expected}")
        print(f"   Should work: {'✅' if should_work else '❌'}")
        
        if status == 'success':
            successful_predictions += 1
            print(f"   Result: ✅ PREDICTION SUCCESS")
            if normalization_correct:
                correct_normalizations += 1
                print(f"   Normalization: ✅ CORRECT")
            else:
                print(f"   Normalization: ❌ INCORRECT")
        else:
            print(f"   Result: ❌ PREDICTION FAILED")
            
        # Check if result matches expectation
        if should_work and status == 'success':
            expected_successes += 1
            print(f"   Expectation: ✅ EXPECTED SUCCESS")
        elif not should_work and status != 'success':
            expected_successes += 1  
            print(f"   Expectation: ✅ EXPECTED FAILURE (no model)")
        elif should_work and status != 'success':
            unexpected_failures += 1
            print(f"   Expectation: ❌ UNEXPECTED FAILURE")
        else:
            print(f"   Expectation: ⚠️  UNEXPECTED SUCCESS")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print("-" * 40)
    print(f"Total symbols tested: {len(results)}")
    print(f"Successful predictions: {successful_predictions}/{len(results)} ({successful_predictions/len(results)*100:.1f}%)")
    print(f"Correct normalizations: {correct_normalizations}/{len(results)} ({correct_normalizations/len(results)*100:.1f}%)")
    print(f"Expected outcomes: {expected_successes}/{len(results)} ({expected_successes/len(results)*100:.1f}%)")
    print(f"Unexpected failures: {unexpected_failures}")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print("-" * 40)
    
    if correct_normalizations > 0:
        print(f"✅ Normalization is working for {correct_normalizations} symbols")
    else:
        print(f"❌ No correct normalizations detected - may need to check deployment")
        
    if unexpected_failures > 0:
        print(f"⚠️  {unexpected_failures} symbols failed unexpectedly - investigate further")
    else:
        print(f"✅ All failures were expected (missing models)")
        
    return {
        'total_tested': len(results),
        'successful_predictions': successful_predictions,
        'correct_normalizations': correct_normalizations,
        'expected_outcomes': expected_successes,
        'unexpected_failures': unexpected_failures
    }

def main():
    print(f"🚀 TESTING NORMALIZATION VIA PREDICTIONS")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test predictions with complete data
    results = test_predictions_with_complete_data()
    
    # Analyze results
    summary = analyze_results(results)
    
    print(f"\n✨ Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final assessment
    print(f"\n🎯 FINAL ASSESSMENT:")
    print("-" * 40)
    if summary['correct_normalizations'] >= 2:
        print("✅ Normalization improvements appear to be working!")
        print("✅ Server-side normalization is successfully handling suffix patterns")
    elif summary['successful_predictions'] > 0:
        print("⚠️  Some predictions working, but normalization verification needed")
        print("🔍 Check Render logs for normalization details")
    else:
        print("❌ No successful predictions - deployment may not be active yet")
        print("🔄 May need to wait for deployment or check service status")
    
    return summary

if __name__ == "__main__":
    main()
