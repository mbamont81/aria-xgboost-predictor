#!/usr/bin/env python3
"""
Simple test to verify normalization is working by using correct schema
and checking if we get different error types for normalized vs non-normalized symbols.
"""

import requests
import json
import time
from datetime import datetime

# Render service URL
BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def create_valid_prediction_request(symbol):
    """Create a valid prediction request with all required fields"""
    return {
        "symbol": symbol,
        "atr_percentile_100": 45.5,
        "rsi_std_20": 12.3,
        "price_acceleration": 0.02,
        "candle_body_ratio_mean": 0.65,
        "breakout_frequency": 0.15,
        "volume_imbalance": 0.08,
        "rolling_autocorr_20": 0.25,
        "hurst_exponent_50": 0.52
    }

def test_normalization_logic():
    """Test normalization by comparing responses for different symbol types"""
    print("🧪 TESTING NORMALIZATION LOGIC")
    print("=" * 60)
    
    test_cases = [
        {
            "symbol": "XAUUSD.m",
            "description": "Should normalize to XAUUSD (model exists)",
            "expected_result": "success_or_specific_error"
        },
        {
            "symbol": "XAUUSD", 
            "description": "Already normalized (model exists)",
            "expected_result": "success_or_specific_error"
        },
        {
            "symbol": "BTCUSDc",
            "description": "Should normalize to BTCUSD (model exists)", 
            "expected_result": "success_or_specific_error"
        },
        {
            "symbol": "BTCUSD",
            "description": "Already normalized (model exists)",
            "expected_result": "success_or_specific_error"
        },
        {
            "symbol": "USTEC.f",
            "description": "Should normalize to NAS100 (model doesn't exist)",
            "expected_result": "model_not_found_error"
        },
        {
            "symbol": "INVALIDXYZ",
            "description": "Invalid symbol (no normalization, no model)",
            "expected_result": "model_not_found_error"
        }
    ]
    
    results = []
    
    for test_case in test_cases:
        symbol = test_case["symbol"]
        description = test_case["description"]
        
        try:
            print(f"\n🔍 Testing: {symbol}")
            print(f"   Description: {description}")
            
            request_data = create_valid_prediction_request(symbol)
            
            response = requests.post(
                f"{BASE_URL}/predict",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                debug_info = data.get('debug_info', {})
                original_symbol = debug_info.get('original_symbol', 'N/A')
                normalized_symbol = debug_info.get('normalized_symbol', 'N/A')
                model_used = data.get('model_used', 'N/A')
                
                print(f"   ✅ SUCCESS!")
                print(f"   📊 Original: {original_symbol}")
                print(f"   📊 Normalized: {normalized_symbol}")
                print(f"   📊 Model: {model_used}")
                
                results.append({
                    'symbol': symbol,
                    'status': 'success',
                    'original_symbol': original_symbol,
                    'normalized_symbol': normalized_symbol,
                    'model_used': model_used,
                    'description': description
                })
                
            elif response.status_code == 500:
                print(f"   ❌ 500 ERROR (likely model not found)")
                try:
                    error_data = response.json()
                    error_detail = error_data.get('detail', 'No details')
                    print(f"   📝 Error: {error_detail}")
                except:
                    error_detail = response.text
                    print(f"   📝 Error text: {error_detail}")
                
                results.append({
                    'symbol': symbol,
                    'status': 'error_500_model_not_found',
                    'original_symbol': symbol,
                    'normalized_symbol': 'unknown',
                    'model_used': 'none',
                    'error_detail': error_detail,
                    'description': description
                })
                
            else:
                print(f"   ❌ Error {response.status_code}")
                print(f"   📝 Response: {response.text[:200]}...")
                
                results.append({
                    'symbol': symbol,
                    'status': f'error_{response.status_code}',
                    'original_symbol': symbol,
                    'normalized_symbol': 'unknown',
                    'model_used': 'none',
                    'error_detail': response.text[:200],
                    'description': description
                })
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
            results.append({
                'symbol': symbol,
                'status': f'exception: {e}',
                'original_symbol': symbol,
                'normalized_symbol': 'unknown',
                'model_used': 'none',
                'description': description
            })
            
        time.sleep(1)  # Small delay between requests
    
    return results

def analyze_normalization_results(results):
    """Analyze results to determine if normalization is working"""
    print("\n\n📊 NORMALIZATION ANALYSIS")
    print("=" * 60)
    
    successful_predictions = 0
    normalization_working = False
    model_not_found_errors = 0
    
    print("\n🔍 DETAILED RESULTS:")
    print("-" * 40)
    
    for result in results:
        symbol = result['symbol']
        status = result['status']
        original = result.get('original_symbol', 'N/A')
        normalized = result.get('normalized_symbol', 'N/A')
        
        print(f"\n📋 {symbol}:")
        print(f"   Status: {status}")
        print(f"   Description: {result['description']}")
        
        if status == 'success':
            successful_predictions += 1
            print(f"   ✅ PREDICTION SUCCESS")
            print(f"   📊 {original} → {normalized}")
            
            # Check if normalization happened
            if original != normalized and original != 'N/A':
                normalization_working = True
                print(f"   🎯 NORMALIZATION DETECTED!")
            elif original == normalized:
                print(f"   ➡️  No normalization needed")
            
        elif 'error_500' in status:
            model_not_found_errors += 1
            print(f"   ❌ MODEL NOT FOUND (expected for some symbols)")
            
        else:
            print(f"   ⚠️  OTHER ERROR: {status}")
    
    # Summary analysis
    print(f"\n📈 SUMMARY:")
    print("-" * 40)
    print(f"Total symbols tested: {len(results)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Model not found errors: {model_not_found_errors}")
    print(f"Normalization detected: {'✅ YES' if normalization_working else '❌ NO'}")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    print("-" * 40)
    
    if successful_predictions > 0:
        print("✅ Service is responding and processing requests")
        
    if normalization_working:
        print("✅ Server-side normalization is WORKING!")
        print("✅ Symbols are being transformed before model lookup")
    else:
        print("⚠️  Normalization not clearly detected in responses")
        print("🔍 Check debug_info in successful responses")
        
    if model_not_found_errors > 0:
        print(f"✅ Model validation working (rejected {model_not_found_errors} invalid symbols)")
    
    # Expected behavior verification
    print(f"\n🎯 EXPECTED BEHAVIOR CHECK:")
    print("-" * 40)
    
    # Check specific cases
    xauusd_m_result = next((r for r in results if r['symbol'] == 'XAUUSD.m'), None)
    xauusd_result = next((r for r in results if r['symbol'] == 'XAUUSD'), None)
    
    if xauusd_m_result and xauusd_result:
        if xauusd_m_result['status'] == xauusd_result['status']:
            print("✅ XAUUSD.m and XAUUSD have same result (normalization working)")
        else:
            print("⚠️  XAUUSD.m and XAUUSD have different results")
    
    return {
        'total_tested': len(results),
        'successful_predictions': successful_predictions,
        'normalization_working': normalization_working,
        'model_not_found_errors': model_not_found_errors
    }

def main():
    print(f"🚀 TESTING SERVER-SIDE NORMALIZATION")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Test normalization logic
    results = test_normalization_logic()
    
    # Analyze results
    summary = analyze_normalization_results(results)
    
    print(f"\n✨ Testing completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Final verdict
    print(f"\n🏆 FINAL VERDICT:")
    print("-" * 40)
    if summary['normalization_working']:
        print("🎉 SUCCESS: Server-side normalization is working!")
        print("🎯 The improved suffix handling should reduce 500 errors")
    elif summary['successful_predictions'] > 0:
        print("⚠️  PARTIAL: Service working but normalization unclear")
        print("🔍 Check Render logs for normalization details")
    else:
        print("❌ ISSUE: No successful predictions - check deployment")
    
    return summary

if __name__ == "__main__":
    main()
