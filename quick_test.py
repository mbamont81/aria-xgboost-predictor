#!/usr/bin/env python3
"""
Quick test to check if server-side normalization is active
"""

import requests
import json
from datetime import datetime

BASE_URL = "https://aria-xgboost-predictor.onrender.com"

def test_root_endpoint():
    """Test root endpoint for new format"""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📄 Response keys: {list(data.keys())}")
            
            # Check for new format indicators
            if 'message' in data and 'symbol_normalization' in data:
                print("🎉 NEW FORMAT DETECTED!")
                print(f"   Message: {data.get('message')}")
                print(f"   Symbol normalization: {data.get('symbol_normalization')}")
                print(f"   Endpoints: {data.get('available_endpoints', [])}")
                return True
            else:
                print("⚠️ Old format still active")
                print(f"   Service: {data.get('service', 'N/A')}")
                print(f"   Version: {data.get('version', 'N/A')}")
                return False
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_normalize_endpoint():
    """Test the normalize endpoint"""
    print("\n🔍 Testing normalize endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/normalize-symbol/TECDE30", timeout=30)
        print(f"📄 Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Normalization working!")
            print(f"   Original: {data.get('original_symbol')}")
            print(f"   Normalized: {data.get('normalized_symbol')}")
            return True
        elif response.status_code == 404:
            print("❌ Endpoint not found (404) - Old deployment still active")
            return False
        else:
            print(f"❌ Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 QUICK DEPLOYMENT CHECK")
    print("=" * 40)
    print(f"⏰ Time: {datetime.now().isoformat()}")
    
    root_ok = test_root_endpoint()
    normalize_ok = test_normalize_endpoint()
    
    print("\n" + "=" * 40)
    if root_ok and normalize_ok:
        print("✅ NEW DEPLOYMENT IS ACTIVE!")
    elif root_ok:
        print("⚠️ Partial deployment - root updated but normalize endpoint missing")
    else:
        print("❌ Old deployment still active - waiting for propagation")
    print("=" * 40)

if __name__ == "__main__":
    main()
