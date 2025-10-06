#!/usr/bin/env python3
"""
Analizar los logs de Render para identificar símbolos que fallan
"""

import re
from collections import defaultdict, Counter

def analyze_failing_symbols():
    """Analizar símbolos que están fallando basado en los logs"""
    print("🔍 ANALIZANDO SÍMBOLOS QUE FALLAN EN LOS LOGS")
    print("=" * 60)
    
    # Símbolos que veo fallando en los logs
    failing_symbols_from_logs = [
        "XAUUSD.m",    # XAUUSD con sufijo .m
        "EURJPYc",     # EURJPY con sufijo c
        "USTEC.f",     # USTEC con sufijo .f
        "GBPJPY",      # GBPJPY sin modelo
        "EURNZDc",     # EURNZD con sufijo c
        "NZDJPY",      # NZDJPY sin modelo
        "USDJPYm",     # USDJPY con sufijo m
        "EURCHFc",     # EURCHF con sufijo c
        "BTCUSDc",     # BTCUSD con sufijo c
        "EURGBPc",     # EURGBP con sufijo c
        "AUDCADc",     # AUDCAD con sufijo c
    ]
    
    # Contar por tipo de problema
    suffix_patterns = defaultdict(list)
    for symbol in failing_symbols_from_logs:
        if symbol.endswith('c'):
            suffix_patterns['Sufijo "c"'].append(symbol)
        elif symbol.endswith('.m'):
            suffix_patterns['Sufijo ".m"'].append(symbol)
        elif symbol.endswith('.f'):
            suffix_patterns['Sufijo ".f"'].append(symbol)
        elif symbol.endswith('m') and len(symbol) > 6:
            suffix_patterns['Sufijo "m"'].append(symbol)
        else:
            suffix_patterns['Sin sufijo - Posible modelo faltante'].append(symbol)
    
    print("📊 SÍMBOLOS PROBLEMÁTICOS POR CATEGORÍA:")
    print("-" * 60)
    
    total_failing = 0
    for pattern, symbols in suffix_patterns.items():
        print(f"📋 {pattern}: {len(symbols)} símbolos")
        total_failing += len(symbols)
        for symbol in sorted(symbols):
            print(f"   ❌ {symbol}")
        print()
    
    print(f"📊 TOTAL SÍMBOLOS PROBLEMÁTICOS: {total_failing}")
    
    return suffix_patterns

def suggest_normalizations(suffix_patterns):
    """Sugerir normalizaciones para símbolos que fallan"""
    print("\n" + "=" * 60)
    print("💡 SUGERENCIAS DE NORMALIZACIÓN ADICIONAL")
    print("=" * 60)
    
    suggestions = []
    
    # Procesar cada categoría
    for pattern, symbols in suffix_patterns.items():
        print(f"\n🔧 {pattern}:")
        print("-" * 40)
        
        for symbol in symbols:
            if symbol.endswith('c'):
                base = symbol[:-1]
                suggestions.append((symbol, base, "Remover sufijo 'c'"))
                print(f"   🔄 {symbol:12} → {base:12} | Remover sufijo 'c'")
            
            elif symbol.endswith('.m'):
                base = symbol[:-2]
                suggestions.append((symbol, base, "Remover sufijo '.m'"))
                print(f"   🔄 {symbol:12} → {base:12} | Remover sufijo '.m'")
            
            elif symbol.endswith('.f'):
                base = symbol[:-2]
                # USTEC.f debería ir a NAS100
                if base == 'USTEC':
                    final = 'NAS100'
                    suggestions.append((symbol, final, "USTEC → NAS100"))
                    print(f"   🔄 {symbol:12} → {final:12} | Mapeo específico USTEC → NAS100")
                else:
                    suggestions.append((symbol, base, "Remover sufijo '.f'"))
                    print(f"   🔄 {symbol:12} → {base:12} | Remover sufijo '.f'")
            
            elif symbol.endswith('m') and len(symbol) > 6:
                base = symbol[:-1]
                suggestions.append((symbol, base, "Remover sufijo 'm'"))
                print(f"   🔄 {symbol:12} → {base:12} | Remover sufijo 'm'")
            
            else:
                # Símbolos sin sufijo que pueden necesitar modelos
                print(f"   ⚠️ {symbol:12} → Verificar si existe modelo entrenado")
    
    return suggestions

def check_available_models():
    """Mostrar modelos disponibles según los logs"""
    print("\n" + "=" * 60)
    print("📋 MODELOS DISPONIBLES (según logs exitosos)")
    print("=" * 60)
    
    # Símbolos que veo funcionando en los logs
    working_symbols = [
        "XAUUSD",    # Gold - funciona perfectamente
        "BTCUSD",    # Bitcoin - funciona
        "GBPUSD",    # GBP/USD - funciona
        "EURUSD",    # EUR/USD - funciona
        "US500",     # S&P 500 - funciona
        "UK100",     # FTSE 100 - funciona
        "ETHUSD",    # Ethereum - funciona
        "SOLUSD",    # Solana - funciona
        "XAGUSD",    # Silver - funciona
        "AUDCHF",    # AUD/CHF - funciona (mapeo de AUDCAD)
    ]
    
    print("✅ SÍMBOLOS CON MODELOS FUNCIONANDO:")
    for symbol in sorted(working_symbols):
        print(f"   ✅ {symbol}")
    
    print(f"\n📊 Total modelos funcionando: {len(working_symbols)}")

def main():
    print("🚀 ANÁLISIS COMPLETO DE SÍMBOLOS PROBLEMÁTICOS")
    print("=" * 60)
    
    suffix_patterns = analyze_failing_symbols()
    suggestions = suggest_normalizations(suffix_patterns)
    check_available_models()
    
    print("\n" + "=" * 60)
    print("🎯 RECOMENDACIONES PRIORITARIAS:")
    print("=" * 60)
    print("1. 🔧 Agregar normalización para sufijos '.m', '.f', 'c'")
    print("2. 🔄 Mapear USTEC.f → NAS100")
    print("3. 📋 Verificar si GBPJPY, NZDJPY necesitan modelos")
    print("4. 🧹 Limpiar sufijos de broker automáticamente")
    print("=" * 60)

if __name__ == "__main__":
    main()
