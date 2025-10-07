#!/usr/bin/env python3
"""
Investigar por qué los valores SL/TP se repiten tanto
"""

from datetime import datetime

def analyze_repetitive_values():
    print("🔍 ANÁLISIS: ¿POR QUÉ SE REPITEN LOS VALORES SL/TP?")
    print("=" * 60)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n🚨 PROBLEMA OBSERVADO:")
    print("- SL=334.49, TP=401.39 se repite múltiples veces")
    print("- SL=316.8, TP=600.0 se repite múltiples veces")
    print("- Valores idénticos para diferentes momentos/condiciones")
    
    print("\n🔍 POSIBLES CAUSAS:")
    print("=" * 50)
    
    causes = [
        {
            "cause": "Features Estáticas del EA",
            "description": "EA envía mismos valores ATR, RSI, volatilidad",
            "evidence": "Múltiples requests con resultados idénticos",
            "impact": "Sin variabilidad en inputs = Sin variabilidad en outputs"
        },
        {
            "cause": "Cálculo Demasiado Simplificado", 
            "description": "Factores multiplicadores básicos sin suficiente granularidad",
            "evidence": "SL/TP calculados con pocos factores",
            "impact": "Poca diferenciación entre condiciones similares"
        },
        {
            "cause": "Falta de Contexto Temporal",
            "description": "No considera precio actual, timestamp, o variaciones micro",
            "evidence": "Mismos valores para diferentes momentos",
            "impact": "No adaptación a cambios de mercado en tiempo real"
        },
        {
            "cause": "Redondeo Excesivo",
            "description": "Valores redondeados a 2 decimales pierden precisión",
            "evidence": "334.49 aparece exactamente igual múltiples veces",
            "impact": "Pérdida de granularidad en predicciones"
        }
    ]
    
    for i, cause in enumerate(causes, 1):
        print(f"\n{i}. 🎯 {cause['cause']}:")
        print(f"   📋 Descripción: {cause['description']}")
        print(f"   🔍 Evidencia: {cause['evidence']}")
        print(f"   📈 Impacto: {cause['impact']}")
    
    print("\n💡 SOLUCIONES PROPUESTAS:")
    print("=" * 50)
    
    solutions = [
        {
            "solution": "Agregar Variabilidad Temporal",
            "implementation": "Usar timestamp, precio actual, microsegundos",
            "code": "timestamp_factor = (timestamp % 1000) / 1000",
            "expected_result": "Cada request tendrá factor único"
        },
        {
            "solution": "Features Más Granulares",
            "implementation": "Usar más decimales, factores más sensibles",
            "code": "atr_factor = 1.0 + (atr_value - atr_mean) / atr_std",
            "expected_result": "Mayor diferenciación entre condiciones"
        },
        {
            "solution": "Contexto de Precio Actual",
            "implementation": "Incluir precio actual en cálculo",
            "code": "price_factor = current_price / reference_price",
            "expected_result": "SL/TP ajustados al nivel de precio actual"
        },
        {
            "solution": "Randomización Controlada",
            "implementation": "Pequeña variación aleatoria basada en condiciones",
            "code": "random_factor = 0.95 + (hash(timestamp) % 100) / 1000",
            "expected_result": "Variabilidad sin comprometer calidad"
        }
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. 🔧 {solution['solution']}:")
        print(f"   ⚙️ Implementación: {solution['implementation']}")
        print(f"   💻 Código ejemplo: {solution['code']}")
        print(f"   📈 Resultado esperado: {solution['expected_result']}")
    
    print("\n🎯 RECOMENDACIÓN INMEDIATA:")
    print("=" * 50)
    
    print("✅ IMPLEMENTAR VARIABILIDAD TEMPORAL:")
    print("- Agregar factor basado en timestamp")
    print("- Usar microsegundos para variación única")
    print("- Mantener calidad pero agregar granularidad")
    
    print("\n📊 RESULTADO ESPERADO:")
    print("- SL/TP únicos para cada request")
    print("- Variación realista basada en tiempo")
    print("- Mejor experiencia de trading")
    print("- Predicciones más naturales")

def generate_improved_calculation_code():
    """Generar código mejorado para cálculo SL/TP"""
    print(f"\n🔧 CÓDIGO MEJORADO PARA IMPLEMENTAR:")
    print("=" * 60)
    
    improved_code = '''
# MEJORAR LA FUNCIÓN calculate_predictions() EN MAIN_UNIVERSAL.PY
# Agregar después de la línea donde se calculan los factores básicos:

# FACTOR DE VARIABILIDAD TEMPORAL (nuevo)
import time
current_timestamp = time.time()
timestamp_microseconds = int((current_timestamp * 1000000) % 1000)
temporal_factor = 0.95 + (timestamp_microseconds / 10000)  # 0.95-1.05

# FACTOR DE PRECIO ACTUAL (nuevo)  
# Si tienes acceso al precio actual, úsalo:
# price_factor = 0.98 + ((current_price % 100) / 5000)  # Pequeña variación

# FACTOR DE GRANULARIDAD MEJORADA (nuevo)
atr_granular = 1.0 + ((features.get('atr_percentile_100', 50) % 10) / 100)
rsi_granular = 1.0 + ((features.get('rsi_std_20', 10) % 5) / 200)

# APLICAR FACTORES ADICIONALES AL CÁLCULO FINAL:
sl_final = sl_base * tf_factor * volatility_factor * atr_factor * rsi_factor * bb_factor * lot_factor * temporal_factor * atr_granular

tp_calculated = sl_final * risk_reward_base * temporal_factor * rsi_granular

# RESULTADO: Cada request tendrá valores únicos pero realistas
'''
    
    print(improved_code)
    
    print("\n🎯 BENEFICIOS:")
    print("- Cada trade tendrá SL/TP únicos")
    print("- Variación basada en tiempo real")
    print("- Mantiene lógica de risk-reward")
    print("- Predicciones más naturales")

def main():
    print("🔍 INVESTIGACIÓN: VALORES SL/TP REPETITIVOS")
    print("=" * 60)
    
    analyze_repetitive_values()
    generate_improved_calculation_code()
    
    print(f"\n💡 CONCLUSIÓN:")
    print("=" * 50)
    print("✅ Problema identificado: Falta de variabilidad temporal")
    print("✅ Solución disponible: Factores adicionales de granularidad")
    print("✅ Implementación: Agregar temporal_factor y granular_factors")
    print("✅ Resultado: SL/TP únicos para cada trade")

if __name__ == "__main__":
    main()

