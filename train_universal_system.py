"""
Script de Entrenamiento del Sistema Universal Multi-Timeframe
==============================================================
Ejecuta este script para entrenar todos los modelos con los datos cargados
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import glob
from pathlib import Path

# Importar el módulo multi-timeframe
from multi_timeframe_predictor import MultiTimeframePredictor

def scan_available_files(data_dir='./'):
    """
    Escanea y lista todos los archivos OHLCV disponibles
    """
    print("📂 Escaneando archivos disponibles...")
    print("-" * 50)
    
    # Patrones de búsqueda
    patterns = [
        '*_M1_*.csv',
        '*_M5_*.csv',
        '*_M15_*.csv',
        '*_M30_*.csv',
        '*_H1_*.csv',
        '*_H4_*.csv',
        '*_D1_*.csv'
    ]
    
    all_files = {}
    
    for pattern in patterns:
        files = glob.glob(os.path.join(data_dir, pattern))
        for file in files:
            filename = os.path.basename(file)
            # Extraer símbolo y timeframe
            parts = filename.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]
                
                if symbol not in all_files:
                    all_files[symbol] = {}
                
                all_files[symbol][timeframe] = file
                print(f"✅ Encontrado: {symbol} - {timeframe} ({os.path.getsize(file)//1024} KB)")
    
    print(f"\n📊 Total símbolos encontrados: {len(all_files)}")
    for symbol in all_files:
        print(f"   {symbol}: {list(all_files[symbol].keys())}")
    
    return all_files

def verify_file_format(filepath):
    """
    Verifica el formato del archivo CSV
    """
    try:
        # Intentar leer como UTF-16 LE
        with open(filepath, 'rb') as f:
            content = f.read(1000)  # Leer primeros bytes
        
        # Detectar encoding
        if content.startswith(b'\xff\xfe'):  # UTF-16 LE BOM
            return 'utf-16-le'
        elif b'\x00' in content[:100]:  # Probablemente UTF-16
            return 'utf-16-le'
        else:
            return 'utf-8'
    except:
        return 'utf-8'

def load_and_prepare_data(symbol, file_dict):
    """
    Carga y prepara los datos para un símbolo específico
    """
    print(f"\n🔄 Procesando {symbol}...")
    print("-" * 40)
    
    all_data = {}
    
    for timeframe, filepath in file_dict.items():
        try:
            print(f"📊 Cargando {timeframe}...", end=" ")
            
            # Detectar encoding
            encoding = verify_file_format(filepath)
            
            if encoding == 'utf-16-le':
                # Leer como UTF-16 LE
                with open(filepath, 'rb') as f:
                    content = f.read()
                
                text = content.decode('utf-16-le')
                lines = text.strip().split('\n')
                
                # Parsear manualmente
                header = lines[0].split(';')
                data = []
                
                for line in lines[1:]:
                    values = line.split(';')
                    if len(values) >= 5:  # Al menos OHLC
                        data.append(values[:6])  # Tomar solo las primeras 6 columnas
                
                df = pd.DataFrame(data, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
            else:
                # Intentar leer como CSV normal
                df = pd.read_csv(filepath, sep=';')
            
            # Convertir tipos
            df['Time'] = pd.to_datetime(df['Time'], format='%Y.%m.%d %H:%M', errors='coerce')
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Manejar volumen
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
            else:
                df['Volume'] = 0
            
            # Limpiar NaN
            df = df.dropna(subset=['Time', 'Open', 'High', 'Low', 'Close'])
            
            df['Symbol'] = symbol
            df['Timeframe'] = timeframe
            df.set_index('Time', inplace=True)
            
            all_data[timeframe] = df
            print(f"✅ {len(df)} velas")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            continue
    
    return all_data

def train_all_symbols(all_files):
    """
    Entrena modelos para todos los símbolos encontrados
    """
    print("\n" + "="*60)
    print("🚀 INICIANDO ENTRENAMIENTO MULTI-SÍMBOLO")
    print("="*60)
    
    all_models = {}
    all_scalers = {}
    all_metadata = {}
    
    for symbol, file_dict in all_files.items():
        print(f"\n{'='*60}")
        print(f"🎯 ENTRENANDO MODELO PARA {symbol}")
        print(f"{'='*60}")
        
        # Cargar datos
        symbol_data = load_and_prepare_data(symbol, file_dict)
        
        if len(symbol_data) < 2:
            print(f"⚠️ Insuficientes timeframes para {symbol}, omitiendo...")
            continue
        
        # Crear predictor
        predictor = MultiTimeframePredictor()
        
        # Crear dataset de entrenamiento
        print(f"\n🔧 Creando dataset unificado para {symbol}...")
        try:
            X, y_sl, y_tp = predictor.create_training_dataset(symbol_data)
            
            if len(X) < 100:
                print(f"⚠️ Insuficientes muestras para {symbol} ({len(X)}), omitiendo...")
                continue
            
            # Entrenar modelo
            predictor.train_universal_model(X, y_sl, y_tp)
            
            # Guardar modelos del símbolo
            all_models[symbol] = predictor.models
            all_scalers[symbol] = predictor.scalers
            all_metadata[symbol] = {
                'feature_columns': predictor.feature_columns,
                'timeframes': list(symbol_data.keys()),
                'total_samples': len(X),
                'trained_at': datetime.now().isoformat()
            }
            
            print(f"✅ Modelo para {symbol} completado")
            
        except Exception as e:
            print(f"❌ Error entrenando {symbol}: {str(e)[:100]}")
            continue
    
    return all_models, all_scalers, all_metadata

def save_universal_models(models, scalers, metadata):
    """
    Guarda todos los modelos en un archivo único
    """
    print("\n" + "="*60)
    print("💾 GUARDANDO MODELOS UNIVERSALES")
    print("="*60)
    
    # Preparar estructura de datos
    universal_data = {
        'models': models,
        'scalers': scalers,
        'metadata': metadata,
        'system_info': {
            'version': 'universal_multi_v2',
            'created_at': datetime.now().isoformat(),
            'total_symbols': len(models),
            'symbols': list(models.keys()),
            'description': 'Sistema universal multi-timeframe para trading'
        }
    }
    
    # Guardar archivo principal
    output_file = 'xgboost_universal_models.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(universal_data, f)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✅ Archivo guardado: {output_file}")
    print(f"📊 Tamaño: {file_size_mb:.2f} MB")
    
    # Guardar resumen en texto
    summary_file = 'model_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("RESUMEN DE MODELOS ENTRENADOS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Fecha: {datetime.now()}\n")
        f.write(f"Versión: universal_multi_v2\n\n")
        
        for symbol in models:
            f.write(f"\n{symbol}:\n")
            f.write(f"  - Timeframes: {metadata[symbol]['timeframes']}\n")
            f.write(f"  - Muestras: {metadata[symbol]['total_samples']}\n")
            f.write(f"  - Modelos: {list(models[symbol].keys())}\n")
    
    print(f"📄 Resumen guardado: {summary_file}")
    
    return output_file

def test_predictions(model_file):
    """
    Prueba las predicciones con diferentes configuraciones
    """
    print("\n" + "="*60)
    print("🧪 PROBANDO PREDICCIONES")
    print("="*60)
    
    # Cargar modelos
    with open(model_file, 'rb') as f:
        data = pickle.load(f)
    
    models = data['models']
    scalers = data['scalers']
    metadata = data['metadata']
    
    # Características de prueba
    test_features = {
        'returns': 0.001,
        'volatility_normalized': 0.015,
        'atr_normalized': 1.2,
        'rsi': 55,
        'rsi_slope': 2,
        'macd_normalized': 0.05,
        'bb_width_normalized': 2.5,
        'bb_position': 0.6,
        'ma_cross_normalized': 0.5,
        'volume_ratio': 1.2,
        'body_ratio': 0.65,
        'momentum_5': 0.01,
        'momentum_10': 0.015,
        'momentum_20': 0.02,
        'trend_strength': 0.8,
        'volatility_regime': 0.02,
        'hurst': 0.55,
        'autocorr': 0.3,
        'higher_high': 1,
        'lower_low': 0,
        'upper_shadow_ratio': 0.2,
        'lower_shadow_ratio': 0.2,
        'volume_trend': 1.1
    }
    
    print("\n📊 Predicciones de ejemplo:\n")
    
    # Probar para cada símbolo
    for symbol in list(models.keys())[:5]:  # Primeros 5 símbolos
        print(f"\n{symbol}:")
        print("-" * 30)
        
        predictor = MultiTimeframePredictor()
        predictor.models = models[symbol]
        predictor.scalers = scalers[symbol]
        predictor.feature_columns = metadata[symbol]['feature_columns']
        
        # Probar diferentes timeframes
        for tf in ['M1', 'M5', 'M15', 'H1', 'H4']:
            try:
                pred = predictor.predict(test_features, timeframe=tf, symbol=symbol)
                print(f"  {tf}: SL={pred['sl_pips']:6.1f} | TP={pred['tp_pips']:6.1f} | "
                      f"R:R={pred['risk_reward_ratio']:.1f} | {pred['regime']}")
            except:
                print(f"  {tf}: No disponible")

def main():
    """
    Función principal
    """
    print("\n" + "🚀 "*20)
    print("SISTEMA DE ENTRENAMIENTO UNIVERSAL MULTI-TIMEFRAME")
    print("🚀 "*20)
    print(f"\nTimestamp: {datetime.now()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working directory: {os.getcwd()}")
    
    # 1. Escanear archivos disponibles
    all_files = scan_available_files()
    
    if not all_files:
        print("\n❌ No se encontraron archivos OHLCV")
        print("Asegúrate de que los archivos estén en el directorio actual")
        return
    
    # 2. Confirmación del usuario
    print("\n" + "="*60)
    print("📋 CONFIRMACIÓN")
    print("="*60)
    print(f"Se encontraron {len(all_files)} símbolos para procesar")
    print("Esto puede tomar 10-30 minutos dependiendo de la cantidad de datos")
    
    response = input("\n¿Deseas continuar? (s/n): ")
    if response.lower() != 's':
        print("❌ Entrenamiento cancelado")
        return
    
    # 3. Entrenar modelos
    models, scalers, metadata = train_all_symbols(all_files)
    
    if not models:
        print("\n❌ No se pudieron entrenar modelos")
        return
    
    # 4. Guardar modelos
    model_file = save_universal_models(models, scalers, metadata)
    
    # 5. Probar predicciones
    test_predictions(model_file)
    
    # 6. Instrucciones finales
    print("\n" + "="*60)
    print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*60)
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("1. El archivo 'xgboost_universal_models.pkl' está listo")
    print("2. Actualiza tu servidor FastAPI con el nuevo modelo")
    print("3. Sube el modelo a GitHub")
    print("4. Despliega en Render")
    
    print("\n💡 TIPS:")
    print("- El modelo ahora soporta múltiples timeframes")
    print("- Ajusta automáticamente SL/TP según el timeframe")
    print("- Puedes agregar más símbolos sin reentrenar todo")
    
    print("\n🎯 Archivo generado: xgboost_universal_models.pkl")
    print("📊 Este archivo contiene todos los modelos entrenados")

if __name__ == "__main__":
    main()
