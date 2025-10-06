#!/usr/bin/env python3
"""
Script para revisar las bases de datos SQLite y verificar si se están guardando datos de trading
"""

import sqlite3
import os
from datetime import datetime

def check_database(db_path, db_name):
    """Check database tables and recent data"""
    if not os.path.exists(db_path):
        print(f"❌ {db_name}: No existe")
        return
    
    print(f"\n🔍 REVISANDO: {db_name}")
    print("=" * 50)
    print(f"📁 Ubicación: {db_path}")
    print(f"📊 Tamaño: {os.path.getsize(db_path):,} bytes")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"📋 Tablas encontradas: {len(tables)}")
        
        for table in tables:
            table_name = table[0]
            print(f"\n  📊 Tabla: {table_name}")
            
            # Count records
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"     📈 Registros: {count:,}")
            
            if count > 0:
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                col_names = [col[1] for col in columns]
                print(f"     📝 Columnas: {', '.join(col_names)}")
                
                # Get recent records
                try:
                    # Try different timestamp column names
                    timestamp_cols = ['created_at', 'timestamp', 'fecha', 'date']
                    timestamp_col = None
                    
                    for col in timestamp_cols:
                        if col in col_names:
                            timestamp_col = col
                            break
                    
                    if timestamp_col:
                        cursor.execute(f"SELECT * FROM {table_name} ORDER BY {timestamp_col} DESC LIMIT 3")
                        recent = cursor.fetchall()
                        print(f"     🕒 Últimos registros:")
                        for i, record in enumerate(recent, 1):
                            print(f"       {i}. {record}")
                    else:
                        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                        sample = cursor.fetchall()
                        print(f"     📋 Muestra de registros:")
                        for i, record in enumerate(sample, 1):
                            print(f"       {i}. {record}")
                            
                except Exception as e:
                    print(f"     ⚠️  Error leyendo registros: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error accediendo a la base de datos: {e}")

def main():
    print("🔍 REVISIÓN DE BASES DE DATOS SQLITE")
    print("=" * 60)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Database paths
    databases = [
        {
            "path": r"C:\Users\Turbo 2\AppData\Roaming\MetaQuotes\Terminal\0727F3F88B5F0FE006962B330B91FF37\MQL4\Experts\aria-audit-api-main\aria_audit.db",
            "name": "Aria Audit DB (Auditoría de Trades)"
        },
        {
            "path": r"C:\Users\Turbo 2\AppData\Roaming\MetaQuotes\Terminal\0727F3F88B5F0FE006962B330B91FF37\MQL4\Experts\xgboost\trading_system_live.db",
            "name": "Trading System Live DB (Predicciones)"
        },
        {
            "path": r"C:\Users\Turbo 2\AppData\Roaming\MetaQuotes\Terminal\0727F3F88B5F0FE006962B330B91FF37\MQL4\Experts\xgboost\trading_system_live_fixed.db",
            "name": "Trading System Fixed DB (Predicciones Corregidas)"
        }
    ]
    
    for db in databases:
        check_database(db["path"], db["name"])
    
    print(f"\n✅ Revisión completada")
    print("=" * 60)
    
    # Summary
    print("\n📋 RESUMEN:")
    print("- Si ves registros recientes → Los datos SÍ se están guardando")
    print("- Si las tablas están vacías → No hay recopilación activa")
    print("- Si hay muchos registros → El sistema está funcionando")

if __name__ == "__main__":
    main()
