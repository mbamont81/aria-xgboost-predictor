#!/usr/bin/env python3
"""
Verificar estructura exacta de la tabla streamed_trades
"""

import psycopg2
from datetime import datetime

DATABASE_URL = "postgresql://aria_audit_db_user:RaXTVt9ddTIRwiDcd3prxjDbf7ga1Ant@dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com/aria_audit_db"

def check_table_structure():
    """Verificar estructura de streamed_trades"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        print("🔍 ESTRUCTURA DE TABLA STREAMED_TRADES")
        print("=" * 60)
        
        # Obtener información de columnas
        cursor.execute("""
            SELECT 
                column_name, 
                data_type, 
                is_nullable, 
                column_default,
                ordinal_position
            FROM information_schema.columns 
            WHERE table_name = 'streamed_trades' AND table_schema = 'public'
            ORDER BY ordinal_position
        """)
        
        columns = cursor.fetchall()
        
        print(f"📊 Total columnas: {len(columns)}")
        print("\n📋 COLUMNAS DISPONIBLES:")
        
        for col_name, data_type, nullable, default, position in columns:
            nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
            default_str = f" DEFAULT {default}" if default else ""
            print(f"  {position:2d}. {col_name:25} | {data_type:20} | {nullable_str}{default_str}")
        
        # Obtener muestra de datos
        print(f"\n📋 MUESTRA DE DATOS (primeros 2 registros):")
        cursor.execute("SELECT * FROM streamed_trades LIMIT 2")
        sample_data = cursor.fetchall()
        
        column_names = [col[0] for col in columns]
        
        for i, record in enumerate(sample_data, 1):
            print(f"\n  📊 Registro {i}:")
            for j, value in enumerate(record):
                if j < len(column_names):
                    # Truncar valores largos
                    display_value = str(value)
                    if len(display_value) > 50:
                        display_value = display_value[:47] + "..."
                    print(f"    {column_names[j]:25}: {display_value}")
        
        # Verificar columnas de timestamp
        print(f"\n🕒 COLUMNAS DE TIMESTAMP:")
        timestamp_columns = []
        for col_name, data_type, _, _, _ in columns:
            if 'timestamp' in data_type.lower() or 'time' in col_name.lower() or 'date' in col_name.lower():
                timestamp_columns.append(col_name)
                print(f"  ✅ {col_name} ({data_type})")
        
        if not timestamp_columns:
            print("  ❌ No se encontraron columnas de timestamp")
        
        # Verificar columnas críticas para análisis
        print(f"\n🎯 COLUMNAS CRÍTICAS PARA ANÁLISIS:")
        critical_columns = ['profit', 'entry_price', 'exit_price', 'confidence', 'market_regime', 'symbol']
        available_critical = []
        
        for critical in critical_columns:
            found = any(col[0] == critical for col in columns)
            status = "✅" if found else "❌"
            print(f"  {status} {critical}")
            if found:
                available_critical.append(critical)
        
        print(f"\n📊 RESUMEN:")
        print(f"  • Total columnas: {len(columns)}")
        print(f"  • Columnas de timestamp: {len(timestamp_columns)}")
        print(f"  • Columnas críticas disponibles: {len(available_critical)}/{len(critical_columns)}")
        
        cursor.close()
        conn.close()
        
        return {
            'columns': columns,
            'timestamp_columns': timestamp_columns,
            'critical_columns': available_critical,
            'column_names': column_names
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    print("🔍 VERIFICACIÓN DE ESTRUCTURA DE STREAMED_TRADES")
    print("=" * 70)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    structure = check_table_structure()
    
    if structure:
        print("\n💡 INFORMACIÓN PARA ANÁLISIS:")
        print("- Usar las columnas de timestamp identificadas para ordenamiento")
        print("- Verificar disponibilidad de columnas críticas antes de análisis")
        print("- Adaptar queries según estructura real de la tabla")

if __name__ == "__main__":
    main()
