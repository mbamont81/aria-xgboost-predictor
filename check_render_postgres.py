#!/usr/bin/env python3
"""
Script para conectar a la base de datos PostgreSQL de Render y verificar datos
"""

import psycopg2
from psycopg2 import OperationalError
from datetime import datetime
import sys

# Datos de conexión de Render
DB_CONFIG = {
    'host': 'dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com',
    'port': '5432',
    'database': 'aria_audit_db',
    'user': 'aria_audit_db_user',
    'password': 'RaXTVt9dqTIRwiDcd3prxjDbf7ga1Ant'
}

def connect_to_database():
    """Connect to PostgreSQL database"""
    try:
        print("🔌 Conectando a PostgreSQL en Render...")
        print(f"   Host: {DB_CONFIG['host']}")
        print(f"   Database: {DB_CONFIG['database']}")
        print(f"   User: {DB_CONFIG['user']}")
        
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Conexión exitosa!")
        return conn
        
    except OperationalError as e:
        print(f"❌ Error de conexión: {e}")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None

def check_database_structure(conn):
    """Check database tables and structure"""
    print("\n🔍 REVISANDO ESTRUCTURA DE LA BASE DE DATOS")
    print("=" * 60)
    
    try:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name, table_type 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print(f"📋 Tablas encontradas: {len(tables)}")
        
        for table_name, table_type in tables:
            print(f"\n  📊 {table_name} ({table_type})")
            
            # Get column information
            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns = cursor.fetchall()
            print(f"     📝 Columnas ({len(columns)}):")
            for col_name, data_type, nullable, default in columns:
                nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                default_str = f" DEFAULT {default}" if default else ""
                print(f"       • {col_name}: {data_type} {nullable_str}{default_str}")
            
            # Count records
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"     📈 Registros: {count:,}")
        
        cursor.close()
        return tables
        
    except Exception as e:
        print(f"❌ Error revisando estructura: {e}")
        return []

def check_recent_data(conn, tables):
    """Check recent data in tables"""
    print("\n📊 REVISANDO DATOS RECIENTES")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    for table_name, _ in tables:
        try:
            print(f"\n🔍 Tabla: {table_name}")
            
            # Get recent records
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_count = cursor.fetchone()[0]
            
            if total_count == 0:
                print("   ❌ Sin registros")
                continue
                
            print(f"   📊 Total registros: {total_count:,}")
            
            # Try to find timestamp columns
            cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND table_schema = 'public'
                AND (column_name LIKE '%%created%%' OR column_name LIKE '%%timestamp%%' 
                     OR column_name LIKE '%%date%%' OR column_name LIKE '%%time%%')
                ORDER BY column_name
            """, (table_name,))
            
            timestamp_cols = [row[0] for row in cursor.fetchall()]
            
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]  # Use first timestamp column
                print(f"   🕒 Ordenando por: {timestamp_col}")
                
                # Get recent records
                cursor.execute(f"""
                    SELECT * FROM {table_name} 
                    ORDER BY {timestamp_col} DESC 
                    LIMIT 3
                """)
                
                recent_records = cursor.fetchall()
                
                # Get column names for display
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position
                """, (table_name,))
                
                column_names = [row[0] for row in cursor.fetchall()]
                
                print(f"   📋 Últimos {len(recent_records)} registros:")
                for i, record in enumerate(recent_records, 1):
                    print(f"     {i}. Registro:")
                    for j, value in enumerate(record):
                        if j < len(column_names):
                            # Truncate long values
                            display_value = str(value)
                            if len(display_value) > 50:
                                display_value = display_value[:47] + "..."
                            print(f"        {column_names[j]}: {display_value}")
                    print()
            else:
                # No timestamp columns, just show sample records
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_records = cursor.fetchall()
                
                print(f"   📋 Muestra de {len(sample_records)} registros:")
                for i, record in enumerate(sample_records, 1):
                    print(f"     {i}. {record}")
                    
        except Exception as e:
            print(f"   ❌ Error revisando {table_name}: {e}")
    
    cursor.close()

def check_prediction_data(conn):
    """Check specifically for prediction/trading data"""
    print("\n🎯 BUSCANDO DATOS DE PREDICCIONES/TRADING")
    print("=" * 60)
    
    cursor = conn.cursor()
    
    # Look for tables that might contain prediction data
    prediction_tables = ['predictions', 'trades', 'signals', 'audits', 'trading_sessions']
    
    for table_name in prediction_tables:
        try:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (table_name,))
            
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"✅ {table_name}: {count:,} registros")
                
                if count > 0:
                    # Get recent data
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                    samples = cursor.fetchall()
                    print(f"   📋 Muestra:")
                    for sample in samples:
                        print(f"     {sample}")
            else:
                print(f"❌ {table_name}: No existe")
                
        except Exception as e:
            print(f"❌ Error verificando {table_name}: {e}")
    
    cursor.close()

def main():
    print("🔍 VERIFICACIÓN DE BASE DE DATOS POSTGRESQL EN RENDER")
    print("=" * 70)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Base de datos: {DB_CONFIG['database']}")
    
    # Check if psycopg2 is available
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 no está instalado. Instálalo con: pip install psycopg2-binary")
        sys.exit(1)
    
    # Connect to database
    conn = connect_to_database()
    if not conn:
        print("❌ No se pudo conectar a la base de datos")
        return
    
    try:
        # Check database structure
        tables = check_database_structure(conn)
        
        if tables:
            # Check recent data
            check_recent_data(conn, tables)
            
            # Check prediction-specific data
            check_prediction_data(conn)
        else:
            print("❌ No se encontraron tablas en la base de datos")
        
    finally:
        conn.close()
        print("\n✅ Conexión cerrada")
    
    print("\n📋 RESUMEN:")
    print("- Si ves tablas con registros recientes → Los datos SÍ se están guardando")
    print("- Si las tablas están vacías → El sistema no está guardando datos")
    print("- Si no hay tablas de predicciones → Falta configurar el logging")

if __name__ == "__main__":
    main()
