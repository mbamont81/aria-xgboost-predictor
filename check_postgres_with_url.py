#!/usr/bin/env python3
"""
Script para conectar usando la URL completa de PostgreSQL
"""

import psycopg2
from psycopg2 import OperationalError
from datetime import datetime
import sys

# URL completa de conexión (External Database URL de tu captura)
DATABASE_URL = "postgresql://aria_audit_db_user:RaXTVt9dqTIRwiDcd3prxjDbf7ga1Ant@dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com/aria_audit_db"

def connect_with_url():
    """Connect using full database URL"""
    try:
        print("🔌 Conectando con URL completa...")
        print(f"   URL: {DATABASE_URL[:50]}...{DATABASE_URL[-20:]}")
        
        conn = psycopg2.connect(DATABASE_URL)
        print("✅ Conexión exitosa!")
        return conn
        
    except OperationalError as e:
        print(f"❌ Error de conexión: {e}")
        
        # Try with SSL disabled
        try:
            print("\n🔄 Intentando sin SSL...")
            conn = psycopg2.connect(DATABASE_URL + "?sslmode=disable")
            print("✅ Conexión exitosa sin SSL!")
            return conn
        except Exception as e2:
            print(f"❌ También falló sin SSL: {e2}")
            
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None

def quick_database_check(conn):
    """Quick check of database content"""
    print("\n🔍 VERIFICACIÓN RÁPIDA DE LA BASE DE DATOS")
    print("=" * 50)
    
    try:
        cursor = conn.cursor()
        
        # Check PostgreSQL version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"📊 PostgreSQL: {version.split(',')[0]}")
        
        # Get all tables
        cursor.execute("""
            SELECT schemaname, tablename, tableowner 
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY tablename
        """)
        
        tables = cursor.fetchall()
        print(f"\n📋 Tablas encontradas: {len(tables)}")
        
        total_records = 0
        for schema, table_name, owner in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                total_records += count
                
                status = "✅" if count > 0 else "❌"
                print(f"  {status} {table_name}: {count:,} registros")
                
                # If table has data, show sample
                if count > 0 and count <= 10:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                    samples = cursor.fetchall()
                    print(f"     📋 Muestra:")
                    for sample in samples:
                        # Truncate long values
                        truncated = []
                        for val in sample:
                            str_val = str(val)
                            if len(str_val) > 30:
                                truncated.append(str_val[:27] + "...")
                            else:
                                truncated.append(str_val)
                        print(f"       {truncated}")
                
            except Exception as e:
                print(f"  ❌ {table_name}: Error - {e}")
        
        print(f"\n📊 Total registros en todas las tablas: {total_records:,}")
        
        # Check for recent activity
        print(f"\n🕒 VERIFICANDO ACTIVIDAD RECIENTE")
        print("-" * 30)
        
        # Look for timestamp columns in any table
        cursor.execute("""
            SELECT table_name, column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'public' 
            AND (column_name LIKE '%created%' OR column_name LIKE '%timestamp%' 
                 OR column_name LIKE '%date%' OR column_name LIKE '%time%')
            ORDER BY table_name, column_name
        """)
        
        timestamp_columns = cursor.fetchall()
        
        if timestamp_columns:
            print("📅 Columnas de fecha/hora encontradas:")
            for table, column in timestamp_columns:
                print(f"   • {table}.{column}")
                
                try:
                    cursor.execute(f"""
                        SELECT {column} FROM {table} 
                        WHERE {column} IS NOT NULL 
                        ORDER BY {column} DESC 
                        LIMIT 1
                    """)
                    
                    latest = cursor.fetchone()
                    if latest:
                        print(f"     Último registro: {latest[0]}")
                except:
                    pass
        else:
            print("❌ No se encontraron columnas de timestamp")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ Error en verificación: {e}")

def main():
    print("🔍 VERIFICACIÓN DE POSTGRESQL CON URL COMPLETA")
    print("=" * 60)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Connect to database
    conn = connect_with_url()
    if not conn:
        print("\n💡 POSIBLES SOLUCIONES:")
        print("1. La contraseña puede haber cambiado")
        print("2. Puede haber restricciones de IP")
        print("3. La base de datos puede estar pausada")
        print("4. Verifica en Render Dashboard si la DB está activa")
        return
    
    try:
        # Quick database check
        quick_database_check(conn)
        
    finally:
        conn.close()
        print("\n✅ Conexión cerrada")
    
    print(f"\n📋 CONCLUSIÓN:")
    print("- Si ves tablas con datos → La base de datos SÍ está siendo usada")
    print("- Si está vacía → No hay logging activo desde tu servicio")

if __name__ == "__main__":
    main()
