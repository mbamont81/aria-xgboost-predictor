#!/usr/bin/env python3
"""
Script para conectar con la URL correcta de PostgreSQL
"""

import psycopg2
from psycopg2 import OperationalError
from datetime import datetime
import sys

# URL correcta de conexión
DATABASE_URL = "postgresql://aria_audit_db_user:RaXTVt9ddTIRwiDcd3prxjDbf7ga1Ant@dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com/aria_audit_db"

def connect_with_correct_url():
    """Connect using correct database URL"""
    try:
        print("🔌 Conectando con URL correcta...")
        print(f"   Host: dpg-d20ht36uk2gs73c9tmbg-a.oregon-postgres.render.com")
        print(f"   Database: aria_audit_db")
        print(f"   User: aria_audit_db_user")
        
        conn = psycopg2.connect(DATABASE_URL)
        print("✅ Conexión exitosa!")
        return conn
        
    except OperationalError as e:
        print(f"❌ Error de conexión: {e}")
        return None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None

def comprehensive_database_analysis(conn):
    """Comprehensive analysis of the database"""
    print("\n🔍 ANÁLISIS COMPLETO DE LA BASE DE DATOS")
    print("=" * 60)
    
    try:
        cursor = conn.cursor()
        
        # PostgreSQL version
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"📊 PostgreSQL: {version.split(',')[0]}")
        
        # Database size
        cursor.execute("SELECT pg_size_pretty(pg_database_size('aria_audit_db'));")
        db_size = cursor.fetchone()[0]
        print(f"💾 Tamaño de la base de datos: {db_size}")
        
        # Get all tables with detailed info
        cursor.execute("""
            SELECT 
                schemaname, 
                tablename, 
                tableowner,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
            FROM pg_tables 
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)
        
        tables = cursor.fetchall()
        print(f"\n📋 Tablas encontradas: {len(tables)}")
        
        if not tables:
            print("❌ No hay tablas en la base de datos")
            cursor.close()
            return
        
        total_records = 0
        active_tables = []
        
        for schema, table_name, owner, size in tables:
            print(f"\n  📊 Tabla: {table_name}")
            print(f"     👤 Owner: {owner}")
            print(f"     💾 Tamaño: {size}")
            
            try:
                # Count records
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                total_records += count
                
                status = "✅" if count > 0 else "❌"
                print(f"     📈 Registros: {count:,} {status}")
                
                if count > 0:
                    active_tables.append(table_name)
                    
                    # Get column info
                    cursor.execute("""
                        SELECT column_name, data_type, is_nullable, column_default
                        FROM information_schema.columns 
                        WHERE table_name = %s AND table_schema = 'public'
                        ORDER BY ordinal_position
                    """, (table_name,))
                    
                    columns = cursor.fetchall()
                    print(f"     📝 Columnas ({len(columns)}):")
                    for col_name, data_type, nullable, default in columns[:5]:  # Show first 5 columns
                        nullable_str = "NULL" if nullable == "YES" else "NOT NULL"
                        print(f"       • {col_name}: {data_type} {nullable_str}")
                    
                    if len(columns) > 5:
                        print(f"       ... y {len(columns) - 5} columnas más")
                    
                    # Get sample data
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 2")
                    samples = cursor.fetchall()
                    
                    if samples:
                        print(f"     📋 Muestra de datos:")
                        for i, sample in enumerate(samples, 1):
                            truncated = []
                            for val in sample:
                                str_val = str(val)
                                if len(str_val) > 40:
                                    truncated.append(str_val[:37] + "...")
                                else:
                                    truncated.append(str_val)
                            print(f"       {i}. {truncated}")
                
            except Exception as e:
                print(f"     ❌ Error analizando tabla: {e}")
        
        print(f"\n📊 RESUMEN GENERAL:")
        print(f"   • Total tablas: {len(tables)}")
        print(f"   • Tablas con datos: {len(active_tables)}")
        print(f"   • Total registros: {total_records:,}")
        print(f"   • Tamaño total: {db_size}")
        
        # Check for recent activity
        if active_tables:
            print(f"\n🕒 ACTIVIDAD RECIENTE:")
            print("-" * 30)
            
            for table_name in active_tables:
                try:
                    # Look for timestamp columns
                    cursor.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s AND table_schema = 'public'
                        AND (column_name LIKE '%%created%%' OR column_name LIKE '%%timestamp%%' 
                             OR column_name LIKE '%%date%%' OR column_name LIKE '%%time%%'
                             OR column_name LIKE '%%updated%%')
                        ORDER BY column_name
                    """, (table_name,))
                    
                    timestamp_cols = cursor.fetchall()
                    
                    if timestamp_cols:
                        timestamp_col = timestamp_cols[0][0]
                        cursor.execute(f"""
                            SELECT {timestamp_col} 
                            FROM {table_name} 
                            WHERE {timestamp_col} IS NOT NULL
                            ORDER BY {timestamp_col} DESC 
                            LIMIT 1
                        """)
                        
                        latest = cursor.fetchone()
                        if latest:
                            print(f"   📅 {table_name}: {latest[0]}")
                            
                except Exception as e:
                    print(f"   ❌ Error verificando {table_name}: {e}")
        
        # Check for specific trading/prediction tables
        print(f"\n🎯 VERIFICACIÓN DE TABLAS DE TRADING:")
        print("-" * 40)
        
        trading_tables = ['predictions', 'trades', 'signals', 'audits', 'trading_sessions', 'performance']
        
        for table_name in trading_tables:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = %s
                )
            """, (table_name,))
            
            exists = cursor.fetchone()[0]
            
            if exists:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                status = "✅" if count > 0 else "⚠️"
                print(f"   {status} {table_name}: {count:,} registros")
            else:
                print(f"   ❌ {table_name}: No existe")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")

def main():
    print("🔍 ANÁLISIS COMPLETO DE POSTGRESQL EN RENDER")
    print("=" * 70)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Base de datos: aria_audit_db")
    print(f"🔗 Host: dpg-d20ht36uk2gs73c9tmbg-a")
    
    # Connect to database
    conn = connect_with_correct_url()
    if not conn:
        print("\n💡 SOLUCIONES:")
        print("1. Verifica que la base de datos esté activa en Render")
        print("2. Usa el comando: render psql dpg-d20ht36uk2gs73c9tmbg-a")
        print("3. Verifica las credenciales en el dashboard")
        return
    
    try:
        # Comprehensive analysis
        comprehensive_database_analysis(conn)
        
    finally:
        conn.close()
        print("\n✅ Conexión cerrada")
    
    print(f"\n📋 CONCLUSIONES:")
    print("- Si ves tablas con datos recientes → Los datos SÍ se están guardando")
    print("- Si las tablas están vacías → No hay logging activo")
    print("- Si no hay tablas de trading → Falta implementar el sistema de logging")

if __name__ == "__main__":
    main()
