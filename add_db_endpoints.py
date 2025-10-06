#!/usr/bin/env python3
"""
Código para agregar endpoints de verificación de base de datos a main.py
"""

def generate_db_code():
    """Generate code to add to main.py for database verification"""
    
    code = '''
# ===== AGREGAR ESTAS IMPORTACIONES AL INICIO =====
import os
import psycopg2
from psycopg2 import OperationalError
import json

# ===== AGREGAR ESTAS FUNCIONES DESPUÉS DE LAS EXISTENTES =====

def get_database_connection():
    """Get PostgreSQL database connection"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return None, "DATABASE_URL not configured"
        
        conn = psycopg2.connect(database_url)
        return conn, None
    except Exception as e:
        return None, str(e)

def check_database_tables():
    """Check what tables exist and their record counts"""
    conn, error = get_database_connection()
    if error:
        return {"error": error, "connected": False}
    
    try:
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        table_info = {}
        
        for (table_name,) in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                table_info[table_name] = {"records": count}
                
                # Get recent record if exists
                cursor.execute(f"""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = %s AND table_schema = 'public'
                    AND (column_name LIKE '%%created%%' OR column_name LIKE '%%timestamp%%')
                    LIMIT 1
                """, (table_name,))
                
                timestamp_col = cursor.fetchone()
                if timestamp_col and count > 0:
                    cursor.execute(f"""
                        SELECT {timestamp_col[0]} 
                        FROM {table_name} 
                        ORDER BY {timestamp_col[0]} DESC 
                        LIMIT 1
                    """)
                    latest = cursor.fetchone()
                    if latest:
                        table_info[table_name]["latest_record"] = str(latest[0])
                        
            except Exception as e:
                table_info[table_name] = {"error": str(e)}
        
        cursor.close()
        conn.close()
        
        return {
            "connected": True,
            "tables": table_info,
            "total_tables": len(tables)
        }
        
    except Exception as e:
        conn.close()
        return {"error": str(e), "connected": False}

def get_prediction_stats():
    """Get statistics about predictions if table exists"""
    conn, error = get_database_connection()
    if error:
        return {"error": error}
    
    try:
        cursor = conn.cursor()
        
        # Check if predictions table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = 'predictions'
            )
        """)
        
        if not cursor.fetchone()[0]:
            return {"predictions_table": False, "message": "No predictions table found"}
        
        # Get prediction statistics
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        stats = {
            "predictions_table": True,
            "total_predictions": total_predictions
        }
        
        if total_predictions > 0:
            # Get date range
            cursor.execute("""
                SELECT MIN(created_at), MAX(created_at) 
                FROM predictions 
                WHERE created_at IS NOT NULL
            """)
            date_range = cursor.fetchone()
            if date_range[0]:
                stats["first_prediction"] = str(date_range[0])
                stats["last_prediction"] = str(date_range[1])
            
            # Get recent predictions
            cursor.execute("""
                SELECT symbol, confidence, created_at 
                FROM predictions 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            recent = cursor.fetchall()
            stats["recent_predictions"] = [
                {
                    "symbol": r[0] if r[0] else "unknown",
                    "confidence": float(r[1]) if r[1] else 0,
                    "timestamp": str(r[2]) if r[2] else "unknown"
                }
                for r in recent
            ]
        
        cursor.close()
        conn.close()
        return stats
        
    except Exception as e:
        conn.close()
        return {"error": str(e)}

# ===== AGREGAR ESTOS ENDPOINTS DESPUÉS DE LOS EXISTENTES =====

@app.get("/db-status")
async def database_status():
    """Check database connection and table status"""
    try:
        result = check_database_tables()
        return {
            "service": "ARIA XGBoost Predictor",
            "database_status": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "ARIA XGBoost Predictor", 
            "database_status": {"error": str(e), "connected": False},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/db-predictions")
async def prediction_statistics():
    """Get prediction statistics from database"""
    try:
        stats = get_prediction_stats()
        return {
            "service": "ARIA XGBoost Predictor",
            "prediction_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "ARIA XGBoost Predictor",
            "prediction_stats": {"error": str(e)},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/db-info")
async def database_info():
    """Get general database information"""
    try:
        database_url = os.getenv('DATABASE_URL', 'Not configured')
        
        # Mask sensitive parts
        if database_url != 'Not configured' and 'postgres' in database_url:
            masked_url = database_url[:25] + "***" + database_url[-20:]
        else:
            masked_url = database_url
            
        return {
            "service": "ARIA XGBoost Predictor",
            "database_configured": database_url != 'Not configured',
            "database_url_masked": masked_url,
            "database_id": "dpg-d20ht36uk2gs73c9tmbg-a",
            "connection_test": check_database_tables(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "service": "ARIA XGBoost Predictor",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
'''
    
    return code

def main():
    print("🔧 CÓDIGO PARA AGREGAR VERIFICACIÓN DE BASE DE DATOS")
    print("=" * 70)
    
    code = generate_db_code()
    
    print("📝 INSTRUCCIONES:")
    print("=" * 30)
    print("1. Copia el código de abajo")
    print("2. Pégalo en main.py en las secciones indicadas")
    print("3. Agrega 'psycopg2-binary' a requirements.txt")
    print("4. Haz commit y push")
    print("5. Espera el deployment")
    print("6. Prueba los endpoints:")
    print("   • /db-status - Estado general de la DB")
    print("   • /db-predictions - Estadísticas de predicciones")
    print("   • /db-info - Información de configuración")
    
    print("\n" + "="*70)
    print("CÓDIGO PARA AGREGAR:")
    print("="*70)
    print(code)
    
    print("\n🎯 ESTO TE PERMITIRÁ:")
    print("=" * 30)
    print("✅ Ver si la base de datos está conectada")
    print("✅ Verificar qué tablas existen")
    print("✅ Contar registros en cada tabla")
    print("✅ Ver las predicciones más recientes")
    print("✅ Verificar si se están guardando datos")

if __name__ == "__main__":
    main()
