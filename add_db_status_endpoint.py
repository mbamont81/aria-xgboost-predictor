#!/usr/bin/env python3
"""
Script para agregar un endpoint de verificación de base de datos al servicio Render
"""

def generate_db_status_code():
    """Generate code to add to main.py for database status checking"""
    
    db_status_code = '''
# Agregar estas importaciones al inicio del archivo
import os
import psycopg2
from psycopg2 import OperationalError

# Agregar esta función después de las otras funciones
def check_database_connection():
    """Check PostgreSQL database connection and return status"""
    try:
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return {
                "connected": False,
                "error": "DATABASE_URL not found in environment variables",
                "database_configured": False
            }
        
        # Try to connect
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        # Test query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        
        # Check if our tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        # Count records if prediction table exists
        prediction_count = 0
        if 'predictions' in tables:
            cursor.execute("SELECT COUNT(*) FROM predictions")
            prediction_count = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()
        
        return {
            "connected": True,
            "database_configured": True,
            "postgres_version": version,
            "tables": tables,
            "prediction_records": prediction_count,
            "database_url_configured": True
        }
        
    except OperationalError as e:
        return {
            "connected": False,
            "database_configured": True,
            "error": f"Connection failed: {str(e)}",
            "database_url_configured": bool(os.getenv('DATABASE_URL'))
        }
    except Exception as e:
        return {
            "connected": False,
            "database_configured": False,
            "error": f"Unexpected error: {str(e)}",
            "database_url_configured": bool(os.getenv('DATABASE_URL'))
        }

# Agregar este endpoint después de los otros endpoints
@app.get("/db-status")
async def database_status():
    """Check database connection status"""
    status = check_database_connection()
    return {
        "database_status": status,
        "service": "ARIA XGBoost Predictor",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/db-info")
async def database_info():
    """Get detailed database information"""
    try:
        database_url = os.getenv('DATABASE_URL', 'Not configured')
        
        # Mask sensitive parts of URL
        if database_url != 'Not configured' and 'postgres' in database_url:
            masked_url = database_url[:20] + "***" + database_url[-15:]
        else:
            masked_url = database_url
            
        return {
            "database_url_configured": database_url != 'Not configured',
            "database_url_masked": masked_url,
            "database_id": "dpg-d20ht36uk2gs73c9tmbg-a",
            "expected_tables": ["predictions", "trades", "performance"],
            "connection_test": check_database_connection()
        }
    except Exception as e:
        return {
            "error": str(e),
            "database_configured": False
        }
'''
    
    return db_status_code

def main():
    print("🔧 CÓDIGO PARA AGREGAR VERIFICACIÓN DE BASE DE DATOS")
    print("=" * 60)
    
    code = generate_db_status_code()
    
    print("📝 Agrega este código a main.py:")
    print("=" * 40)
    print(code)
    
    print("\n📋 PASOS PARA IMPLEMENTAR:")
    print("=" * 40)
    print("1. Copia el código de arriba")
    print("2. Pégalo en main.py en las secciones indicadas")
    print("3. Agrega psycopg2 a requirements.txt si no está")
    print("4. Haz commit y push a GitHub")
    print("5. Espera el deployment en Render")
    print("6. Prueba los nuevos endpoints:")
    print("   - https://aria-xgboost-predictor.onrender.com/db-status")
    print("   - https://aria-xgboost-predictor.onrender.com/db-info")
    
    print("\n🎯 ESTO TE DIRÁ:")
    print("=" * 40)
    print("✅ Si hay conexión a PostgreSQL")
    print("✅ Qué tablas existen")
    print("✅ Cuántos registros hay")
    print("✅ Si DATABASE_URL está configurada")
    print("✅ Errores de conexión específicos")

if __name__ == "__main__":
    main()
