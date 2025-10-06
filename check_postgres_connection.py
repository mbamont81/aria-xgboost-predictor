#!/usr/bin/env python3
"""
Script para verificar si hay conexión a PostgreSQL en Render
y si se están guardando datos de predicciones
"""

import os
import requests
from datetime import datetime

def check_render_postgres_connection():
    """Check if the Render service has PostgreSQL connection"""
    print("🔍 VERIFICANDO CONEXIÓN POSTGRESQL EN RENDER")
    print("=" * 60)
    
    # Check if service has database connection info
    try:
        response = requests.get("https://aria-xgboost-predictor.onrender.com/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Servicio Render respondiendo")
            print(f"📊 Versión: {data.get('version', 'N/A')}")
            print(f"📋 Modelos cargados: {data.get('models_loaded', 'N/A')}")
            
            # Check for database-related endpoints
            endpoints = data.get('available_endpoints', [])
            db_endpoints = [ep for ep in endpoints if 'db' in ep.lower() or 'data' in ep.lower() or 'log' in ep.lower()]
            
            if db_endpoints:
                print(f"🗄️  Endpoints relacionados con DB: {db_endpoints}")
            else:
                print("❌ No se encontraron endpoints de base de datos")
                
        else:
            print(f"❌ Error conectando al servicio: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def check_for_database_endpoints():
    """Check for potential database-related endpoints"""
    print("\n🔍 VERIFICANDO ENDPOINTS DE BASE DE DATOS")
    print("=" * 50)
    
    # Common database endpoints to test
    test_endpoints = [
        "/health",
        "/status", 
        "/db-status",
        "/database",
        "/logs",
        "/predictions",
        "/trades",
        "/data",
        "/stats"
    ]
    
    base_url = "https://aria-xgboost-predictor.onrender.com"
    
    for endpoint in test_endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint} - Disponible")
                try:
                    data = response.json()
                    if isinstance(data, dict):
                        # Look for database-related keys
                        db_keys = [k for k in data.keys() if any(word in k.lower() for word in ['db', 'database', 'postgres', 'connection', 'records', 'count'])]
                        if db_keys:
                            print(f"   🗄️  Claves relacionadas con DB: {db_keys}")
                            for key in db_keys:
                                print(f"      {key}: {data[key]}")
                except:
                    print(f"   📝 Respuesta no JSON")
            elif response.status_code == 404:
                print(f"❌ {endpoint} - No encontrado")
            else:
                print(f"⚠️  {endpoint} - Status {response.status_code}")
                
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")

def check_environment_variables():
    """Check if there are environment variables that might indicate PostgreSQL"""
    print("\n🔍 VARIABLES DE ENTORNO POTENCIALES")
    print("=" * 50)
    
    # Check local environment for any database URLs
    db_vars = ['DATABASE_URL', 'POSTGRES_URL', 'DB_URL', 'RENDER_DATABASE_URL']
    
    found_vars = []
    for var in db_vars:
        value = os.getenv(var)
        if value:
            found_vars.append((var, value))
            
    if found_vars:
        print("✅ Variables de base de datos encontradas:")
        for var, value in found_vars:
            # Mask sensitive parts
            if 'postgres' in value.lower():
                masked = value[:20] + "***" + value[-10:] if len(value) > 30 else "***"
                print(f"   {var}: {masked}")
            else:
                print(f"   {var}: {value}")
    else:
        print("❌ No se encontraron variables de base de datos locales")
        print("   (Esto es normal si la DB está configurada solo en Render)")

def suggest_verification_methods():
    """Suggest ways to verify PostgreSQL connection"""
    print("\n💡 MÉTODOS PARA VERIFICAR LA BASE DE DATOS POSTGRESQL")
    print("=" * 60)
    
    print("1. 🖥️  RENDER DASHBOARD:")
    print("   - Ve a render.com → Tu servicio")
    print("   - Revisa la pestaña 'Environment'")
    print("   - Busca DATABASE_URL o variables similares")
    print("   - Revisa los logs del servicio")
    
    print("\n2. 📊 LOGS DE RENDER:")
    print("   - En el dashboard, ve a 'Logs'")
    print("   - Busca mensajes como:")
    print("     • 'Connected to database'")
    print("     • 'PostgreSQL connection'") 
    print("     • 'Saving prediction to DB'")
    print("     • Errores de conexión a DB")
    
    print("\n3. 🔧 AGREGAR ENDPOINT DE VERIFICACIÓN:")
    print("   - Podemos agregar un endpoint /db-status")
    print("   - Que muestre el estado de la conexión")
    print("   - Y estadísticas de registros guardados")
    
    print("\n4. 📝 REVISAR CÓDIGO FUENTE:")
    print("   - Buscar imports de psycopg2 o sqlalchemy")
    print("   - Verificar si hay código de inserción de datos")
    print("   - Comprobar si se registran las predicciones")

def main():
    print("🔍 VERIFICACIÓN DE BASE DE DATOS POSTGRESQL EN RENDER")
    print("=" * 70)
    print(f"⏰ Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Base de datos objetivo: dpg-d20ht36uk2gs73c9tmbg-a")
    
    # Run checks
    check_render_postgres_connection()
    check_for_database_endpoints()
    check_environment_variables()
    suggest_verification_methods()
    
    print(f"\n✅ Verificación completada")
    print("=" * 70)

if __name__ == "__main__":
    main()
