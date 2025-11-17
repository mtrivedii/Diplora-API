import psycopg2
import sys

DATABASE_URL = "postgresql://postgres.vcdvtrqrqoegtjmtaulm:GQR-bqp.zqg*mcq3pma@aws-1-eu-west-1.pooler.supabase.com:5432/postgres"

print("Attempting to connect via Session Pooler...")

try:
    conn = psycopg2.connect(DATABASE_URL)
    print("✅ Connection successful!")
        
    cursor = conn.cursor()
    
    test_query = "SELECT * FROM ecg_data LIMIT 1;"
    
    print(f"Executing test query: {test_query}")
    
    cursor.execute(test_query)
    
    data = cursor.fetchone()
    
    print("✅ Query successful!")
    print("Test data:", data)
    
    cursor.close()
    conn.close()
    
    
except Exception as e:
    print("\n❌ Operation failed.")
    print(f"Error: {e}")