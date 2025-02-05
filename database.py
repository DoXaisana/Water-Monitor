import psycopg2
import pandas as pd

# PostgreSQL Configuration
DB_CONFIG = {
    "dbname": "your_database",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

def get_historical_data():
    """Fetch the last 30 days of water usage."""
    conn = psycopg2.connect(**DB_CONFIG)
    query = "SELECT date, SUM(usage_percentage) AS daily_usage FROM water_usage GROUP BY date ORDER BY date DESC LIMIT 30"
    df = pd.read_sql(query, conn)
    conn.close()
    
    df = df.sort_values(by="date")  # Ensure ascending order
    return df

def get_latest_sensor_data():
    """Fetch the latest real-time sensor reading."""
    conn = psycopg2.connect(**DB_CONFIG)
    query = "SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1"
    df = pd.read_sql(query, conn)
    conn.close()

    return df.to_dict(orient="records")[0] if not df.empty else {"error": "No data available"}