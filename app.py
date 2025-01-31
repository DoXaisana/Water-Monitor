import time
import serial
import psycopg2
from datetime import datetime
from flask import Flask, render_template
import threading

# PostgreSQL Configuration
DB_CONFIG = {
    "dbname": "your_database",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

# Arduino Serial Configuration
SERIAL_PORT = "/dev/ttyUSB0"  # Change for Windows: "COM3"
BAUD_RATE = 9600

# Flask App
app = Flask(__name__)

def setup_database():
    """Create the sensor_data table if it doesn't exist."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id SERIAL PRIMARY KEY,
            value FLOAT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_sensor_data(value):
    """Insert sensor data into PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("INSERT INTO sensor_data (value, timestamp) VALUES (%s, %s)", (value, datetime.now()))
    conn.commit()
    cur.close()
    conn.close()
    print(f"Inserted: {value} at {datetime.now()}")

def read_from_arduino():
    """Read data from Arduino and insert it into the database every 1 hour."""
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print("Connected to Arduino... Waiting for data every 1 hour.")

    while True:
        try:
            data = ser.readline().decode().strip()
            if data:
                value = float(data)
                insert_sensor_data(value)
        except Exception as e:
            print(f"Error reading data: {e}")

        time.sleep(3600)  # Wait 1 hour before reading again

@app.route("/")
def index():
    """Flask Route to display the latest sensor data."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 10")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return render_template("index.html", data=rows)

if __name__ == "__main__":
    setup_database()
    
    # Start serial reading in a separate thread
    thread = threading.Thread(target=read_from_arduino, daemon=True)
    thread.start()

    # Start Flask server
    app.run(debug=True, host="0.0.0.0", port=5000)