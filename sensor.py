import time
import serial
import psycopg2
from datetime import datetime

# Arduino Serial Configuration
SERIAL_PORT = "/dev/ttyUSB0"  # Change for Windows: "COM3"
BAUD_RATE = 9600

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
    """Read data from Arduino and insert it into the database every 10 seconds."""
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print("Connected to Arduino... Reading sensor data.")

    while True:
        try:
            data = ser.readline().decode().strip()
            if data:
                value = float(data)
                insert_sensor_data(value)
        except Exception as e:
            print(f"Error reading data: {e}")

        time.sleep(10)  # Read every 10 seconds