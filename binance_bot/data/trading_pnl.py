import time
import os
from binance.client import Client
import certifi
from flask import Blueprint
from dconfig import read_db_config
from conn_ssh import create_conn

# Initialize the blueprint for the bot
database_mgmt_bp = Blueprint('database_mgmt', __name__, url_prefix='/api/db/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

# Function to fetch income history
def fetch_position_history(client, symbol, start_time, end_time):
    try:
        # Fetch position history for the given symbol within the time range
        income_history = client.futures_income_history(symbol=symbol, startTime=start_time, endTime=end_time)
        return income_history
    except Exception as e:
        print(f"Error fetching position history: {e}")
        return []

# Function to process the fetched data and insert it into the MySQL table
def process_and_insert_pnl(income_history):
    conn,tunnel = create_conn()
    for record in income_history:
        symbol = record['symbol']
        income = float(record['income'])
        timestamp = record['time']
        date = time.strftime('%Y-%m-%d', time.gmtime(timestamp / 1000))  # Convert timestamp to date

        cursor = conn.cursor()

        # Check if the record already exists for that symbol and date
        cursor.execute("SELECT total_pnl FROM symbol_daily_pnl WHERE symbol = %s AND date = %s", (symbol, date))
        result = cursor.fetchone()

        if result:
            # Update the existing record with accumulated pnl
            total_pnl = result[0] + income
            cursor.execute("UPDATE symbol_daily_pnl SET total_pnl = %s WHERE symbol = %s AND date = %s", (total_pnl, symbol, date))
        else:
            # Insert a new record for the symbol and date
            cursor.execute("INSERT INTO symbol_daily_pnl (symbol, date, total_pnl) VALUES (%s, %s, %s)", (symbol, date, income))
        
        conn.commit()

# Main function
def analyze_trading_results(client, symbols, start_time, end_time):
    for symbol in symbols:
        income_history = fetch_position_history(client, symbol, start_time, end_time)
        process_and_insert_pnl(income_history)

# Example usage
if __name__ == "__main__":
    # Initialize Binance client
    client = Client(api_key="your_api_key", api_secret="your_api_secret")

    # Set the time range for analysis (e.g., last 24 hours)
    end_time = int(time.time() * 1000)  # Current time in milliseconds
    start_time = end_time - (24 * 60 * 60 * 1000)  # 24 hours ago in milliseconds

    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Add your symbols here

    analyze_trading_results(client, symbols, start_time, end_time)

    print("Trading results have been analyzed and stored.")
