import os
import mysql.connector
from binance.client import Client
from flask import Blueprint, jsonify
from dconfig import read_db_config
from conn_ssh import create_conn

# Initialize the blueprint for the bot
get_orders_bp = Blueprint('get_orders', __name__, url_prefix='/api/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client
client = Client(API_KEY, API_SECRET)

# Get the latest open order for a specific symbol (e.g., SOLUSDT)
def get_latest_order(symbol):
    """Fetch the latest open order for the symbol"""
    try:
        orders = client.get_open_orders(symbol=symbol)
        if orders:
            latest_order = orders[-1]  # Get the last open order (most recent)
            return latest_order
        else:
            return None
    except Exception as e:
        print(f"Error fetching order: {e}")
        return None

# Insert order into the MySQL database
def insert_order_to_db(order_data):
    """Insert the latest order into the database"""

    conn, tunnel = create_conn()

    try:
        cursor = conn.cursor()

        # SQL query to insert the order data
        query = """
            INSERT INTO orders (order_id, symbol, side, price, quantity, status, time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Prepare order data to insert
        order_values = (
            order_data['orderId'],
            order_data['symbol'],
            order_data['side'],
            order_data['price'],
            order_data['origQty'],
            order_data['status'],
            order_data['time']
        )
        
        cursor.execute(query, order_values)
        conn.commit()
        print("Order inserted successfully into the database.")
    
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Fetch the latest order for the given symbol and insert it into the database
@get_orders_bp.route('/get_orders', methods=['GET'])
def process_latest_order():
    symbol = 'SOLUSDT'  # Corrected the symbol format for Binance
    """Fetch and insert the latest order into the database"""
    latest_order = get_latest_order(symbol)
    
    if latest_order:
        print(f"Latest Order for {symbol}: {latest_order}")
        insert_order_to_db(latest_order)
        return jsonify(latest_order), 200
    else:
        return jsonify({'message': f"No open orders found for {symbol}."}), 404
