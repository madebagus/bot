import time
import os
from venv import create
from binance.client import Client
import certifi
from flask import Blueprint, jsonify
from dconfig import read_db_config
from conn_ssh import create_conn
from decimal import Decimal, ROUND_DOWN
from routers.wallet import get_wallet_balance, calculate_dynamic_safe_trade_amount
from datetime import datetime, timedelta


# Initialize the blueprint for the bot
database_mgmt_bp = Blueprint('database_mgmt', __name__, url_prefix='/api/db/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})



#fetching last 4 hours order from binance and insert or update order in table by calling process_order
from datetime import datetime, timedelta
import pytz
import mysql.connector

# Shared cancel flag for task interruption
cancel_fetch_orders = False

def fetch_recent_orders():
    global cancel_fetch_orders

    # Calculate the start time as 4 hours ago from the current time
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(hours=1)

    # Convert to timestamps for the Binance API
    start_timestamp = int(start_time.timestamp() * 1000)  # in milliseconds
    end_timestamp = int(end_time.timestamp() * 1000)      # in milliseconds

    try:
        print("Starting fetch_recent_orders.")

        # Fetch recent futures orders across all symbols within the 4-hour window
        recent_orders = client.futures_get_all_orders(
            startTime=start_timestamp,
            endTime=end_timestamp
        )

        # Process each order
        for order in recent_orders:
            # Check for cancel signal
            if cancel_fetch_orders:
                print("fetch_recent_orders interrupted by run_trading_bot_task.")
                return  # Gracefully exit

            # Insert or update each order in the database
            insert_update_orders(order)

        print("fetch_recent_orders completed.")
    except Exception as e:
        print(f"Error fetching recent orders: {e}")


# Update order data in MySQL table based on fetch_recent_orders
def insert_update_orders(order):

    conn, tunnel = create_conn()

    try:
        cursor = conn.cursor()

        # Ensure correct key 'orderId' from API response
        order_id = order['orderId']

        # Check if the order exists in the database
        check_query = "SELECT * FROM order_list WHERE order_id = %s"
        cursor.execute(check_query, (order_id,))
        existing_order = cursor.fetchone()

        # Insert or update the order details
        if existing_order is None:
            # Insert new order if it doesn't exist
            insert_query = """
                INSERT INTO order_list (
                    order_id, client_order_id, symbol, status, orig_qty, executed_qty,
                    avg_fill_price, price, side, type, time, update_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                order_id, order['clientOrderId'], order['symbol'], order['status'], 
                order['origQty'], order['executedQty'], order['avgPrice'], order['price'], 
                order['side'], order['type'], 
                order['time'], order['updateTime']
            ))
        else:
            # Update the order if there is a change in the status or other fields
            update_query = """
                UPDATE order_list SET
                    status = %s, executed_qty = %s, avg_fill_price = %s, update_time = %s
                WHERE order_id = %s
            """
            cursor.execute(update_query, (
                order['status'], order['executedQty'], order['avgPrice'], order['updateTime'], order_id
            ))
        
        conn.commit()  # Commit once after the operation
    
    except mysql.connector.Error as e:
        print(f"Database error processing order {order['orderId']}: {e}")
    
    except Exception as e:
        print(f"Error processing order {order['orderId']}: {e}")
    
    finally:
        if conn:
            cursor.close()
            conn.close()  # Ensure connection is closed after the operation
        time.sleep(30)


import pymysql
from datetime import datetime, timedelta
from binance.client import Client

def fetch_and_upsert_order_history():
    """
    Fetch historical orders from Binance Futures and upsert them into the database.
    """
    # Connect to the database
    conn, tunnel = create_conn()
    
    try:
        # Calculate the time range (last 15 minutes)
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=15)
        start_timestamp = int(start_time.timestamp() * 1000)  # Convert to milliseconds
        end_timestamp = int(now.timestamp() * 1000)  # Convert to milliseconds

        # Fetch historical orders from Binance Futures
        trades = client.futures_account_trades(startTime=start_timestamp, endTime=end_timestamp)
        if not trades:
            print("No trades found in the last 15 minutes.")
            return
        
        # Prepare the upsert query
        sql = """
        INSERT INTO order_list (symbol, side, created_at, updated_at, status, closing_price,
                                orig_qty, order_id, client_order_id, executed_qty, avg_fill_price, price,
                                type, closing_time)
        VALUES (%(symbol)s, %(side)s, %(created_at)s, %(updated_at)s, %(status)s, %(closing_price)s,
                %(orig_qty)s, %(order_id)s, %(client_order_id)s, %(executed_qty)s, %(avg_fill_price)s, %(price)s,
                %(type)s, %(closing_time)s)
        ON DUPLICATE KEY UPDATE
            symbol = VALUES(symbol),
            side = VALUES(side),
            created_at = VALUES(created_at),
            updated_at = VALUES(updated_at),
            status = VALUES(status),
            closing_price = VALUES(closing_price),
            orig_qty = VALUES(orig_qty),
            client_order_id = VALUES(client_order_id),
            executed_qty = VALUES(executed_qty),
            avg_fill_price = VALUES(avg_fill_price),
            price = VALUES(price),
            type = VALUES(type),
            closing_time = VALUES(closing_time);
        """

        # Iterate over each trade and prepare the data
        trade_data_list = []
        for trade in trades:
            side = "BUY" if trade.get("side", "UNKNOWN").upper() == "BUY" else "SELL"

            trade_data = {
                "symbol": trade["symbol"],
                "side": side,
                "created_at": datetime.utcfromtimestamp(trade["time"] / 1000),
                "updated_at": datetime.utcnow(),
                "status": "FILLED",  # Binance only returns filled trades in history
                "closing_price": float(trade["price"]),
                "orig_qty": float(trade["qty"]),
                "order_id": trade["orderId"],
                "client_order_id": None,  # Placeholder as Binance doesn't include clientOrderId
                "executed_qty": float(trade["qty"]),
                "avg_fill_price": float(trade["price"]),
                "price": float(trade["price"]),
                "type": trade.get("type", None),  # Default to None if 'type' is missing
                "closing_time": None  # Placeholder, populate as needed
            }
            trade_data_list.append(trade_data)

        # Execute batch upserts
        with conn.cursor() as cursor:
            cursor.executemany(sql, trade_data_list)

        # Commit changes to the database
        conn.commit()
        print(f"Upserted {len(trades)} trades into the database.")
    
    except Exception as e:
        print(f"Error while fetching and upserting orders: {e}")
    
    finally:
        if conn:
            conn.close()






