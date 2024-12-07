from flask import Blueprint
from conn_ssh import create_conn

# Initialize the blueprint for the bot
database_mgmt_bp = Blueprint('database_mgmt', __name__, url_prefix='/api/db/')

# Update order data in MySQL table based on fetch_recent_orders
def insert_orders(symbol, trend, entry_price, position_size):
    conn, tunnel = None, None
    try:
        # Establish database connection
        conn, tunnel = create_conn()
        insert_query = """
            INSERT INTO orders (symbol, trend, entry_price, position_size, status) 
            VALUES (%s, %s, %s, %s, 'NEW')
        """
        # Use context manager for cursor
        with conn.cursor() as cursor:
            cursor.execute(insert_query, (symbol, trend, entry_price, position_size))
            conn.commit()  # Commit transaction after execution

    except Exception as e:
        print(f"Error processing insert order {symbol} {trend}: {e}")

    finally:
        # Close resources
        if conn:
            conn.close()

# Update order data in MySQL table based on fetch_recent_orders
def update_orders(symbol, trend, closing_price, usdt_profit):
    try:
        conn, tunnel = create_conn()
        with conn.cursor() as cursor:
            # Fetch the order to check if it exists
            cursor.execute(
                "SELECT id FROM orders WHERE symbol = %s AND trend = %s AND status = 'NEW'",
                (symbol, trend)
            )
            order = cursor.fetchone()  # Fetch one matching row

            if order:  # If an order exists
                order_id = order[0]  # Extract the ID from the tuple
                update_query = """
                    UPDATE orders
                    SET closing_price = %s, estimate_pnl = %s, closing_time = now(), status = 'CLOSED'
                    WHERE id = %s 
                """
                cursor.execute(update_query, (closing_price, usdt_profit, order_id))
                conn.commit()  # Commit the transaction

    except Exception as e:
        print(f"Error processing update order {symbol} {trend}: {e}")
    finally:
        # Ensure connection is closed properly
        if conn:
            conn.close()






