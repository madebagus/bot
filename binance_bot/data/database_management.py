from flask import Blueprint
from conn_ssh import create_conn

# Initialize the blueprint for the bot
database_mgmt_bp = Blueprint('database_mgmt', __name__, url_prefix='/api/db/')

# Update order data in MySQL table based on fetch_recent_orders
def insert_orders(symbol,trend,entry_price,position_size):

    conn, tunnel = create_conn()

    try:
        cursor = conn.cursor()

        insert_query = """
                INSERT INTO orders (symbol, trend, entry_price, position_size, created_at) 
                VALUES (%s, %s, %s, %s, now())
            """
        cursor.execute(
            insert_query, (symbol,trend,entry_price,position_size)
        )
        
        conn.commit()  # Commit once after the operation
    
    except conn.Error as e:
        print(f"Insert data {symbol} {trend}: {e}")
    
    except Exception as e:
        print(f"Error processing insert order {symbol} {trend}: {e}")
    
    finally:
        if conn:
            cursor.close()
            conn.close()  # Ensure connection is closed after the operation
        

# Update order data in MySQL table based on fetch_recent_orders
def update_orders(symbol,trend,closing_price):

    conn, tunnel = create_conn()

    try:
        cursor = conn.cursor()

        insert_query = """
                UPDATE orders SET closing_pice = %s, closing_time = now(), status = 'CLOSED'
                WHERE symbol = %s and trend= %s and status = 'NEW')
            """
        cursor.execute(
            insert_query, (closing_price,symbol,trend)
        )
        
        conn.commit()  # Commit once after the operation
    
    except conn.Error as e:
        print(f"Insert data {symbol} {trend}: {e}")
    
    except Exception as e:
        print(f"Error processing insert order {symbol} {trend}: {e}")
    
    finally:
        if conn:
            cursor.close()
            conn.close()  # Ensure connection is closed after the operation






