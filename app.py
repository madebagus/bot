from threading import Thread
import time
from flask import Flask
from binance_bot.bot.new_bot_new_no_sltp import run_bot_real_bp
from binance_bot.reversal.reversal_monitor_new_no_sltp import monitor_positions
from binance_bot.data.database_management import fetch_and_upsert_order_history

# Initialize the Flask application
app = Flask(__name__)

# Register the blueprints with the Flask app
app.register_blueprint(run_bot_real_bp)

# Function to run the reversal monitor in a separate thread
def start_profit_monitor():
    print("Starting Profit Monitoring...")
    while True:
        try:
            monitor_positions()
        except Exception as e:
            print(f"Reversal monitor error: {e}")
        time.sleep(3)  # Adjust frequency of checks as needed

# Function to run the fetch orders in a separate thread
#def start_fetch_order(): 
#    print(">> Starting Fetching Order...")
#    while True:
#        try:
#            fetch_and_upsert_order_history()
#        except Exception as e:
#            print(f"Fetching order error: {e}")
#        time.sleep(30)  # Adjust frequency of checks as needed

# Run the app and the background tasks
if __name__ == '__main__':
    
    # Start the background threads
    monitor_thread = Thread(target=start_profit_monitor)
    monitor_thread.daemon = True  # Daemon thread will exit when the main program exits
    monitor_thread.start()

#    fetch_order_thread = Thread(target=start_fetch_order)
#    fetch_order_thread.daemon = True
#    fetch_order_thread.start()

    # Run the Flask app (it will not block background threads)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)  # `use_reloader=False` to prevent running the background tasks multiple times
