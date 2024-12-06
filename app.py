from threading import Thread
import time
import requests
from flask import Flask
from binance_bot.bot.new_bot_new_no_sltp import run_trading_bot_task
from binance_bot.reversal.reversal_monitor_new_no_sltp import monitor_positions
from binance_bot.data.database_management import fetch_and_upsert_order_history
from binance_bot.messaging.chat_bot import send_status_message

# Initialize the Flask application
app = Flask(__name__)

# Replace with your Telegram Bot Token and Chat ID
BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"



# Function to periodically send the status message
def start_status_scheduler():
    print("Starting Status Scheduler...")
    while True:
        send_status_message()
        time.sleep(1800)  # Send every 30 minutes (1800 seconds)

# Function to run the reversal monitor in a separate thread
def start_profit_monitor():
    print("Starting Profit Monitoring...")
    while True:
        try:
            monitor_positions()
        except Exception as e:
            print(f"Reversal monitor error: {e}")
        time.sleep(1)  # Adjust frequency of checks as needed

# Function to run the BOT in a separate thread
def start_bot(): 
    print(">> Starting Bot Engine...")
    while True:
        try:
            run_trading_bot_task()
        except Exception as e:
            print(f"Starting BOT error: {e}")
        time.sleep(5)  # Adjust frequency of checks as needed

# Run the app and the background tasks
if __name__ == '__main__':
    
    # Start the background threads
    monitor_thread = Thread(target=start_profit_monitor)
    monitor_thread.daemon = True  # Daemon thread will exit when the main program exits
    monitor_thread.start()

    bot_thread = Thread(target=start_bot)
    bot_thread.daemon = True  # Daemon thread will exit when the main program exits
    bot_thread.start()

    status_thread = Thread(target=start_status_scheduler)
    status_thread.daemon = True  # Daemon thread will exit when the main program exits
    status_thread.start()

    # Run the Flask app (it will not block background threads)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)  # `use_reloader=False` to prevent running the background tasks multiple times
