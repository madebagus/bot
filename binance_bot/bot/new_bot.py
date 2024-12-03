import os
from symtable import Symbol
from binance.client import Client
import certifi
from matplotlib.patheffects import PathEffectRenderer
import ta
import pandas as pd
import pandas_ta as pd_ta
from flask import Blueprint, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dconfig import read_db_config
from conn_ssh import create_conn
from decimal import Decimal, ROUND_DOWN
from routers.wallet import get_wallet_balance, calculate_dynamic_safe_trade_amount
from decimal import Decimal


# Initialize the blueprint for the bot
bot_bp = Blueprint('run_bot_atr_new', __name__, url_prefix='/api/bot/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

# Fetch historical data from Binance

import pandas as pd

def get_historical_data(symbol, interval='15m', limit=1500):
    """Fetch historical candlestick data from Binance API."""
    try:
        # Fetch the historical data from Binance API
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Convert the data into a DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert relevant columns to numeric types
        df['high'] = pd.to_numeric(df['high'])
        df['low'] = pd.to_numeric(df['low'])
        df['close'] = pd.to_numeric(df['close'])
        df['open'] = pd.to_numeric(df['open'])
        df['volume'] = pd.to_numeric(df['volume'])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Set the timestamp as the index for easier analysis
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


# Define the RSI calculation logic
@bot_bp.route('/rsi',methods=['GET'])

def rsi_decision(df,price_column='close'):
    """Fetch historical data, calculate RSI, and check trading conditions."""

    # Check for NaN values in the 'close' column
    if df[price_column].isnull().sum() > 0:
        return 'HOLD'
    
    # Ensure enough data for RSI calculation
    if len(df) < 14:
        return 'HOLD'
    
    # Calculate RSI using pandas-ta
    df['RSI'] = pd_ta.rsi(df[price_column], length=14)
    #print('Data Frame df[RSI] :', df)

    # Handle NaN values in RSI column by replacing them with the mean of surrounding values
    df.dropna(subset=['RSI'], inplace=True)

    # Check if RSI contains NaN values
    if df['RSI'].isnull().sum() > 0:
        return 'HOLD'

    # Get the latest RSI values
    latest_rsi = df['RSI'].iloc[-1]
    previous_rsi = df['RSI'].iloc[-2]
    
    #print(f"Symbol: {symbol}, Latest RSI: {latest_rsi}, Previous RSI: {previous_rsi}")

    # Focus 1: Overbought and Oversold Conditions
    if latest_rsi >= 70:
        return "SELL"  # Overbought condition
    elif latest_rsi <= 30:
        return "BUY"  # Oversold condition

    # Focus 2: Normal sell or buy
    elif 50 <= latest_rsi < 70 and previous_rsi > 70:
        return "SELL"  # RSI crossed below 70 (strong sell zone)
    elif 30 <= latest_rsi < 50 and previous_rsi < 30:
        return "BUY"  # RSI crossed above 30 (strong buy zone)

    # Default to HOLD
    else:
        return "HOLD"

def stoch_rsi_decision(df, stoch_rsi_period=14, stoch_rsi_k_period=3, stoch_rsi_d_period=3):
    """Analyze Stochastic RSI for overbought/oversold, momentum, and midline crossovers."""

    # Calculate Stochastic RSI using pandas-ta (stochastic RSI uses %K and %D)
    stoch_rsi = pd_ta.stochrsi(df['close'], length=stoch_rsi_period, rsi_length=stoch_rsi_k_period, stoch_length=stoch_rsi_d_period)

    # Use the correct column from the Stochastic RSI DataFrame
    df['stoch_rsi_k'] = stoch_rsi['STOCHRSIk_14_3_3_3']  # Adjust the column name based on what you find
    df['stoch_rsi_d'] = stoch_rsi['STOCHRSId_14_3_3_3'] 

    # Drop rows with NaN values in 'stoch_rsi' column
    df = df.dropna(subset=['stoch_rsi_k','stoch_rsi_d'])

    # Ensure enough data points
    if len(df) < stoch_rsi_period:
        return "HOLD"

    # Get the latest StochRSI values
    latest_stoch_rsi_k = df['stoch_rsi_k'].iloc[-1]
    latest_stoch_rsi_d = df['stoch_rsi_d'].iloc[-1]
    previous_stoch_rsi_k = df['stoch_rsi_k'].iloc[-2]
    previous_stoch_rsi_d = df['stoch_rsi_d'].iloc[-2]

    # BULLISH when K > D, BEARISH when K < D

    # **1. Overbought and Oversold Conditions**
    if latest_stoch_rsi_k > 80 and latest_stoch_rsi_k < latest_stoch_rsi_d:
        return "SELL"  # Overbought condition
    elif latest_stoch_rsi_k < 20 and latest_stoch_rsi_k > latest_stoch_rsi_d:
        return "BUY"  # Oversold condition
    # **1. Overbought and Oversold Conditions**
    
    # **2. Normal buy or sell**
    elif 50 <= latest_stoch_rsi_k < 80 and previous_stoch_rsi_k > 80 and latest_stoch_rsi_k < latest_stoch_rsi_d: 
            return "SELL"  # Both K and D are crossing down from overbought region
    elif 20 < latest_stoch_rsi_k <= 50 and previous_stoch_rsi_k < 20 and latest_stoch_rsi_k > latest_stoch_rsi_d:
            return "BUY"  # Both K and D are crossing up from oversold region
    elif 80 <= latest_stoch_rsi_k:
            return "SELL"  # Both K and D are crossing up from oversold region
    elif 20 >= latest_stoch_rsi_k:
            return "BUY"  # Both K and D are crossing up from oversold region
    # Default to HOLD if no strong signal
    else:
        return "HOLD"


def detect_triangle_pattern(df):
    # Use the last N candles to analyze the pattern
    n = 10  # Number of candles to consider for the pattern
    highs = df['high'].iloc[-n:]
    lows = df['low'].iloc[-n:]

    # Calculate the trendlines
    upper_trendline = highs.expanding().max()  # Simulates an upper boundary
    lower_trendline = lows.expanding().min()  # Simulates a lower boundary

    # Get the most recent upper and lower bounds
    upper_bound = upper_trendline.iloc[-1]
    lower_bound = lower_trendline.iloc[-1]

    return upper_bound, lower_bound

def calculate_ema(df, short_period=7, long_period=14):
    """Calculate two EMAs with different periods."""
    df = df.copy()  # Create an explicit copy if you're slicing.
    df['EMA_short'] = pd_ta.ema(df['close'], length=short_period)
    df['EMA_long'] = pd_ta.ema(df['close'], length=long_period)
    return df

def detect_ema_crossover(latest_ema_short, latest_ema_long, previous_ema_short, previous_ema_long):
    if previous_ema_short < previous_ema_long and latest_ema_short > latest_ema_long:
        return "BUY"
    elif previous_ema_short > previous_ema_long and latest_ema_short < latest_ema_long:
        return "SELL"
    
    return "HOLD"

def get_ema_decision(df, short_ema_period=7, long_ema_period=14):
    
    # Calculate EMA
    df = calculate_ema(df, short_ema_period, long_ema_period)

    # Drop rows where any NaN or None values are in the EMA columns
    df = df.dropna(subset=['EMA_short', 'EMA_long'])

    # Ensure there is enough data (at least 2 rows) after dropping NaN values
    if len(df) < 2:
        print("Error: Not enough data after dropping NaN values")
        return "HOLD"
    
    try:
        latest_ema_short = df['EMA_short'].iloc[-1]
        latest_ema_long = df['EMA_long'].iloc[-1]
        previous_ema_short = df['EMA_short'].iloc[-2]
        previous_ema_long = df['EMA_long'].iloc[-2]

    except IndexError:
        print(f"Error accessing EMA data for {symbol}. Skipping.")
        return 'HOLD'

    # EMA Crossover Detection
    crossover_decision = detect_ema_crossover(latest_ema_short, latest_ema_long, previous_ema_short, previous_ema_long)
    if crossover_decision == 'BUY':
        return 'BUY'
    elif crossover_decision == 'SELL':
        return 'SELL'
    
    # Default action
    return "HOLD"



#++++++++++++analysi candle trend

def analyze_candle_trend(df, tolerance_percentage=0.05):
    
    df = df.copy()  # Create an explicit copy if you're slicing.
    
    # Ensure sufficient data
    if len(df) < 10:
        print("Insufficient data for candle trend analysis.")
        return 'HOLD'
    
    df['EMA_short'] = pd_ta.ema(df['close'], length=7)
    df['EMA_long'] = pd_ta.ema(df['close'], length=14)

    latest_ema_short = df['EMA_short'].iloc[-1]
    latest_ema_long = df['EMA_long'].iloc[-1]
    previous_ema_short = df['EMA_short'].iloc[-2]
    previous_ema_long = df['EMA_long'].iloc[-2]

    # EMA Crossover Detection
    crossover_decision = detect_ema_crossover(latest_ema_short, latest_ema_long, previous_ema_short, previous_ema_long)
    
    # Get the last and previous highs and lows
    last_high = df['high'].iloc[-1]
    last_low = df['low'].iloc[-1]
    
    prev_high_1 = df['high'].iloc[-2]
    prev_high_2 = df['high'].iloc[-3]
    
    prev_low_1 = df['low'].iloc[-2]
    prev_low_2 = df['low'].iloc[-3]

    # Detect triangle pattern
    upper_bound, lower_bound = detect_triangle_pattern(df)

    # Calculate triangle range and tolerance
    if upper_bound and lower_bound:
        triangle_range = upper_bound - lower_bound
        tolerance = tolerance_percentage * triangle_range
    else:
        tolerance = 0  # No triangle detected, no tolerance applied

    # Initialize trend
    trend = 'HOLD'

    # Identify higher high and potential BUY signal
    if last_high > prev_high_1 and last_high > prev_high_2:
        if upper_bound and last_high > upper_bound + tolerance:
            print("Triangle breakout above detected!")
            trend = 'BUY'
        elif crossover_decision == 'BUY': #if bullish crossover detect then BUY
            trend = 'BUY'
        elif crossover_decision == 'SELL': #if bullish crossover detect then BUY
            trend = 'SELL'
    # Identify lower low and potential SELL signal
    elif last_low < prev_low_1 and last_low < prev_low_2:
        if lower_bound and last_low < lower_bound - tolerance:
            print("Triangle breakout below detected!")
            trend = 'SELL'
        elif crossover_decision == 'SELL': #if Bearish crossover detect the SELL
            trend = 'SELL'
        elif crossover_decision == 'BUY': #if Bearish crossover detect the SELL
            trend = 'BUY'


    return trend



def suggest_trade(symbol, fixed_stop_loss=0.003, fixed_take_profit=0.006):
    """
    """
    # Fetch historical data
    df = get_historical_data(symbol, interval='15m', limit=1500)
    entry_price = df['close'].iloc[-1]  # Entry price is the latest close price

    try:
        # Indicator Calculations
        rsi_dec = rsi_decision(df)
        stoc_rsi_dec = stoch_rsi_decision(df)
        candle_dec = analyze_candle_trend(df)
        ema_dec = get_ema_decision(df)
        
        # Assign Scores
        def assign_score(decision):
            scores = {'STRONG BUY': 1.5, 'BUY': 1, 'HOLD': 0, 'SELL': -1, 'STRONG SELL': -1.5}
            return scores.get(decision, 0)

        rsi_score = assign_score(rsi_dec)
        stoc_rsi_score = assign_score(stoc_rsi_dec)
        ema_score = assign_score(ema_dec)
        candle_score = assign_score(candle_dec)

        # Define weights
        rsi_weight = 0.2
        stoc_rsi_weight = 0.25
        ema_weight = 0.25
        candle_weight = 0.15

        # Total score calculation
        total_score = (rsi_score * rsi_weight) + (ema_score * ema_weight) + (stoc_rsi_score * stoc_rsi_weight) + (candle_score * candle_weight)

        # Determine Trend
        if total_score >= 0.3:
            trend = 'BUY'
        elif total_score <= -0.3:
            trend = 'SELL'
        else:
            trend = 'HOLD'

        #calculate stop loss and take profit
        if trend == 'BUY':
            stop_loss = entry_price - (entry_price * fixed_stop_loss)
            take_profit = entry_price + (entry_price * fixed_take_profit)
        elif trend == 'SELL':
            stop_loss = entry_price + (entry_price * fixed_stop_loss)
            take_profit = entry_price - (entry_price *fixed_take_profit)
        else:
            stop_loss = take_profit = entry_price  # No action

        # Output result
        print(f"Symbol: {symbol}, RSI: {rsi_dec}, Stoch RSI: {stoc_rsi_dec}, EMA: {ema_dec}, Candle: {candle_dec}, Total Score: {total_score}, Final Trend: {trend}")

        return {
            'trend': trend,
            'suggest_entry_price': entry_price,
            'suggest_stop_loss': stop_loss,
            'suggest_take_profit': take_profit,
        }

    except Exception as e:
        print(f"Error calculating trade suggestion for {symbol}: {e}")
        return {
            'trend': 'HOLD',
            'suggest_entry_price': entry_price,
            'suggest_stop_loss': entry_price,
            'suggest_take_profit': entry_price,
        }

    
# Function to insert an order into the database
from binance.enums import SIDE_BUY, SIDE_SELL

def get_symbol_precision(symbol):
    """Fetch the quantity and price precision for a given symbol."""
    exchange_info = client.futures_exchange_info()
    symbol_info = next((item for item in exchange_info['symbols'] if item['symbol'] == symbol), None)
    if symbol_info:
        # Quantity precision is the number of decimals allowed for quantity (stepSize)
        quantity_precision = len(symbol_info['filters'][1]['stepSize'].split('.')[1])
        # Price precision is the number of decimals allowed for price (tickSize)
        price_precision = len(symbol_info['filters'][0]['tickSize'].split('.')[1])
        return price_precision, quantity_precision
    else:
        raise ValueError(f"Symbol {symbol} not found in exchange information.")

def format_decimal(value, precision):
    # Format the value to match the specified precision
    return Decimal(str(value)).quantize(Decimal(f'1e-{precision}'), rounding=ROUND_DOWN)


def check_existing_orders(symbol, trend):
    """Check if there are existing orders for the given symbol and trend."""
    orders = client.futures_get_open_orders(symbol=symbol)
    for order in orders:
        if order['positionSide'] == trend:
            return True
    return False

def place_futures_order(symbol, trend, entry_price, stop_loss, take_profit, leverage, quantity, usdt_to_trade, position_size):
    try:
        # Ensure proper data types for the parameters
        entry_price = Decimal(entry_price) if not isinstance(entry_price, Decimal) else entry_price
        stop_loss = Decimal(stop_loss) if not isinstance(stop_loss, Decimal) else stop_loss
        take_profit = Decimal(take_profit) if not isinstance(take_profit, Decimal) else take_profit
        usdt_to_trade = Decimal(usdt_to_trade) if not isinstance(usdt_to_trade, Decimal) else usdt_to_trade
        position_size = Decimal(position_size) if not isinstance(position_size, Decimal) else position_size
        leverage = Decimal(leverage) if not isinstance(leverage, Decimal) else leverage
        quantity = Decimal(quantity) if not isinstance(quantity, Decimal) else quantity

        # Check for existing open positions for the symbol
        open_positions = client.futures_position_information(symbol=symbol)
        for position in open_positions:
            if position['positionSide'] == ('LONG' if trend == 'BUY' else 'SHORT') and float(position['positionAmt']) != 0:
                print(f"Existing position detected for {symbol} in {trend} direction. Skipping new order.")
                return  # Exit if there's already an open position in the same direction

        # Get symbol precision
        price_precision, qty_precision = get_symbol_precision(symbol)

        # Round prices and quantity to the symbol's precision
        entry_price = format_decimal(entry_price, price_precision)
        stop_loss = format_decimal(stop_loss, price_precision)
        take_profit = format_decimal(take_profit, price_precision)
        quantity = format_decimal(quantity, qty_precision)

        # Set leverage for the symbol
        client.futures_change_leverage(symbol=symbol, leverage=int(leverage))

        # Calculate the trade quantity based on the USDT amount to trade
        notional_value = entry_price * quantity
        if notional_value < 5:
            quantity = Decimal(5 / entry_price).quantize(Decimal(f'0.{"0" * qty_precision}'))  # Adjust quantity if too small

        # Determine the position side based on the trend
        position_side = 'LONG' if trend == 'BUY' else 'SHORT'

        # Place the market order for entry
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if trend == 'BUY' else SIDE_SELL,
            type='MARKET',  # Use MARKET instead of LIMIT
            quantity=float(quantity),
            positionSide=position_side  # Specify the position side (LONG or SHORT)
        )
        print(f"Placed market order: {order}")


        # Fetch open orders to check if stop loss and take profit orders already exist
        try:
            orders = client.futures_get_open_orders(symbol=symbol)
            print(f"Open orders for {symbol}: {orders}")  # Log the raw orders data

            # Check for existing stop loss orders
            existing_stop_loss_orders = [order for order in orders if order.get('type') == 'STOP_MARKET' and order.get('positionSide') == position_side]
            print(f"Existing Stop Loss Orders: {existing_stop_loss_orders}")  # Log existing stop loss orders

            # Check for existing take profit orders
            existing_take_profit_orders = [order for order in orders if order.get('type') == 'TAKE_PROFIT_MARKET' and order.get('positionSide') == position_side]
            print(f"Existing Take Profit Orders: {existing_take_profit_orders}")  # Log existing take profit orders

        except Exception as e:
            print(f"Error fetching open orders for {symbol}: {e}")


        # Initialize Stop Loss order
        if not existing_stop_loss_orders:
            stop_loss_order = client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if trend == 'BUY' else SIDE_BUY,
                type='STOP_MARKET',
                stopPrice=float(stop_loss),  # Price to trigger stop loss order
                quantity=float(quantity),
                positionSide=position_side,
                timeInForce='GTE_GTC',
                closePosition=True,
                workingType='MARK_PRICE'  # Use mark price for the stop trigger
            )
            print(f"Placed Stop Loss order: {stop_loss_order}")
        else:
            print(f"Existing stop loss found for {symbol}. Skipping new stop_loss_order.")


        # Initialize Take Profit order
        if not existing_take_profit_orders:
            take_profit_order = client.futures_create_order(
                symbol=symbol,
                side=SIDE_SELL if trend == 'BUY' else SIDE_BUY,
                type='TAKE_PROFIT_MARKET',
                stopPrice=float(take_profit),  # Price to trigger take profit order
                quantity=float(quantity),
                positionSide=position_side,
                timeInForce='GTE_GTC',
                closePosition=True,
                workingType='MARK_PRICE'  # Use mark price for the take profit trigger
            )
            print(f"Placed Take Profit order: {take_profit_order}")
        else:
            print(f"Existing take profit order found for {symbol}. Skipping new take_profit_order.")

    except Exception as e:
        print(f"Error placing order for {symbol} : {e}")



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
    start_time = end_time - timedelta(hours=4)

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
        check_query = "SELECT * FROM binance_futures_orders WHERE order_id = %s"
        cursor.execute(check_query, (order_id,))
        existing_order = cursor.fetchone()

        # Insert or update the order details
        if existing_order is None:
            # Insert new order if it doesn't exist
            insert_query = """
                INSERT INTO binance_futures_orders (
                    order_id, client_order_id, symbol, status, orig_qty, executed_qty,
                    avg_fill_price, price, side, type, time_in_force, stop_price, time, update_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                order_id, order['clientOrderId'], order['symbol'], order['status'], 
                order['origQty'], order['executedQty'], order['avgPrice'], order['price'], 
                order['side'], order['type'], order['timeInForce'], order['stopPrice'], 
                order['time'], order['updateTime']
            ))
        else:
            # Update the order if there is a change in the status or other fields
            update_query = """
                UPDATE binance_futures_orders SET
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


import threading

# Initialize the list of coin pairs with a default value
# ****** Symbol criteria to meet BOT : 1 < price < 10 
coin_pairs = ['ATOMUSDT','DOTUSDT']  # Example with 4 pairs
# Flag to control whether the fetch_recent_orders task should be canceled
cancel_fetch_orders = False
# Function to handle trading for each symbol
def run_symbol_task(symbol):
    print(f"+ + + Running bot for {symbol} + + +")
    
    usdt_to_trade = Decimal('10')  # Example trade amount
    trade_suggestion = suggest_trade(symbol)

    # Handle trade conditions
    if trade_suggestion['trend'] == 'BUY':
        if trade_suggestion['suggest_take_profit'] <= trade_suggestion['suggest_entry_price'] or trade_suggestion['suggest_stop_loss'] >= trade_suggestion['suggest_entry_price']:
            trade_signal = 'HOLD'
        else:
            trade_signal = trade_suggestion['trend']
    else:
        if trade_suggestion['suggest_take_profit'] >= trade_suggestion['suggest_entry_price'] or trade_suggestion['suggest_stop_loss'] <= trade_suggestion['suggest_entry_price']:
            trade_signal = 'HOLD'
        else:
            trade_signal = trade_suggestion['trend']
    

    if trade_signal != 'HOLD':
        leverage = Decimal('5')  # Fixed leverage
        entry_price = Decimal(str(trade_suggestion['suggest_entry_price']))
        effective_position_size = usdt_to_trade * leverage
        quantity = effective_position_size / entry_price

        # Debugging
        print (f"Symbol : ",symbol)
        print(f"Trend: {trade_signal}")
        print(f"Entry Price: {entry_price}")
        print(f"Stop Loss: {Decimal(str(trade_suggestion['suggest_stop_loss']))}")
        print(f"Take profit: {Decimal(str(trade_suggestion['suggest_take_profit']))}")
        print(f"Leverage: {leverage}")
        print(f"Quantity: {quantity}")
        print(f"Trade Size: ",usdt_to_trade)
        print(f"Effective Position Size: {effective_position_size}")
        

        # Place order
        place_futures_order(symbol, trade_signal, entry_price,
                            Decimal(str(trade_suggestion['suggest_stop_loss'])),
                            Decimal(str(trade_suggestion['suggest_take_profit'])),
                            leverage, quantity, usdt_to_trade, effective_position_size)
    else:
        print(f"+ + + Trade suggestion: {trade_suggestion['trend']} + + +")

# Main trading bot function that runs trading tasks for each symbol concurrently
def run_trading_bot_task():
    global cancel_fetch_orders

    # Signal to cancel fetch_recent_orders
    cancel_fetch_orders = True
    print("Starting run_trading_bot_task. Cancelling fetch_recent_orders if running.")
    
    threads = []
    
    # Create and start a thread for each symbol to run trading tasks concurrently
    for symbol in coin_pairs:
        thread = threading.Thread(target=run_symbol_task, args=(symbol,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    # Reset the cancel flag after bot execution
    cancel_fetch_orders = False
    print("All tasks completed.")

# Initialize scheduler
scheduler = BackgroundScheduler()
# Flag to track bot status
bot_running = False

# Function to start the trading bot
def start_trading_bot():
    global bot_running
    if not bot_running:
        # Start the scheduler and add jobs only when the bot is started
        scheduler.start()
        # Add the trading bot task to run periodically
        scheduler.add_job(run_trading_bot_task, IntervalTrigger(seconds=5), max_instances=1)  # Adjust the interval as needed
        scheduler.add_job(fetch_recent_orders, IntervalTrigger(seconds=70), max_instances=1)  # Track orders every 30 seconds
        bot_running = True
        print("Bot started.")

# Function to stop the trading bot
def stop_trading_bot():
    global bot_running
    if bot_running:
        # Stop the scheduler and all jobs
        scheduler.shutdown()
        bot_running = False
        print("Bot stopped.")

# API routes
@bot_bp.route('/start_bot')
def start_bot():
    """Start the trading bot."""
    start_trading_bot()
    return jsonify({"message": "BOT Started!"})

@bot_bp.route('/stop_bot')
def stop_bot():
    """Stop the trading bot."""
    stop_trading_bot()
    return jsonify({"message": "BOT Stopped!"})
