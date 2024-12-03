import copy
import os
from symtable import Symbol
from tracemalloc import stop
from binance.client import Client
import certifi
import ta
import pandas as pd
from flask import Blueprint, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dconfig import read_db_config
from conn_ssh import create_conn
from decimal import Decimal, ROUND_DOWN
from routers.wallet import get_wallet_balance, calculate_dynamic_safe_trade_amount
from decimal import Decimal
import pandas_ta as pd_ta
from reversal.reversal_monitor_new import get_position_details
# Initialize the blueprint for the bot
run_bot_real_bp = Blueprint('run_bot_atr_new', __name__, url_prefix='/api/bot/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

# ++ DATA FRAME SECTION ++
 
# Fetch historical data from Binance
import time

def get_historical_data(symbol, interval='1m', limit=1500, retries=5, delay=2):
    """
    Fetch historical data for the specified symbol and interval, with retry logic.

    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT').
        interval (str): Timeframe interval (e.g., '1m', '5m').
        limit (int): Number of data points to fetch.
        retries (int): Maximum number of retries.
        delay (int): Delay in seconds between retries.

    Returns:
        pd.DataFrame: DataFrame containing historical data, or None if unsuccessful.
    """
    attempt = 0

    while attempt < retries:
        try:
            #print(f"Attempt {attempt + 1} to fetch data for {symbol}...")
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            # Convert relevant columns to numeric types
            for col in ['high', 'low', 'close', 'open', 'volume']:
                df[col] = pd.to_numeric(df[col])

            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Check for minimum data requirement
            if df.empty or len(df) < 20:
                raise ValueError("Insufficient data to calculate indicators.")
            
            # Drop rows with NaN values
            df = df.dropna()

            #print(f"Data fetched successfully for {symbol} on attempt {attempt + 1}")
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to fetch data for {symbol} after {retries} attempts.")
                return None


# ++ TREND ANALYSIS SECTION ++

def rsi_decision(symbol, price_column='close', overbought=80, oversold=20, breakout_upper=60, breakout_lower=40):
    """Fetch historical data, calculate RSI, and check trading conditions."""
    df = get_historical_data(symbol)
    if df is None or df.empty:
        print(f"Insufficient RSI data for {symbol}")
        return 'HOLD'

    # Drop rows with NaN values
    df = df.dropna()

    # Check for enough data and valid close prices
    if len(df) < 4 or df[price_column].isnull().sum() > 0:
        print(f"RSI data insufficient or contains null values for {symbol}")
        return 'HOLD'

    # Calculate RSI
    df['RSI'] = pd_ta.rsi(df[price_column], length=4)
    df['RSI'].fillna(50, inplace=True)  # Neutral value for NaN RSI

    # Check if RSI calculation failed
    if df['RSI'].isnull().sum() > 0:
        print(f"RSI calculation failed for {symbol}")
        return 'HOLD'

    # Get the latest and previous RSI values
    latest_rsi = df['RSI'].iloc[-1]
    previous_rsi = df['RSI'].iloc[-2]

    # Overbought and Oversold Conditions
    if latest_rsi < overbought and previous_rsi > overbought:  # Overbought reversal
        return "SELL"
    elif latest_rsi > oversold and previous_rsi < oversold:  # Oversold reversal
        return "BUY"
    
    # Default to HOLD
    return "HOLD"


# Combine rsi and bollinger 

def combine_bollinger_and_rsi(
    symbol, 
    price_column='close', 
    window=20, 
    num_std_dev=2, 
    buy_threshold=0.2, 
    sell_threshold=0.8,
    max_loss_usdt=0.2,  # Maximum loss in USDT
    leverage=5  # Leverage factor (default is 5x leverage)
):
    """
    Combine Bollinger Bands and RSI to calculate trade trends, stop loss, and take profit
    with a guaranteed 1:2 risk-reward ratio, considering leverage.
    """
    # Fetch historical data
    df = get_historical_data(symbol)
    if df is None or df.empty or len(df) < window:
        print(f"Insufficient data for {symbol}")
        return {
            'trend': 'HOLD',
            'suggest_entry_price': 0,
            'suggest_stop_loss': 0,
            'suggest_take_profit': 0,
        }

    # Drop rows with NaN values
    df = df.dropna()

    # Calculate Bollinger Bands
    sma = df[price_column].rolling(window=window).mean()
    std = df[price_column].rolling(window=window).std()
    upper_band = sma + (std * num_std_dev)
    lower_band = sma - (std * num_std_dev)

    # Ensure Bollinger Band range is valid
    latest_upper_band = upper_band.iloc[-1]
    latest_lower_band = lower_band.iloc[-1]
    band_range = latest_upper_band - latest_lower_band

    if band_range <= 0:
        print(f"Invalid Bollinger Band range for {symbol}")
        return {
            'trend': 'HOLD',
            'suggest_entry_price': 0,
            'suggest_stop_loss': 0,
            'suggest_take_profit': 0,
        }

    # Check avoid conditions
    if avoid_conditions(symbol, df):
        print(f"Decision is HOLD for {symbol}")
        return {
            'trend': 'HOLD',
            'suggest_entry_price': 0,
            'suggest_stop_loss': 0,
            'suggest_take_profit': 0,
        }

    # Check RSI decision
    rsi_dec = rsi_decision(symbol)
    if rsi_dec not in ['BUY', 'SELL']:
        print(f"RSI decision is HOLD for {symbol}")
        return {
            'trend': 'HOLD',
            'suggest_entry_price': 0,
            'suggest_stop_loss': 0,
            'suggest_take_profit': 0,
        }

    try:
        # Latest price
        latest_close = df[price_column].iloc[-1]

        # Price position within Bollinger Bands
        price_position_percentage = (latest_close - latest_lower_band) / band_range

        # Determine the trend
        trend = 'HOLD'
        if latest_close >= latest_upper_band and rsi_dec == 'SELL':  # Above upper band
            trend = 'SELL'
        elif latest_close <= latest_lower_band and rsi_dec == 'BUY':  # Below lower band
            trend = 'BUY'
        elif price_position_percentage is not None:
            if price_position_percentage >= sell_threshold and rsi_dec == 'SELL':
                trend = 'SELL'
            elif price_position_percentage <= buy_threshold and rsi_dec == 'BUY':
                trend = 'BUY'

        if trend == 'HOLD':
            return {
                'trend': trend,
                'suggest_entry_price': 0,
                'suggest_stop_loss': 0,
                'suggest_take_profit': 0,
            }

        # Calculate risk and reward percentages
        entry_price = latest_close
        stop_loss_percentage = max_loss_usdt / entry_price
        take_profit_percentage = stop_loss_percentage * 2  # Ensure 1:2 risk-reward ratio

        # Scale stop loss and take profit for leverage
        stop_loss_percentage_leverage = stop_loss_percentage * leverage
        take_profit_percentage_leverage = take_profit_percentage * leverage

        # Calculate stop loss and take profit levels, adjusted for leverage
        if trend == 'BUY':
            stop_loss = latest_lower_band - (latest_lower_band * 0.005)
            take_profit = latest_upper_band - (0.5 * band_range)
        elif trend == 'SELL':
            stop_loss = latest_upper_band + (latest_upper_band * 0.005)
            take_profit = latest_lower_band + (0.5 * band_range)

        return {
            'trend': trend,
            'suggest_entry_price': entry_price,
            'suggest_stop_loss': stop_loss,
            'suggest_take_profit': take_profit,
        }

    except Exception as e:
        print(f"Error in combine_bollinger_and_rsi for {symbol}: {e}")
        return {
            'trend': 'HOLD',
            'suggest_entry_price': 0,
            'suggest_stop_loss': 0,
            'suggest_take_profit': 0,
        }


    
# ++ AVOID SECTION ++

# Calculating avoid market situation to HOLD trading in extreem condition

# Calculate Bollinger 
def bollinger_band_width(df):
    """Calculate Bollinger Band width."""
    sma = df['close'].rolling(window=20).mean()
    upper_band = sma + (df['close'].rolling(window=20).std() * 2)
    lower_band = sma - (df['close'].rolling(window=20).std() * 2)
    width = upper_band.iloc[-1] - lower_band.iloc[-1]
    return width

# Calculate ADX
def adx_indicator(df):
    """Calculate ADX and return its value, removing NaN values."""
    # Ensure that the dataframe contains the necessary columns
    if 'high' not in df or 'low' not in df or 'close' not in df:
        raise ValueError("Dataframe must contain 'high', 'low', and 'close' columns.")
    
    # Calculate ADX using pandas_ta with a default length of 14
    adx_df = pd_ta.adx(df['high'], df['low'], df['close'], length=14)  # ADX returns a DataFrame
    
    # Select only the ADX column from the resulting DataFrame
    df['ADX'] = adx_df['ADX_14']

    # Drop any NaN values that may be present
    df = df.dropna(subset=['ADX'])

    # Return the last valid ADX value
    return df['ADX'].iloc[-1] 

# Decission on AVOID market condition
def avoid_conditions(symbol,df):
    """Check conditions to avoid entering a trade."""
    # Calculate ATR using a rolling window of 14 periods (default for ATR)
    atr = df['close'].rolling(window=14).apply(lambda x: x.max() - x.min())
    
    # Get the latest ATR value
    latest_atr = atr.iloc[-1]

    # Example conditions:
    # 1. Avoid if ATR is too low (indicating low volatility)
    if latest_atr < 0.005:  # Adjust threshold as needed
        print(f"* * * Avoiding trade {symbol} due to low ATR ({latest_atr} < 0.005)")
        return True

    # 2. Avoid if ADX is above 25, indicating a strong trend (could be adjusted)
    adx_value = adx_indicator(df)
    if adx_value > 25:
        print(f"* * * Avoiding trade {symbol} due to strong trend ADX ({adx_value} > 25)")
        return True

    return False

# Function to calculate ATR (Average True Range)
def calculate_atr(df, period=14):
    """Calculate ATR (Average True Range) for volatility measurement."""
    atr = pd_ta.atr(df['high'], df['low'], df['close'], length=period)
    return atr.iloc[-1]  # Return the most recent ATR value

# ++ ORDER SECTION ++

from binance.enums import SIDE_BUY, SIDE_SELL

# Get Symbol precission to match symbol precission when placing order 
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

# +++ Placing future Order with stop loss and take profit +++
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



# ++ MAIN SECTION ++

# Calling the whole function to start trading process
import threading

# Initialize the list of coin pairs with a default value
coin_pairs = ['DOTUSDT','ATOMUSDT']  # Example with 4 pairs
# Flag to control whether the fetch_recent_orders task should be canceled
cancel_fetch_orders = False
# Function to handle trading for each symbol
def run_symbol_task(symbol):
    print(f"+ + + Running bot for == {symbol} ==")
    
    usdt_to_trade = Decimal('5')  # Example trade amount

    # Fetch trade suggestion
    trade_suggestion = combine_bollinger_and_rsi(symbol)

    # Debugging: Check the type and content of trade_suggestion
    #print(f"Debug: trade_suggestion = {trade_suggestion} ({type(trade_suggestion)})")

    # Validate trade_suggestion
    if not isinstance(trade_suggestion, dict):
        print(f"Error: Expected trade_suggestion to be a dictionary, but got {type(trade_suggestion)}")
        return

    # Ensure required keys exist in trade_suggestion
    required_keys = ['trend', 'suggest_entry_price', 'suggest_stop_loss', 'suggest_take_profit']
    for key in required_keys:
        if key not in trade_suggestion:
            print(f"Error: Missing key '{key}' in trade_suggestion. Received: {trade_suggestion}")
            return

    # Extract the trade signal
    trade_signal = trade_suggestion['trend']
    stop_loss = trade_suggestion['suggest_stop_loss']
    take_profit = trade_suggestion['suggest_take_profit']

    # Check if the trade signal is actionable
    if trade_signal in ['BUY', 'SELL']:
        leverage = Decimal('5')  # Fixed leverage
        entry_price = Decimal(str(trade_suggestion['suggest_entry_price']))
        effective_position_size = usdt_to_trade * leverage
        quantity = effective_position_size / entry_price

        # Debugging: Print trade details
        print(f"Symbol: {symbol}")
        print(f"Trend: {trade_signal}")
        print(f"Trade Size: {usdt_to_trade}")
        print(f"Leverage: {leverage}")
        print(f"Entry Price: {entry_price}")
        print(f"Stop Loss: {stop_loss}")
        print(f"Take Profit: {take_profit}")
        print(f"Effective Position Size: {effective_position_size}")
        print(f"Quantity: {quantity}")

        # Place order
        
        place_futures_order(
            symbol, trade_signal, entry_price,
            Decimal(str(trade_suggestion['suggest_stop_loss'])),
            Decimal(str(trade_suggestion['suggest_take_profit'])),
            leverage, quantity, usdt_to_trade, effective_position_size
        )
        
    else:
        print(f"Trade Suggestion = {trade_signal}")


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
        scheduler.add_job(run_trading_bot_task, IntervalTrigger(seconds=10), max_instances=1)  # Adjust the interval as needed
        #scheduler.add_job(fetch_and_store_orders, IntervalTrigger(seconds=60), max_instances=1)  # Track orders every 30 seconds
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
@run_bot_real_bp.route('/start_bot')
def start_bot():
    """Start the trading bot."""
    start_trading_bot()
    return jsonify({"message": "BOT Started!"})

@run_bot_real_bp.route('/stop_bot')
def stop_bot():
    """Stop the trading bot."""
    stop_trading_bot()
    return jsonify({"message": "BOT Stopped!"})



