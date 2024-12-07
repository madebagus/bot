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
from decimal import Decimal
import pandas_ta as pd_ta
from binance_bot.messaging.chat_bot import send_telegram_message
from binance_bot.data.database_management import insert_orders
from binance_bot.routers.wallet import safe_trade_amount


# Initialize the blueprint for the bot
run_bot_real_bp = Blueprint('run_bot_atr_new', __name__, url_prefix='/api/bot/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

def sync_binance_time(client):
    server_time = client.get_server_time()
    local_time = int(time.time() * 1000)
    client.time_offset = server_time['serverTime'] - local_time
    #Sprint(f"Time offset set to {client.time_offset} ms")

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

# use for ADX > 20

def check_entry_signal(df, symbol):
    """
    Determines if a coin meets entry criteria based on combined indicators.
    Parameters:
        df (pd.DataFrame): Dataframe containing price data for the coin.
        symbol (str): The coin symbol (e.g., 'BTCUSDT').
    Returns:
        str: 'BUY', 'SELL', or 'HOLD' based on entry conditions.
    """
    try:
        # Calculate indicators
        rsi = pd_ta.rsi(df['close'], length=9).iloc[-1]
        prev_rsi = pd_ta.rsi(df['close'], length=9).iloc[-2]
        macd_result = pd_ta.macd(df['close'], fast=6, slow=13, signal=4)
        macd = macd_result[f"MACD_6_13_4"].iloc[-1]
        macd_signal = macd_result[f"MACDs_6_13_4"].iloc[-1]
        atr = pd_ta.atr(df['high'], df['low'], df['close'], length=9).iloc[-1]
        upper_band, lower_band, bb_mean = pd_ta.bbands(df['close'], length=9)[
            ['BBU_9_2.0', 'BBL_9_2.0', 'BBM_9_2.0']].iloc[-1]

        price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]

        # Indicators logic
        if(rsi < 20) or (rsi >= 30 and prev_rsi < 30): # Only entry after oversold reversal
            rsi_dec = 'BUY' 
        elif (rsi > 80) or (rsi <= 70 and prev_rsi > 70): # Only entry after overbough reversal 
            rsi_dec = 'SELL'
        else:
            rsi_dec = 'HOLD' 

        #Indicator macd
        if (macd > macd_signal and macd <= 0):
            macd_dec = 'BUY'
        elif (macd < macd_signal and macd >= 0):
            macd_dec = 'SELL'
        else:
            macd_dec = 'HOLD'
        
        #Indicator Bollinger
        if (price <= lower_band and price < bb_mean):
            bb_dec = 'BUY'
        elif (price >= upper_band and price > bb_mean):
            bb_dec = 'SELL'
        else:
            bb_dec = 'HOLD'
        
        volume_spike = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5

        atr_threshold = atr * 0.25
        price_change = abs(price - prev_price)

        # Final signal logic
        if price_change >= atr_threshold:
            if rsi_dec == 'BUY' and macd_dec == 'BUY' and bb_dec=='BUY':
                return 'BUY'
            elif rsi_dec=='SELL' and macd_dec=='SELL' and bb_dec=='SELL':
                return 'SELL'
            
            #print(f"[+ SIGNAL +] {symbol} atr_threshold: {atr_threshold:.2f}, price_change: {price_change:.2f}, rsi: {rsi_dec}, macd: {macd_dec}, boll: {bb_dec}, spike: {volume_spike}")

        return 'HOLD'

    except Exception as e:
        print(f"Error in check_entry_signal for {symbol}: {e}")
        return 'HOLD'

# use for ADX < 20

def check_normal_trend_signal(df, symbol):
    """
    Determines if a coin meets entry criteria based on combined indicators.
    Parameters:
        df (pd.DataFrame): Dataframe containing price data for the coin.
        symbol (str): The coin symbol (e.g., 'BTCUSDT').
    Returns:
        str: 'BUY', 'SELL', or 'HOLD' based on entry conditions.
    """
    try:
        # Calculate indicators
        rsi = pd_ta.rsi(df['close'], length=9).iloc[-1]
        prev_rsi = pd_ta.rsi(df['close'], length=9).iloc[-2]
        macd_result = pd_ta.macd(df['close'], fast=6, slow=13, signal=4)
        macd = macd_result[f"MACD_6_13_4"].iloc[-1]
        macd_signal = macd_result[f"MACDs_6_13_4"].iloc[-1]
        atr = pd_ta.atr(df['high'], df['low'], df['close'], length=9).iloc[-1]
        upper_band, lower_band, bb_mean = pd_ta.bbands(df['close'], length=9)[
            ['BBU_9_2.0', 'BBL_9_2.0', 'BBM_9_2.0']].iloc[-1]

        price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]

        # Indicators RSI 
        if (rsi <= 30) or (rsi > 30 and prev_rsi < 30):
            rsi_dec = 'BUY'
        elif(rsi >= 70) or (rsi < 70 and prev_rsi > 70):
            rsi_dec = 'SELL'
        else:
            rsi_dec = 'HOLD'
        
        # Indicator MACD
        if (macd > macd_signal and macd <= 0):
            macd_dec = 'BUY'
        elif (macd < macd_signal and macd >= 0):
            macd_dec = 'SELL'
        else:
            macd_dec = 'HOLD'

        position_percentage = (price - lower_band) / (upper_band - lower_band)
        
        # Indicator Bollinger
        if ((price <= lower_band or position_percentage <= 0.05) and price < bb_mean):
            bb_dec = 'BUY'
        elif ((price >= upper_band or position_percentage >= 0.95) and price > bb_mean):
            bb_dec = 'SELL'
        else:
            bb_dec = 'HOLD' 
        
        #volume_spike = df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] * 1.5

        atr_threshold = atr * 0.15
        price_change = abs(price - prev_price)
        
        # Final signal logic
        if price_change >= atr_threshold:
            if rsi_dec == 'BUY' and bb_dec == 'BUY':
                return 'BUY'
            elif rsi_dec == 'SELL' and bb_dec == 'SELL':
                return 'SELL'
            
            #print(f"[+ SIGNAL +] {symbol} atr_threshold: {atr_threshold:.2f}, price_change: {price_change:.2f}, rsi: {rsi_dec}, boll:{bb_dec}, position_percentage: {position_percentage:.2f}")

        return 'HOLD'

    except Exception as e:
        print(f"Error in check_entry_signal for {symbol}: {e}")
        return 'HOLD'


# Combine rsi and bollinger 
def combine_bollinger_and_rsi(
    symbol, 
    price_column='close',  
    adx_threshold=25,
):
    """
    Combine Bollinger Bands and RSI to determine a trade trend (BUY, SELL, or HOLD).
    When ADX > 25, combine MACD with RSI and Bollinger Bands for stronger trend confirmation.
    """
    # Fetch historical data
    df = get_historical_data(symbol)
    if df is None or df.empty:
        print(f"Insufficient data for {symbol}")
        return {'trend': 'HOLD', 'entry_price': 0, 'adx': 0}

    # Ensure the price column exists
    if price_column not in df.columns:
        print(f"Missing column '{price_column}' in data for {symbol}")
        return {'trend': 'HOLD', 'entry_price': 0, 'adx':0}

    # Drop rows with NaN values
    df = df.dropna()

    # Check for avoid conditions
    if avoid_conditions(symbol, df):
        print(f"Decision is HOLD for {symbol} due to avoid conditions.")
        return {'trend': 'HOLD', 'entry_price': 0, 'adx' : 0}
    
    
    # Check ADX for trend confirmation
    latest_adx = adx_indicator(df)
    trend = 'HOLD'  # Default trend

    try:
        # Latest price
        latest_close = df[price_column].iloc[-1]

        # If ADX > 25, use MACD for confirmation
        if latest_adx > adx_threshold:
            # Calculate MACD
            signal = check_entry_signal(df, symbol)
            
            # Combine MACD with RSI and Bollinger Bands
            if signal == 'BUY':
                trend = 'BUY'
            elif signal == 'SELL':
                trend = 'SELL'
             
            print(f"[> > Strong Trend] {symbol} adx: {latest_adx:.2f}, strong_signal: {trend}" )

        # If ADX <= 25, use only RSI and Bollinger Bands
        elif latest_adx < adx_threshold:

            signal = check_normal_trend_signal(df, symbol)

            # Combine MACD with RSI and Bollinger Bands
            if signal == 'BUY':
                trend = 'BUY'
            elif signal == 'SELL':
                trend = 'SELL'
            
            print(f"[+ + Normal Trend] {symbol}, adx: {latest_adx:.2f}, normal_signal:{signal}")

        return {'trend': trend, 'entry_price': latest_close, 'adx':latest_adx}

    except Exception as e:
        print(f"Error in combine_bollinger_and_rsi for {symbol}: {e}")
        return {'trend': 'HOLD-S', 'entry_price': 0, 'adx':latest_adx}

    
# ++ AVOID SECTION ++

# Calculating avoid market situation to HOLD trading in extreem condition

# Calculate ADX
def adx_indicator(df):
    """Calculate ADX and return its value, removing NaN values."""
    # Ensure that the dataframe contains the necessary columns
    if 'high' not in df or 'low' not in df or 'close' not in df:
        raise ValueError("Dataframe must contain 'high', 'low', and 'close' columns.")
    
    # Calculate ADX using pandas_ta with a default length of 9
    adx_df = pd_ta.adx(df['high'], df['low'], df['close'], length=9)  # ADX returns a DataFrame
    
    # Select only the ADX column from the resulting DataFrame
    df['ADX'] = adx_df['ADX_9']

    # Drop any NaN values that may be present
    df = df.dropna(subset=['ADX'])

    # Return the last valid ADX value
    return df['ADX'].iloc[-1] 


# Decission on AVOID market condition
def avoid_conditions(symbol,df):
    """Check conditions to avoid entering a trade."""
    # Calculate ATR using a rolling window of 14 periods (default for ATR)
    atr = df['close'].rolling(window=9).apply(lambda x: x.max() - x.min())
    
    # Get the latest ATR value
    latest_atr = atr.iloc[-1]

    # Example conditions:
    # 1. Avoid if ATR is too low (indicating low volatility)
    if latest_atr < 0.005:  # Adjust threshold as needed
        print(f"* * * Avoiding trade {symbol} due to low ATR ({latest_atr} < 0.005)")
        return True

    return False

# Function to calculate ATR (Average True Range)
def calculate_atr(df, period=9):
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

# placing future order at the mark price 

def place_futures_order(symbol, trend, leverage, quantity, entry_price, usdt_to_trade, trend_condition):
    """
    Place a market order to open a position without setting stop loss or take profit.
    """
    try:
        # Ensure time is synced
        #sync_binance_time(client)
        # Ensure proper data types for the parameters
        leverage = Decimal(leverage) if not isinstance(leverage, Decimal) else leverage
        quantity = Decimal(quantity) if not isinstance(quantity, Decimal) else quantity

        # Check for existing open positions for the symbol
        open_positions = client.futures_position_information(recvWindow=10000, symbol=symbol)
        for position in open_positions:
            if position['positionSide'] == ('LONG' if trend == 'BUY' else 'SHORT') and float(position['positionAmt']) != 0:
                print(f"Existing position detected for {symbol} in {trend} direction. Skipping new order.")
                return  # Exit if there's already an open position in the same direction

        # Get symbol quantity precision
        _, qty_precision = get_symbol_precision(symbol)

        # Round quantity to the symbol's precision
        quantity = format_decimal(quantity, qty_precision)

        # Set leverage for the symbol
        client.futures_change_leverage(symbol=symbol, leverage=int(leverage))

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
        message = f"+ + + Placed Order {symbol}\nSide: {trend}\nTrend: {trend_condition}\nQuantity: {quantity}\nSize:{usdt_to_trade}\nLeverage: {leverage}\nEntry Price:{entry_price}"
        # send telegram message
        send_telegram_message(message)
        # insert to database for analysis
        insert_orders(symbol,trend,entry_price,quantity)

    except Exception as e:
        print(f"Error placing order for {symbol} : {e}")



# ++ MAIN SECTION ++

# Calling the whole function to start trading process
import threading

# Initialize the list of coin pairs with a default value
coin_pairs = ['BCHUSDT','DOTUSDT','LTCUSDT','XMRUSDT','ETHUSDT','INJUSDT','XRPUSDT','BNBUSDT','SUIUSDT','MAGICUSDT']  # Example with 4 pairs
# Flag to control whether the fetch_recent_orders task should be canceled
cancel_fetch_orders = False
# Function to handle trading for each symbol
def run_symbol_task(symbol):
    print(f"+ + + Running bot for {symbol}")
    num_symbols = len(coin_pairs)
    safe_trade_usdt = safe_trade_amount(num_symbols,two_side=True)
    usdt_to_trade = Decimal(safe_trade_usdt)  # Example trade amount

    #print (f"safe amount per trade: {usdt_to_trade}")
    # Fetch trade suggestion
    trade_signal_sugest = combine_bollinger_and_rsi(symbol)
    trade_signal = trade_signal_sugest['trend']
    entry_price = Decimal(str(trade_signal_sugest['entry_price']))
    adx = trade_signal_sugest['adx']

    if adx > 25: 
        trend_condition = 'Strong Trend'
    else:
        trend_condition = 'Normal Trend'

    # Check if the trade signal is actionable
    if trade_signal in ['BUY', 'SELL']:
        leverage = Decimal('5')  # Fixed leverage
        effective_position_size = usdt_to_trade * leverage
        quantity = effective_position_size / entry_price

        # Debugging: Print trade details
        print(f"trend condition: {trend_condition}")
        print(f"Symbol: {symbol}")
        print(f"Trend: {trade_signal}")
        print(f"Leverage: {leverage}")
        print(f"Quantity: {quantity}")

        # Place order
        
        place_futures_order(
            symbol, 
            trade_signal,
            leverage, 
            quantity,
            entry_price, 
            usdt_to_trade,
            trend_condition
        )
        
    #else:
    #    print(f"Trade Suggestion = {trade_signal}")


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
        #scheduler.add_job(process_orders, IntervalTrigger(seconds=55), max_instances=1)  # Track orders every 30 seconds
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



