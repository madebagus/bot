import os
from symtable import Symbol
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


# Initialize the blueprint for the bot
run_bot_real_bp = Blueprint('run_bot_atr', __name__, url_prefix='/api/bot/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

# Fetch historical data from Binance

def get_historical_data(symbol, interval, lookback="1 day ago UTC"):
    """Fetch historical candlestick data from Binance API."""
    klines = client.get_historical_klines(symbol, interval, lookback)
    
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    
    # Convert relevant columns to numeric types
    df['high'] = pd.to_numeric(df['high'])
    df['low'] = pd.to_numeric(df['low'])
    df['close'] = pd.to_numeric(df['close'])
    df['open'] = pd.to_numeric(df['open'])
    df['volume'] = pd.to_numeric(df['volume'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set the timestamp as the index
    df.set_index('timestamp', inplace=True)
    
    return df

# Calculate RSI using ta
def calculate_rsi(df, period=14):
    """Calculate RSI using ta"""
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    return df


# Function to calculate MACD with divergence checks
import ta
import pandas as pd

import ta

# Calculate MACD using ta
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    """Calculate MACD using ta"""
    macd = ta.trend.MACD(df['close'], window_slow=long_period, window_fast=short_period, window_sign=signal_period)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    return df

# Calculate Bollinger Bands using ta
def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands using ta"""
    bollinger = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std_dev)
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_lower'] = bollinger.bollinger_lband()
    return df

def calculate_ema(data, period=50):
    """
    Calculate the Exponential Moving Average (EMA) using the ta library
    :param data: List of historical price data (OHLCV format)
    :param period: The period for the EMA (e.g., 50 or 200)
    :return: The calculated EMA value
    """
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])  # Create a DataFrame from the data
    df['ema'] = ta.trend.ema_indicator(df['close'], window=period)  # Use ta to calculate EMA
    return df['ema'].iloc[-1]  # Return the latest EMA value

# Calculate indicators for a given timeframe
def calculate_indicators(symbol, interval):
    """Combine indicators using ta"""
    df = get_historical_data(symbol, interval, lookback="1 week ago UTC")

    # Calculate RSI
    if len(df) >= 14:
        df = calculate_rsi(df)
    else:
        print("Not enough data for RSI calculation")

    # Calculate MACD
    df = calculate_macd(df)

    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df)

    # Calculate 50-period EMA
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)

    # Volume trend (Basic comparison to rolling mean)
    df['volume_trend'] = df['volume'] > df['volume'].rolling(window=10).mean()

    # Add ATR calculation in calculate_indicators
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    
    return df

# Determine trend for each indicator (long or short)
def determine_trend_for_indicators(df):
    """Determine trend based on indicators"""
    trend = {'rsi': None, 'macd': None, 'bollinger': None, 'volume': None, 'ema': None}
    
    print("Calculating trend")
    # RSI Trend with middle line check
    rsi = df['rsi'].iloc[-1]  # Get the latest RSI value (from the last row of the DataFrame)
    previous_rsi = df['rsi'].iloc[-2]  # Get the previous RSI value (from the second-to-last row)

    # Check for RSI crossing the middle line (50) and overbought/oversold zones
    if rsi < 30 and previous_rsi >= 50:
        trend['rsi'] = 'BUY'  # Crossing down fast
    if rsi > 50 and previous_rsi >= 80:
        trend['rsi'] = 'SELL'  # Crossing down fast
    if rsi < 50 and previous_rsi >= 70:
        trend['rsi'] = 'SELL'  # Crossing down fast    
    elif rsi > 50 and previous_rsi <= 30:
        trend['rsi'] = 'BUY'  # Crossing up fast
    elif rsi > 30 and previous_rsi <= 20:
        trend['rsi'] = 'BUY'  # Crossing up fast
    elif rsi <= 20:
        trend['rsi'] = 'BUY'  # Over Bought
    elif rsi >= 80:
        trend['rsi'] = 'SELL'  # Over Sold
    else: 
        trend['rsi'] = 'No Trade'

    print("RSI Trend:", trend['rsi'])

    
    # MACD Trend
    macd = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    if macd > macd_signal:
        trend['macd'] = 'BUY'
    elif macd < macd_signal:
        trend['macd'] = 'SELL'

    # Output the trend for MACD
    print("MACD Trend:", trend['macd'])

    # Bollinger Bands Trend with Delta Check
    close_price = df['close'].iloc[-1]
    upper_band = df['bollinger_upper'].iloc[-1]
    lower_band = df['bollinger_lower'].iloc[-1]
    # Calculate the delta between the upper and lower bands
    delta = upper_band - lower_band

    # Define a threshold for delta (e.g., 0.01% of the close price or a fixed amount like 0.10)
    threshold = 0.25 # This can be adjusted based on your preference (0.10 means 10 cents difference)

    # If the delta is too small, avoid trading
    if delta < threshold:
        trend['bollinger'] = 'No Trade'  # Avoid trading if the bands are too close
    else:
        if close_price >= upper_band:
            trend['bollinger'] = 'SELL'  # Overbought, so SELL
        elif close_price <= lower_band:
            trend['bollinger'] = 'BUY'  # Oversold, so BUY
        else:
            trend['bollinger'] = 'No Trade'  # No trade if price is within the bands

    print("bollinger Trend:", trend['bollinger'])

    # Volume Trend based on Buy and Sell Volume
    buy_volume = df.loc[df['close'] > df['open'], 'volume'].sum()
    sell_volume = df.loc[df['close'] < df['open'], 'volume'].sum()

    # Determine volume trend based on comparison
    volume_trend = 'BUY' if buy_volume > sell_volume else 'SELL'
    trend['volume'] = volume_trend
    
    print("volume Trend:", trend['volume'])

    # EMA 50 Trend with slope and ATR buffer
    ema_50 = df['EMA_50'].iloc[-1]
    prev_ema_50 = df['EMA_50'].iloc[-2]  # Previous EMA value for slope
    close_price = df['close'].iloc[-1]
    atr_value = df['atr'].iloc[-1]  # Assuming ATR was already calculated in df

    # Check EMA slope and buffer distance
    if close_price > ema_50 and (ema_50 > prev_ema_50) and (close_price > ema_50 + (0.5 * atr_value)):
        trend['ema'] = 'BUY'
    elif close_price < ema_50 and (ema_50 < prev_ema_50) and (close_price < ema_50 - (0.5 * atr_value)):
        trend['ema'] = 'SELL'
    else:
        trend['ema'] = 'No Trade'  # Sideways trend if not a clear breakout
    
    print("ema Trend:", trend['ema'])

    return trend

# Score the trends across multiple timeframes
def score_trends(trends):
    """Score the trends"""
    score = 0
    for trend in trends:
        if trend == 'BUY':
            score += 1
        elif trend == 'SELL':
            score -= 1
        elif trend == 'No Trade':
            score = 0
    return score

def calculate_atr(df, window=14):
    """Calculate ATR for volatility-based stop loss and take profit"""
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")

    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window).average_true_range()
    return df
    
#calculate the trend, entry price, stop loss and take profit
def suggest_trade(symbol, atr_multiplier=5):  # Default ATR multiplier set to 2.5
    """Return trade suggestion based on multi-timeframe indicator trends"""
    timeframes = ['5m', '15m', '1h', '4h']
    trend_scores = []

    try:
        # Calculate indicator trends for each timeframe
        for interval in timeframes:
            data = calculate_indicators(symbol, interval)
            trend = determine_trend_for_indicators(data)
            trend_scores.append(trend)

        # Aggregate trends for each indicator
        rsi_trends = [t['rsi'] for t in trend_scores]
        macd_trends = [t['macd'] for t in trend_scores]
        bollinger_trends = [t['bollinger'] for t in trend_scores]
        volume_trends = [t['volume'] for t in trend_scores]
        ma_trends = [t['ema'] for t in trend_scores]  # Ensure 'ma' is added here for MA trend

        # Get the overall trend scores
        rsi_score = score_trends(rsi_trends)
        macd_score = score_trends(macd_trends)
        bollinger_score = score_trends(bollinger_trends)
        volume_score = score_trends(volume_trends)
        ma_score = score_trends(ma_trends)

        trend_score = [rsi_score,macd_score,bollinger_score,volume_score,ma_score]
        buy_score = trend_score.count(1)
        sell_score = trend_score.count(-1)


        # Decide final trend based on total score
        if buy_score >= 3:
            final_trend = 'BUY'
        elif sell_score >= 3:
            final_trend = 'SELL'
        else:
            final_trend = 'No Trade Signal'
        

        # Get current price and apply dynamic stop loss and take profit based on ATR
        price_data = get_historical_data(symbol, '1m')
        current_price = price_data['close'].iloc[-1]

        # Ensure ATR data is available
        data_with_atr = calculate_atr(price_data)
        if 'atr' not in data_with_atr.columns or data_with_atr['atr'].iloc[-1] is None:
            raise ValueError("ATR data is not available")

        # Set dynamic stop loss and take profit based on ATR with multiplier
        atr_value = data_with_atr['atr'].iloc[-1] * atr_multiplier  # Adjust ATR with multiplier

        if final_trend == 'BUY':
            stop_loss = current_price - atr_value  # ATR-based stop loss
            take_profit = current_price + atr_value  # ATR-based take profit
        elif final_trend == 'SELL':
            stop_loss = current_price + atr_value  # ATR-based stop loss
            take_profit = current_price - atr_value  # ATR-based take profit
        else:
            stop_loss = take_profit = current_price  # No trade signal

        return {
            'trend': final_trend,
            'suggest_entry_price': current_price,
            'suggest_stop_loss': stop_loss,
            'suggest_take_profit': take_profit
        }

    except Exception as e:
        print(f"Error generating trade suggestion for {symbol}: {e}")
        return None



# Function to insert an order into the database
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_LIMIT

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

        # Place the limit order for entry
        order = client.futures_create_order(
            symbol=symbol,
            side=SIDE_BUY if trend == 'BUY' else SIDE_SELL,
            type='LIMIT',
            quantity=float(quantity),
            price=float(entry_price),  # Ensure price is set for the limit order
            timeInForce='GTC',
            positionSide=position_side  # Specify the position side (LONG or SHORT)
        )
        print(f"Placed limit order: {order}")

        # Fetch open orders to check if stop loss and take profit orders already exist
        orders = client.futures_get_open_orders(symbol=symbol)
        existing_stop_loss_orders = [order for order in orders if order['type'] == 'STOP_MARKET' and order['positionSide'] == position_side]
        existing_take_profit_orders = [order for order in orders if order['type'] == 'TAKE_PROFIT_MARKET' and order['positionSide'] == position_side]


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
        print(f"Error placing order: {e}")



#fetching last 4 hours order from binance and insert or update order in table by calling process_order
from datetime import datetime, timedelta
import pytz
import mysql.connector

def fetch_recent_orders():
    # Calculate the start time as 4 hours ago from the current time
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(hours=4)
    
    # Convert to timestamps for the Binance API
    start_timestamp = int(start_time.timestamp() * 1000)  # in milliseconds
    end_timestamp = int(end_time.timestamp() * 1000)      # in milliseconds
    
    try:
        # Fetch recent futures orders across all symbols within the 4-hour window
        recent_orders = client.futures_get_all_orders(
            startTime=start_timestamp,
            endTime=end_timestamp
        )
        
        for order in recent_orders:
            # Insert or update each order in the database based on order ID and status
            insert_update_orders(order)
            
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
            

# Define the list of coin pairs you want the bot to handle
coin_pairs = ['DOTUSDT','LTCUSDT']

def run_trading_bot_task():
    for symbol in coin_pairs:
        print(f"Running bot for {symbol}")
        
        # Get available balance from the wallet
        available_balance = get_wallet_balance()  # Use the wallet's API to get the balance
        num_symbols = 1  # Example: Number of symbols being traded
        two_sided = True  # Example: Whether two-sided orders are being placed
        
        # Call the function to calculate the safe trade amount per symbol
        usdt_to_trade = calculate_dynamic_safe_trade_amount(available_balance, num_symbols, two_sided)

        # Suggest trade for the current coin pair
        trade_suggestion = suggest_trade(symbol)

        if trade_suggestion is not None and trade_suggestion['trend'] != 'No Trade Signal':

            # Set leverage and get safe trade amount dynamically
            leverage = Decimal('5')  # Fixed leverage (2x)
            
            # Calculate the entry price (ensure it's a Decimal)
            entry_price = Decimal(str(trade_suggestion['suggest_entry_price']))

            # Calculate the effective position size based on leverage
            effective_position_size = usdt_to_trade * leverage

            # Calculate the quantity based on the entry price and effective position size
            quantity = effective_position_size / entry_price

            # Debugging: Check the calculated values
            print(f"Leverage: {leverage}")
            print(f"USDT to trade: {usdt_to_trade}")
            print(f"Entry Price: {entry_price}")
            print(f"Effective Position Size: {effective_position_size}")
            print(f"Quantity: {quantity}")

            # Insert the order with leverage and quantity
            place_futures_order(symbol, 
                                trade_suggestion['trend'],
                                entry_price,
                                Decimal(str(trade_suggestion['suggest_stop_loss'])),
                                Decimal(str(trade_suggestion['suggest_take_profit'])),
                                leverage,
                                quantity, 
                                usdt_to_trade, 
                                effective_position_size)
        else:
            print("Trade suggestion : ",trade_suggestion['trend'])


# Scheduler to run the task periodically
scheduler = BackgroundScheduler()
if not scheduler.running:
    scheduler.start()

scheduler.add_job(run_trading_bot_task, IntervalTrigger(seconds=30), max_instances=1)  # Adjust the interval as needed
scheduler.add_job(fetch_recent_orders, IntervalTrigger(seconds=60), max_instances=1)  # Track orders every 30 seconds

#run the BOT
@run_bot_real_bp.route('/start_bot')
def start_bot():
    return jsonify({"message": "BOT Started!"})