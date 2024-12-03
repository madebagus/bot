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
    if df.empty or len(df) < period:
        print("Not enough data to calculate RSI.")
        return df  # Return the DataFrame as-is if there's insufficient data

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    df['rsi'] = df['rsi'].fillna(50)  # Replace NaN values with neutral RSI (50)
    return df

# Calculate MACD using ta
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    """Calculate MACD using ta"""
    if df.empty or len(df) < long_period:
        print("Not enough data to calculate MACD.")
        return df  # Return the DataFrame as-is if there's insufficient data

    macd = ta.trend.MACD(df['close'], window_slow=long_period, window_fast=short_period, window_sign=signal_period)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # Fill NaN values with 0 (neutral value)
    df['macd'] = df['macd'].fillna(0)
    df['macd_signal'] = df['macd_signal'].fillna(0)
    df['macd_hist'] = df['macd_hist'].fillna(0)
    
    return df

# Calculate Bollinger Bands using ta
def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculate Bollinger Bands using ta"""
    if df.empty or len(df) < period:
        print("Not enough data to calculate Bollinger Bands.")
        return df  # Return the DataFrame as-is if there's insufficient data

    bollinger = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std_dev)
    df['bollinger_upper'] = bollinger.bollinger_hband()
    df['bollinger_lower'] = bollinger.bollinger_lband()

    # Fill NaN values with close price (or other logical fallback value)
    df['bollinger_upper'] = df['bollinger_upper'].fillna(df['close'])
    df['bollinger_lower'] = df['bollinger_lower'].fillna(df['close'])
    
    return df

# Calculate EMA using ta
def calculate_ema(df, period=50):
    """Calculate the Exponential Moving Average (EMA) using the ta library"""
    if df.empty or len(df) < period:
        print("Not enough data to calculate EMA.")
        return None  # Return None if there's insufficient data
    
    df['ema'] = ta.trend.ema_indicator(df['close'], window=period)
    df['ema'] = df['ema'].fillna(df['close'])  # Replace NaN with close price as a fallback
    return df['ema'].iloc[-1]  # Return the latest EMA value

#calculate from all indicators
def calculate_indicators(symbol, interval):
    """Combine indicators using ta"""
    df = get_historical_data(symbol, interval)
    
    if df.empty:
        print(f"Error: No data for {symbol} on interval {interval}")
        return None  # Return None if there's no data

    # Calculate RSI
    df = calculate_rsi(df)
    # Handle NaN values in RSI and replace with neutral value (50 for RSI)
    df['RSI'] = df['RSI'].fillna(50)

    # Calculate MACD
    df = calculate_macd(df)
    # Handle NaN values in MACD and replace with a neutral value (0 for MACD)
    df['MACD'] = df['MACD'].fillna(0)

    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df)
    # Handle NaN values in Bollinger Bands (if any) and replace with neutral values
    df['Bollinger_Upper'] = df['Bollinger_Upper'].fillna(df['close'])
    df['Bollinger_Lower'] = df['Bollinger_Lower'].fillna(df['close'])

    # Calculate 50-period EMA
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['EMA_50'] = df['EMA_50'].fillna(df['close'])  # Fill NaN with the close price

    # Volume trend (Basic comparison to rolling mean)
    df['volume_trend'] = df['volume'] > df['volume'].rolling(window=10).mean()
    df['volume_trend'] = df['volume_trend'].fillna(False)  # Fill NaN with False (no trend)

    return df

#determine the trend to BUY or SELL
def determine_trend_for_indicators(df):
    """Determine trend based on indicators"""
    trend = {'rsi': None, 'macd': None, 'bollinger': None, 'volume': None, 'ema': None}

    # RSI Trend
    # Add logging to check the values
    print("Calculating RSI trend")
    rsi = df['rsi'].iloc[-1]
    if rsi <= 20:
        trend['rsi'] = 'BUY'
    elif rsi >= 80:
        trend['rsi'] = 'SELL'
    print("RSI Trend:", trend['rsi'])

    # MACD Trend
    macd = df['macd'].iloc[-1] if pd.notna(df['macd'].iloc[-1]) else None
    macd_signal = df['macd_signal'].iloc[-1] if pd.notna(df['macd_signal'].iloc[-1]) else None
    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            trend['macd'] = 'BUY'
        elif macd < macd_signal:
            trend['macd'] = 'SELL'

    # Bollinger Bands Trend
    close_price = df['close'].iloc[-1] if pd.notna(df['close'].iloc[-1]) else None
    upper_band = df['bollinger_upper'].iloc[-1] if pd.notna(df['bollinger_upper'].iloc[-1]) else None
    lower_band = df['bollinger_lower'].iloc[-1] if pd.notna(df['bollinger_lower'].iloc[-1]) else None
    if close_price is not None and upper_band is not None and lower_band is not None:
        if close_price > upper_band:
            trend['bollinger'] = 'BUY'
        elif close_price < lower_band:
            trend['bollinger'] = 'SELL'

    # Volume Trend
    volume_trend = df['volume_trend'].iloc[-1] if pd.notna(df['volume_trend'].iloc[-1]) else None
    if volume_trend is not None:
        trend['volume'] = 'BUY' if volume_trend else 'SELL'

    # EMA 50 Trend
    ema_50_trend = df['EMA_50'].iloc[-1] if pd.notna(df['EMA_50'].iloc[-1]) else None
    if close_price is not None and ema_50_trend is not None:
        if close_price > ema_50_trend:
            trend['ema'] = 'BUY'
        elif close_price < ema_50_trend:
            trend['ema'] = 'SELL'

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
    return score

def calculate_atr(df, window=14):
    """Calculate ATR for volatility-based stop loss and take profit"""
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")

    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window).average_true_range()
    return df

#Signal Confirmation with Multiple Indicators
def signal_confirmation(rsi_trend, macd_trend, bollinger_trend, volume_trend, ma_trend):
    """Check if trade signals are confirmed by multiple indicators"""
    if rsi_trend == 'BUY' and macd_trend == 'BUY' and bollinger_trend == 'BUY' and volume_trend == 'BUY' and ma_trend == 'BUY':
        return 'BUY'
    elif rsi_trend == 'SELL' and macd_trend == 'SELL' and bollinger_trend == 'SELL' and volume_trend == 'SELL' and ma_trend == 'SELL':
        return 'SELL'
    else:
        return 'No Trade'

# Updated dynamic_rsi_threshold function
def dynamic_rsi_threshold(rsi_value, volatility):
    """Adjust RSI threshold based on volatility"""
    if volatility > 0.03:  # High volatility
        return 80  # Overbought if RSI exceeds 80
    else:  # Low volatility
        return 70  # More conservative threshold (70)

#Volatility and Trend Sensitivity
def adjust_position_size(atr_value, current_balance, risk_percentage):
    # Assuming the risk is based on the distance between the entry price and stop loss
    risk_amount = current_balance * risk_percentage  # Amount of balance to risk
    position_size = risk_amount / atr_value  # Calculate how many units to trade based on ATR
    return position_size

#Filter Out Noisy Signals with Confirmation Time
def check_signal_confirmation_time(prices, signal, required_candles=3):
    """Ensure the signal persists over multiple candles before triggering a trade"""
    count = 0
    for i in range(1, required_candles + 1):
        if (signal == 'BUY' and prices[-i] > prices[-(i+1)]) or (signal == 'SELL' and prices[-i] < prices[-(i+1)]):
            count += 1
    return count == required_candles

#calculate the trend, entry price, stop loss and take profit
def suggest_trade(symbol, atr_multiplier=2.0, risk_percentage=0.02):
    """Fine-tuned trade suggestion with multiple indicators and dynamic thresholds"""
    timeframes = ['5m', '15m', '1h', '4h']
    trend_scores = []

    try:
        # Get the data for each timeframe and calculate indicator trends
        for interval in timeframes:
            data = calculate_indicators(symbol, interval)
            trend = determine_trend_for_indicators(data)
            trend_scores.append(trend)

        # Combine indicator trends (using confirmation and dynamic thresholds)
        rsi_trends = [t['rsi'] for t in trend_scores]
        macd_trends = [t['macd'] for t in trend_scores]
        bollinger_trends = [t['bollinger'] for t in trend_scores]
        volume_trends = [t['volume'] for t in trend_scores]
        ma_trends = [t['ema'] for t in trend_scores]

        # Print the RSI trends and dynamic RSI thresholds for debugging
        print(f"RSI Trends: {rsi_trends}")
        price_data = get_historical_data(symbol, '1m')
        data_with_atr = calculate_atr(price_data)
        atr_value = data_with_atr['atr'].iloc[-1]
        # Dynamically adjust the RSI threshold based on ATR (volatility)
        dynamic_rsi = [dynamic_rsi_threshold(rsi, atr_value) for rsi in rsi_trends]
        print(f"Dynamic RSI Thresholds: {dynamic_rsi}")

        # Update RSI trend to reflect dynamic threshold adjustment
        adjusted_rsi_trends = [
            'BUY' if rsi < dynamic_rsi[i] else 'SELL' if rsi > dynamic_rsi[i] else 'No Trade'
            for i, rsi in enumerate(rsi_trends)
        ]


        print(f"Adjusted RSI Trends: {adjusted_rsi_trends}")
        
        # Signal confirmation from adjusted RSI and other indicators
        final_signal = signal_confirmation(rsi_trends, macd_trends, bollinger_trends, volume_trends, ma_trends)
        print(f"Final Signal: {final_signal}")

        if final_signal != 'No Trade':
            current_price = price_data['close'].iloc[-1]

            # Calculate ATR with multiplier for dynamic stop loss and take profit
            atr_value = data_with_atr['atr'].iloc[-1] * atr_multiplier

            stop_loss = current_price - atr_value if final_signal == 'BUY' else current_price + atr_value
            take_profit = current_price + atr_value if final_signal == 'BUY' else current_price - atr_value

            # get wallet balance
            available_balance = get_wallet_balance()
            # Adjust position size based on volatility (ATR)
            position_size = adjust_position_size(atr_value, current_balance=available_balance, risk_percentage=risk_percentage)

            return {
                'trend': final_signal,
                'suggest_entry_price': current_price,
                'suggest_stop_loss': stop_loss,
                'suggest_take_profit': take_profit,
                'suggest_position_size': position_size
            }
        else:
            print(f"No trade signal for {symbol}.")
            return {'trend': 'No Trade'}
    
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
coin_pairs = ['DOTUSDT']

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

        if trade_suggestion['trend'] is None:
            print(f"No trade signal for {symbol}. Skipping.")
            continue

        if trade_suggestion['trend'] == 'No Trade':
            print(f"No trade signal for {symbol}. Skipping.")
            continue
        
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