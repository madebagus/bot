import os
from binance.client import Client
import certifi
import ta
import pandas as pd
from flask import Blueprint, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from dconfig import read_db_config
from conn_ssh import create_conn
from decimal import Decimal
import requests



# Initialize the blueprint for the bot
run_bot_atr_proxy_bp = Blueprint('run_bot_atr_proxy', __name__, url_prefix='/api/bot/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

# Fetch historical data from Binance

def get_historical_data(symbol, interval, lookback="1 day ago UTC"):
    """Fetch historical candlestick data through a proxy to the Binance API."""
    # Define the parameters for the request
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": lookback
    }
    
    # Make the request to your local proxy instead of directly to Binance
    url = f'http://localhost:5000/binance?endpoint=klines'
    
    # Add symbol and interval as query parameters for the proxy
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        klines = response.json()
        
        # Continue with DataFrame processing as before
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                           'close_time', 'quote_asset_volume', 'number_of_trades', 
                                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 
                                           'ignore'])
        
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
    else:
        print("Failed to fetch data from proxy:", response.json())
        return None

# Calculate RSI using ta
def calculate_rsi(df, period=14):
    """Calculate RSI using ta"""
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
    return df

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
    df = get_historical_data(symbol, interval)

    # Calculate RSI
    df = calculate_rsi(df)

    # Calculate MACD
    df = calculate_macd(df)

    # Calculate Bollinger Bands
    df = calculate_bollinger_bands(df)

    # Calculate 50-period EMA
    df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)

    # Volume trend (Basic comparison to rolling mean)
    df['volume_trend'] = df['volume'] > df['volume'].rolling(window=10).mean()
    
    return df

# Determine trend for each indicator (long or short)
def determine_trend_for_indicators(df):
    """Determine trend based on indicators"""
    trend = {'rsi': None, 'macd': None, 'bollinger': None, 'volume': None, 'ema': None}

    # RSI Trend
    rsi = df['rsi'].iloc[-1]
    if rsi < 30:
        trend['rsi'] = 'long'
    elif rsi > 70:
        trend['rsi'] = 'short'

    # MACD Trend
    macd = df['macd'].iloc[-1]
    macd_signal = df['macd_signal'].iloc[-1]
    if macd > macd_signal:
        trend['macd'] = 'long'
    elif macd < macd_signal:
        trend['macd'] = 'short'

    # Bollinger Bands Trend
    close_price = df['close'].iloc[-1]
    upper_band = df['bollinger_upper'].iloc[-1]
    lower_band = df['bollinger_lower'].iloc[-1]
    if close_price > upper_band:
        trend['bollinger'] = 'short'
    elif close_price < lower_band:
        trend['bollinger'] = 'long'

    # Volume Trend
    volume_trend = 'long' if df['volume_trend'].iloc[-1] else 'short'
    trend['volume'] = volume_trend

    # EMA 50 Trend
    ema_50_trend = df['EMA_50'].iloc[-1]
    if close_price > ema_50_trend:
        trend['ema'] = 'long'
    elif close_price < ema_50_trend:
        trend['ema'] = 'short'

    return trend

# Score the trends across multiple timeframes
def score_trends(trends):
    """Score the trends"""
    score = 0
    for trend in trends:
        if trend == 'long':
            score += 1
        elif trend == 'short':
            score -= 1
    return score

def calculate_atr(df, window=14):
    """Calculate ATR for volatility-based stop loss and take profit"""
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")

    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window).average_true_range()
    return df
    

def suggest_trade(symbol):
    """Return trade suggestion based on multi-timeframe indicator trends"""
    timeframes = ['5m', '15m', '1h', '4h']
    trend_scores = []

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
   
    # Weighted scoring (adjusted for HFT strategy)
    rsi_weight = 0.1       # Lower weight for RSI as it's slower to react
    macd_weight = 0.35     # Higher weight for MACD due to its relevance in momentum
    bollinger_weight = 0.2 # Medium weight for Bollinger Bands to capture volatility
    volume_weight = 0.25   # High weight for Volume as it confirms the strength of moves
    ma_weight = 0.1        # Lower weight for MA in HFT to reduce lag
    
    # Calculate total score by applying weights
    total_score = (rsi_score * rsi_weight) + (macd_score * macd_weight) + \
                 (bollinger_score * bollinger_weight) + (volume_score * volume_weight) + (ma_score * ma_weight)

    # Decide final trend based on total score
    if total_score > 0:
        final_trend = 'long'
    elif total_score < 0:
        final_trend = 'short'
    else:
        final_trend = 'no trade signal'

    # Get current price and apply dynamic stop loss and take profit based on ATR
    price_data = get_historical_data(symbol, '1m')
    current_price = price_data['close'].iloc[-1]

    # Calculate ATR
    data_with_atr = calculate_atr(price_data)

    # Set dynamic stop loss and take profit based on ATR
    if final_trend == 'long':
        stop_loss = current_price - data_with_atr['atr'].iloc[-1]  # ATR-based stop loss
        take_profit = current_price + data_with_atr['atr'].iloc[-1]  # ATR-based take profit
    elif final_trend == 'short':
        stop_loss = current_price + data_with_atr['atr'].iloc[-1]  # ATR-based stop loss
        take_profit = current_price - data_with_atr['atr'].iloc[-1]  # ATR-based take profit
    else:
        stop_loss = take_profit = current_price  # No trade signal

    return {
        'trend': final_trend,
        'suggest_entry_price': current_price,
        'suggest_stop_loss': stop_loss,
        'suggest_take_profit': take_profit
    }

# Function to insert an order into the database
def insert_order_if_not_exists(symbol, trend, entry_price, stop_loss, take_profit, leverage, quantity, usdt_to_trade, position_size):
    conn, tunnel = create_conn()
    try:
        cursor = conn.cursor()
        
        # Ensure proper data types
        entry_price = Decimal(entry_price) if not isinstance(entry_price, Decimal) else entry_price
        stop_loss = Decimal(stop_loss) if not isinstance(stop_loss, Decimal) else stop_loss
        take_profit = Decimal(take_profit) if not isinstance(take_profit, Decimal) else take_profit
        usdt_to_trade = Decimal(usdt_to_trade) if not isinstance(usdt_to_trade, Decimal) else usdt_to_trade
        position_size = Decimal(position_size) if not isinstance(position_size, Decimal) else position_size
        leverage = Decimal(leverage) if not isinstance(leverage, Decimal) else leverage
        quantity = Decimal(quantity) if not isinstance(quantity, Decimal) else quantity
        
        # Check if an order already exists for the given symbol and trend
        check_query = """
            SELECT COUNT(*) FROM orders 
            WHERE symbol = %s AND trend = %s AND status IN ('NEW', 'PARTIALLY_FILLED');
        """
        cursor.execute(check_query, (symbol, trend))
        result = cursor.fetchone()
        
        # If no existing order, insert a new one
        if result[0] == 0:
            insert_query = """
                INSERT INTO orders (symbol, trend, entry_price, stop_loss, take_profit, leverage, quantity, usdt_to_trade, position_size, status)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'NEW');
            """
            cursor.execute(insert_query, (symbol, trend, entry_price, stop_loss, take_profit, leverage, quantity, usdt_to_trade, position_size))
            conn.commit()
            print(f"Order inserted: {symbol}, Trend: {trend}, Entry Price: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}, Leverage: {leverage}, Quantity: {quantity}")
        else:
            print(f"Open order already exists for {symbol} with trend {trend}; no new order inserted.")
    
    except Exception as e:
        print(f"Error in insert_order_if_not_exists: {e}")
    finally:
        if conn:
            cursor.close()


# Function to get the current price from Binance (or any other source)
def get_current_price(symbol):
    try:
        # Use the proxy server to fetch current market price
        url = f"http://localhost:5000/binance?endpoint=ticker/price&symbol={symbol}"
        response = requests.get(url)
        data = response.json()
        
        # Check if the response contains the price
        if "price" in data:
            return Decimal(data["price"])  # Return price as Decimal
        else:
            print(f"Error fetching price for {symbol}: {data}")
            return None
    except Exception as e:
        print(f"Error fetching price for {symbol}: {e}")
        return None

#calculate profit or loss, tracking the market movement
def track_and_update_orders():
    conn, tunnel = create_conn()
    try:
        # Fetch open orders from the database
        cursor = conn.cursor()
        select_query = "SELECT symbol,trend,entry_price,stop_loss,take_profit,order_id,usdt_to_trade,leverage,status FROM orders WHERE status IN ('NEW', 'PARTIALLY_FILLED');"
        cursor.execute(select_query)
        orders = cursor.fetchall()

        for order in orders:
            # Extract order details
            symbol, trend, entry_price, stop_loss, take_profit, order_id, usdt_to_trade, leverage, status = (
                order[0], order[1], Decimal(order[2]), Decimal(order[3]), Decimal(order[4]),
                order[5], Decimal(order[6]), Decimal(order[7]), order[8]
            )

            # Get current market price
            current_price = get_current_price(symbol)
            if current_price is None:
                continue  # Skip this order if the current price couldn't be fetched

            # Initialize variables
            percent_profit_loss = Decimal('0.0')
            value_profit_loss = Decimal('0.0')
            closing_price = current_price
            new_status = None  # Only update if an exit condition is met

            # Calculate profit/loss based on trend
            if trend == 'long':
                if current_price <= stop_loss:
                    # Hit stop loss
                    percent_profit_loss = ((current_price - entry_price) / entry_price) * 100
                    value_profit_loss = usdt_to_trade * ((percent_profit_loss / 100) * leverage)
                    new_status = 'CLOSED'
                elif current_price >= take_profit:
                    # Hit take profit
                    percent_profit_loss = ((current_price - entry_price) / entry_price) * 100
                    value_profit_loss = usdt_to_trade * ((percent_profit_loss / 100) * leverage)
                    new_status = 'CLOSED'
            elif trend == 'short':
                if current_price >= stop_loss:
                    # Hit stop loss for short position
                    percent_profit_loss = ((entry_price - current_price) / entry_price) * 100
                    value_profit_loss = usdt_to_trade * ((percent_profit_loss / 100) * leverage)
                    new_status = 'CLOSED'
                elif current_price <= take_profit:
                    # Hit take profit for short position
                    percent_profit_loss = ((entry_price - current_price) / entry_price) * 100
                    value_profit_loss = usdt_to_trade * ((percent_profit_loss / 100) * leverage)
                    new_status = 'CLOSED'

            if new_status == 'CLOSED':
                # Update order status, profit/loss, and closing price in the database
                update_query = """
                    UPDATE orders
                    SET status = %s, percent_profit_loss = %s, value_profit_loss = %s, closing_price = %s
                    WHERE order_id = %s;
                """
                cursor.execute(update_query, (new_status, percent_profit_loss, value_profit_loss, closing_price, order_id))
                conn.commit()

                print(f"Order {order_id} updated: {new_status}, Profit/Loss: {percent_profit_loss}%, {value_profit_loss} USDT")
            else:
                # Log active status for troubleshooting
                print(f"Order {order_id} remains active. Current price: {current_price}, Entry price: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")

    except Exception as e:
        print(f"Error in tracking and updating orders: {e}")
    finally:
        if conn:
            cursor.close()

def calculate_atr_2(df, window=14):
    """Calculate ATR for volatility-based stop loss and take profit"""
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("DataFrame must contain 'high', 'low', and 'close' columns")

    # Calculate ATR using the ta library and return the ATR value as a scalar
    atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window).average_true_range()
    return atr.iloc[-1]  # Return the last value of ATR as a scalar

def calculate_dynamic_allocation(available_balance, leverage, assets):
    allocations = {}
    total_volatility_weight = Decimal('0')
    volatility_data = {}

    for asset in assets:
        # Fetch and process historical data to get the ATR for each symbol individually
        price_data = get_historical_data(asset, '1m')
        if price_data is not None and len(price_data) > 0:
            atr = Decimal(calculate_atr_2(price_data))  # Now we use the ATR scalar value
            volatility_data[asset] = atr
            total_volatility_weight += Decimal(1) / atr
        else:
            print(f"Warning: No historical data available for {asset}. Setting allocation to zero.")
            volatility_data[asset] = Decimal('0')

    for asset, atr in volatility_data.items():
        if atr != 0:
            # Calculate the weight based on ATR
            weight = (Decimal(1) / atr) / total_volatility_weight
            safe_trade_amount = (available_balance * weight) / leverage
            allocations[asset] = float(safe_trade_amount)
        else:
            allocations[asset] = 0.0  # Zero allocation for assets with no data

    return allocations

def dynamic_safe_trade_amount(symbol):
    try:
        leverage = Decimal('5')
        available_balance = Decimal('600')
        assets = [symbol]  # Pass a list with a single symbol
        
        # Calculate allocations for each symbol
        allocations = calculate_dynamic_allocation(available_balance, leverage, assets)

        return allocations.get(symbol, 0.0)  # Return allocation for the specific symbol
        
    except Exception as e:
        print(f"Error calculating dynamic safe trade amount for {symbol}: {e}")
        return 0.0  # Return zero if there was an error
    

# Define the list of coin pairs you want the bot to handle
coin_pairs = ['ETHUSDT', 'SOLUSDT', 'BTCUSDT','BNBUSDT','DOTUSDT','MAGICUSDT','LTCUSDT','AVAXUSDT','ADAUSDT','LINKUSDT']

def run_trading_bot_task():
    for symbol in coin_pairs:
        print(f"Running bot for {symbol}")

        # Suggest trade for the current coin pair
        trade_suggestion = suggest_trade(symbol)
        
        if trade_suggestion['trend'] == 'no trade signal':
            print(f"No trade signal for {symbol}. Skipping.")
            continue
        
        # Set leverage and get safe trade amount dynamically
        leverage = Decimal('5')  # Fixed leverage (5x)
        usdt_to_trade = Decimal(dynamic_safe_trade_amount(symbol))  # Get only the float amount for the symbol
        
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
        insert_order_if_not_exists(symbol, trade_suggestion['trend'],
                                   entry_price,
                                   Decimal(str(trade_suggestion['suggest_stop_loss'])),
                                   Decimal(str(trade_suggestion['suggest_take_profit'])),
                                   leverage,
                                   quantity,
                                   usdt_to_trade,
                                   effective_position_size)

# Scheduler to run the task periodically
scheduler = BackgroundScheduler()
scheduler.add_job(run_trading_bot_task, IntervalTrigger(seconds=30), max_instances=1)  # Adjust the interval as needed
scheduler.add_job(track_and_update_orders, IntervalTrigger(seconds=30), max_instances=1)  # Track orders every 30 seconds
scheduler.start()


@run_bot_atr_proxy_bp.route('/run')
def start_bot():
    return jsonify({"message": "BOT Started!"})