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

def get_historical_data(symbol, interval, lookback="12 hour ago UTC"):
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


# Define the RSI calculation logic
@bot_bp.route('/rsi',methods=['GET'])

def rsi_decision(symbol, price_column='close'):
    """Fetch historical data, calculate RSI, and check trading conditions."""
    df = get_historical_data(symbol, interval='5m', lookback="12 hour ago UTC")
    #print('Data Frame RSI :', df)

    # Check for NaN values in the 'close' column
    if df[price_column].isnull().sum() > 0:
        print(f"Data contains NaN values in the {price_column} column for {symbol}. Returning HOLD.")
        return 'HOLD'
    
    # Ensure enough data for RSI calculation
    if len(df) < 14:
        print(f"Insufficient data for {symbol}. Need at least 14 data points for RSI calculation.")
        return 'HOLD'
    
    # Calculate RSI using pandas-ta
    df['RSI'] = pd_ta.rsi(df[price_column], length=14)
    #print('Data Frame df[RSI] :', df)

    # Handle NaN values in RSI column by replacing them with the mean of surrounding values
    df.dropna(subset=['RSI'], inplace=True)

    # Check if RSI contains NaN values
    if df['RSI'].isnull().sum() > 0:
        print(f"RSI calculation failed for {symbol}. Returning HOLD.")
        return 'HOLD'

    # Get the latest RSI values
    latest_rsi = df['RSI'].iloc[-1]
    previous_rsi = df['RSI'].iloc[-2]
    previous_two_rsi = df['RSI'].iloc[-3]

    #print(f"Symbol: {symbol}, Latest RSI: {latest_rsi}, Previous RSI: {previous_rsi}")

    # Focus 1: Overbought and Oversold Conditions
    if latest_rsi > 70:
        return "STRONG SELL"  # Overbought condition
    elif latest_rsi < 30:
        return "STRONG BUY"  # Oversold condition

    # Focus 2: Fast Momentum (Sharp Changes)
    rsi_change = latest_rsi - previous_rsi
    momentum_threshold = 10  # Adjustable based on market behavior

    if rsi_change > momentum_threshold:
        return "STRONG BUY"  # Strong upward momentum
    elif rsi_change < -momentum_threshold:
        return "STRONG SELL"  # Strong downward momentum

    # Focus 3: Cross Midline (50) Reversal
    if previous_rsi > 50 and latest_rsi < 50:
        return "SELL"  # Bearish reversal
    elif previous_rsi < 50 and latest_rsi > 50:
        return "BUY"  # Bullish reversal
    
    # Focus 4: sudent refersal from overbough and oversold
    if 70 > latest_rsi > 55 and previous_rsi > 70:
        return "SELL"
    elif 30 < latest_rsi < 45 and previous_rsi < 30:
        return "BUY"
    
    # Default to HOLD
    return "HOLD"


@bot_bp.route('/stoc_rsi',methods=['GET'])
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
    previous_stoch_rsi_k = df['stoch_rsi_k'].iloc[-2]
    latest_stoch_rsi_d = df['stoch_rsi_d'].iloc[-1]
    previous_stoch_rsi_d = df['stoch_rsi_d'].iloc[-2]

    latest_delta = abs(latest_stoch_rsi_k - latest_stoch_rsi_d)
    previous_delta = abs(previous_stoch_rsi_k - previous_stoch_rsi_d)

    # **1. Overbought and Oversold Conditions**
    if latest_stoch_rsi_k > 80 and latest_stoch_rsi_d > 80:
        return "STRONG SELL"
    elif latest_stoch_rsi_k < 20 and latest_stoch_rsi_d < 20:
        return "STRONG BUY"


    # **2. %K/%D Crossovers**
    if latest_stoch_rsi_k > latest_stoch_rsi_d and latest_stoch_rsi_k < 40:
        return "BUY"
    elif latest_stoch_rsi_k < latest_stoch_rsi_d and latest_stoch_rsi_k > 60:
        return "SELL"

   # **3. Momentum-Based Condition above of below middle line
    if latest_delta > previous_delta and latest_stoch_rsi_k < 50:
        return "BUY"
    elif latest_delta > previous_delta and latest_stoch_rsi_k > 50:
        return "SELL"

    # **4. Sudden Reversal from Overbought/Oversold**
    if 80 > previous_stoch_rsi_k > 60 and previous_stoch_rsi_k > 80:
        return "SELL"
    elif 20 < latest_stoch_rsi_k < 40 and previous_stoch_rsi_k < 20:
        return "BUY"
    # Default to HOLD
    return "HOLD"


# Define the EMA calculation logic
def calculate_ema(df, short_period=7, long_period=14):
    """Calculate two EMAs with different periods."""
    df['EMA_short'] = pd_ta.ema(df['close'], length=short_period)
    df['EMA_long'] = pd_ta.ema(df['close'], length=long_period)
    return df

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


def get_ema_decision(symbol, short_ema_period, long_ema_period):
    # Fetch historical data
    df = get_historical_data(symbol, interval='5m', lookback="12 hour ago UTC")
    
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
        return None


    # Get the latest and previous prices
    latest_price = df['close'].iloc[-1]
    previous_price = df['close'].iloc[-2]

    # Calculate the EMA deltas
    ema_delta = latest_ema_short - latest_ema_long
    previous_ema_delta = previous_ema_short - previous_ema_long

    # Standardize the delta as a percentage of the price
    relative_delta = (ema_delta / latest_price) * 100
    previous_relative_delta = (previous_ema_delta / previous_price) * 100

    print(f"Symbol: {symbol}, Latest EMA Delta: {relative_delta:.4f}%, Previous EMA Delta: {previous_relative_delta:.4f}%")

    # Define thresholds
    strong_buy_threshold = 0.1  # Example thresholds
    strong_sell_threshold = -0.1

    # Detect EMA crossovers
    if previous_ema_short < previous_ema_long and latest_ema_short > latest_ema_long:
        return 'BUY'  # Red EMA crossed above Blue EMA
    elif previous_ema_short > previous_ema_long and latest_ema_short < latest_ema_long:
        return 'SELL'  # Red EMA crossed below Blue EMA

    # Detect reversals
    if previous_relative_delta > 0 and relative_delta < 0:
        return 'SELL'  # Downward reversal
    elif previous_relative_delta < 0 and relative_delta > 0:
        return 'BUY'  # Upward reversal

    # Detect strong trends
    if relative_delta > strong_buy_threshold and relative_delta > previous_relative_delta:
        return 'BUY'  # Current delta is large and increasing
    elif relative_delta < strong_sell_threshold and relative_delta < previous_relative_delta:
        return 'SELL'  # Current delta is large and decreasing
    
    # Focus on the price position relative to the EMAs
    if latest_ema_short > latest_ema_long:
        # Condition 1: Short EMA > Long EMA, but price closes below Short EMA, trigger BUY
        if latest_price < latest_ema_short and latest_price < latest_ema_long:
            return "BUY"
    elif latest_ema_short < latest_ema_long:
        # Condition 2: Short EMA < Long EMA, but price closes above Short EMA, trigger BUY
        if latest_price > latest_ema_short and latest_price > latest_ema_long:
            return "BUY"
    
    # Check for delta reduction (50% of previous delta)
    if previous_ema_short > previous_ema_long and ema_delta < previous_ema_delta * 0.75:
        return "SELL"  # SELL condition: Short EMA is above Long EMA, but delta is reducing by 50%
    elif previous_ema_short < previous_ema_long and ema_delta > previous_ema_delta * 0.75:
        return "BUY"  # BUY condition: Short EMA is below Long EMA, but delta is increasing by 50%


    # Default action
    return 'HOLD'

def analyze_ema_and_triangle(symbol, short_ema_period=7, long_ema_period=14):
    # Fetch historical data
    df = get_historical_data(symbol, interval='5m', lookback="12 hour ago UTC")

    # Calculate EMA
    df = calculate_ema(df, short_ema_period, long_ema_period)

    # Drop NaN values
    df = df.dropna(subset=['EMA_short', 'EMA_long'])

    # Ensure there is enough data
    if len(df) < 10:
        print(f"Not enough data for {symbol}.")
        return 'HOLD'

    # Detect EMA signals
    ema_signal = get_ema_decision(symbol, short_ema_period, long_ema_period)

    # If no EMA signal, return HOLD
    if ema_signal == 'HOLD':
        return 'HOLD'

    # Detect Triangle Pattern
    upper_bound, lower_bound = detect_triangle_pattern(df)
    if not upper_bound or not lower_bound:
        return ema_signal  # Default to EMA signal if no triangle detected

    # Check for triangle breakout
    last_close = df['close'].iloc[-1]
    prev_closes = df['close'].iloc[-3:-1]

    if ema_signal == 'BUY' and last_close > upper_bound and all(prev_closes > upper_bound):
        print(f"{symbol}: Triangle breakout confirmed. BUY.")
        return 'BUY'
    elif ema_signal == 'SELL' and last_close < lower_bound and all(prev_closes < lower_bound):
        print(f"{symbol}: Triangle breakout confirmed. SELL.")
        return 'SELL'

    return 'HOLD'

#++++++++++++analysi candle trend

def analyze_candle_trend(df):
    # Get the last and previous highs and lows
    last_high = df['high'].iloc[-1]
    last_low = df['low'].iloc[-1]
    
    prev_high_1 = df['high'].iloc[-2]
    prev_high_2 = df['high'].iloc[-3]
    
    prev_low_1 = df['low'].iloc[-2]
    prev_low_2 = df['low'].iloc[-3]


    # Initialize trend as HOLD
    trend = 'HOLD'

    # Detect triangle pattern in the last 10 candles
    upper_bound, lower_bound = detect_triangle_pattern(df)

    # Identify if the current candle makes a higher high (potential BUY)
    if last_high > prev_high_1 and last_high > prev_high_2:
        # Check if the price breaks out of the upper triangle boundary
        if upper_bound and last_high > upper_bound:
            print("Triangle breakout above detected!")
            trend = 'BUY'
        else:
            trend = 'BUY'

    # Identify if the current candle makes a lower low (potential SELL)
    elif last_low < prev_low_1 and last_low < prev_low_2:
        # Check if the price breaks below the lower triangle boundary
        if lower_bound and last_low < lower_bound:
            print("Triangle breakout below detected!")
            trend = 'SELL'
        else:
            trend = 'SELL'

    return trend


# Define the ATR calculation logic
def calculate_atr(df, period=14):
    """Calculate the Average True Range (ATR)."""
    df['ATR'] = pd_ta.atr(df['high'], df['low'], df['close'], length=period)
    return df

def get_atr(symbol, atr_period):
    # Fetch historical data
    df = get_historical_data(symbol, interval='5m', lookback="12 hour ago UTC")
    
    # Calculate ATR
    df = calculate_atr(df, atr_period)
    
    atr_value = df['ATR'].iloc[-1]
    return atr_value

# Define the strategy logic with EMA, RSI, and ATR
def suggest_trade(symbol,  atr_period=14):
    """Calculate EMAs, RSI, and ATR and determine trading signals."""
    
    # Fetch historical data (this part is assumed to be handled by another function)
    df = get_historical_data(symbol, interval='5m', lookback="12 hour ago UTC")
    
    # Initialize the trade signal
    trend = 'HOLD'
    entry_price = df['close'].iloc[-1]  # Entry price is the latest close price

    atr_value = get_atr(symbol,atr_period)
    rsi_dec = rsi_decision(symbol)
    ema_dec = analyze_ema_and_triangle (symbol)
    candle_dec = analyze_candle_trend(df)
    stoc_rsi_dec = stoch_rsi_decision(df)

    print(f"Symbol: {symbol}, RSI Trend: {rsi_dec}, Stoc RSI Trend: {stoc_rsi_dec}, EMA Trend: {ema_dec}, Candle Trend : {candle_dec}")
    #print(f"Symbol: {symbol}, RSI Trend: {rsi_dec}, EMA Trend: {ema_dec}, Candle Trend : {candle_dec}, atr_value : {atr_value}")


    #define RSI trend score 
    if rsi_dec == 'BUY':
        rsi_score = 1
    elif rsi_dec == 'STRONG BUY':
        rsi_score = 1.5
    elif rsi_dec == 'SELL':
        rsi_score = -1
    elif rsi_dec == 'STRONG SELL':
        rsi_score = -1.5
    else: 
        rsi_score = 0

    #define RSI trend score     
    if stoc_rsi_dec == 'BUY':
        stoc_rsi_score = 1
    elif stoc_rsi_dec == 'STRONG BUY':
        stoc_rsi_score = 1.5
    elif rsi_dec == 'SELL':
        stoc_rsi_score = -1
    elif stoc_rsi_dec == 'STRONG SELL':
        stoc_rsi_score = -1.5
    else: 
        stoc_rsi_score = 0

    #define EMA trend score 
    if ema_dec == 'BUY':
        ema_score = 1
    elif ema_dec == 'STRONG BUY':
        ema_score = 1.5
    elif ema_dec == 'SELL':
        ema_score = -1
    elif ema_dec == 'STRONG SELL':
        ema_score = -1.5
    else:
        ema_score = 0
    
    #define candle trend score 
    if candle_dec == 'BUY':
        candle_score = 1
    elif candle_dec == 'STRONG BUY':
        candle_score = 1.5
    elif candle_dec == 'SELL':
        candle_score = -1
    elif ema_dec == 'STRONG SELL':
        candle_score = -1.5
    else:
        candle_score = 0

    total_score = ema_score + stoc_rsi_score + candle_score
    
    #define the trend base on score calculation
    if total_score >= 2:
        trend = 'BUY'
    elif total_score <= -2:
        trend = 'SELL'
    else:
        trend = 'HOLD'
   
    # ATR-based Risk Management (Volatility-based)
    if trend == 'BUY':
        stop_loss = entry_price - (1 * atr_value)  # 1.5x ATR below entry
        take_profit = entry_price + (2 * atr_value)  # 2x ATR above entry
        final_trend = 'BUY'
    elif trend == 'SELL':
        stop_loss = entry_price + (1 * atr_value)  # 1.5x ATR above entry
        take_profit = entry_price - (2 * atr_value)  # 2x ATR below entry
        final_trend = 'SELL'
    else:
        stop_loss = entry_price
        take_profit = entry_price
        final_trend = 'HOLD'
    
    print(f"Symbol: {symbol}, total_score: {total_score}, Final Trend: {final_trend}")
    
    if stop_loss == take_profit:
            return {
                'trend': 'HOLD',
                'suggest_entry_price': entry_price,
                'suggest_stop_loss': stop_loss,
                'suggest_take_profit': take_profit
            }
    else:
            return {
                'trend': final_trend,
                'suggest_entry_price': entry_price,
                'suggest_stop_loss': stop_loss,
                'suggest_take_profit': take_profit
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
coin_pairs = ['ETHUSDT','AVAXUSDT','BNBUSDT','LTCUSDT','DOTUSDT']  # Example with 4 pairs
# Flag to control whether the fetch_recent_orders task should be canceled
cancel_fetch_orders = False
# Function to handle trading for each symbol
def run_symbol_task(symbol):
    print(f"+ + + Running bot for {symbol} + + +")
    
    usdt_to_trade = Decimal('5')  # Example trade amount
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
        scheduler.add_job(run_trading_bot_task, IntervalTrigger(seconds=30), max_instances=1)  # Adjust the interval as needed
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
