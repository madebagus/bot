import copy
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
from decimal import Decimal
import pandas_ta as pd_ta

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
def get_historical_data(symbol, interval='3m', limit=1500):
    try:
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

        if df.empty or len(df) < 20:  # Check for minimum data requirement
            raise ValueError("Insufficient data to calculate indicators.")
        
        return df

    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return None

# ++ TREND ANALYSIS SECTION ++

# Define the RSI calculation logic
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

    # Fill NaN values with neutral RSI value (50)
    df['RSI'].fillna(50, inplace=True)

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
    
    # Focus 2: Cross middle
    elif 50 > latest_rsi > 45 and previous_rsi > 50:
        return "SELL"  # RSI cross down 50
    elif 50 < latest_rsi < 55 and previous_rsi < 50:
        return "BUY"  # RSI cross up 50

    # Default to HOLD
    else:
        return "HOLD"

# Define Bollinger Band Logic
def bollinger_bands_decision(df, price_column='close', window=20, num_std_dev=2, buy_threshold=0.2, sell_threshold=0.8):
    """Calculate Bollinger Bands and compare with closing price for trend indication."""
    df = df.copy()

    # Calculate the Bollinger Bands
    sma = df[price_column].rolling(window=window).mean()
    std = df[price_column].rolling(window=window).std()
    upper_band = sma + (std * num_std_dev)
    lower_band = sma - (std * num_std_dev)
    
    latest_close = df[price_column].iloc[-1]
    previous_close = df[price_column].iloc[-2]
    latest_lower_band = lower_band.iloc[-1]
    latest_upper_band = upper_band.iloc[-1]
    
    # Calculate the percentage of the price's position within the Bollinger Bands range
    band_range = latest_upper_band - latest_lower_band
    price_position_percentage = (latest_close - latest_lower_band) / band_range

    #print(f"Latest close: {latest_close}, Upper band: {latest_upper_band}, Lower band: {latest_lower_band}, Band range: {band_range}")

    # Decision based on Bollinger Bands

    if latest_close >= latest_upper_band:  # price touches or above upper band
        return 'SELL'
    elif latest_close <= latest_lower_band:  # price touches or below lower band
        return 'BUY'
    elif latest_close >= sell_threshold:  # price touches or below lower band
        return 'SELL'
    elif latest_close <= buy_threshold:  # price touches or below lower band
        return 'BUY'
    elif latest_close > 50 and previous_close < 30:  # momentum base, when price crosses middle line up
        return 'BUY'
    elif latest_close < 50 and previous_close > 70:  # momentum base, when price crosses middle line down
        return 'SELL'
    
    else:
        # If no strong crossover, band movement, or price positioning detected, return HOLD
        return 'HOLD'
    

# ==============================================================   

# Define cancle stick logic to get Shooter Star and Hammer
def detect_candlestick_patterns(df):
    """
    Detect candlestick patterns and return signals.
    Recognizes Hammer, Shooting Star, Hanging Man, Inverted Hammer, Morning Star, and Evening Star.
    """
    if len(df) < 3:
        return 'HOLD'  # Ensure enough data for multi-candle patterns

    # Latest three candles
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    prev_prev_candle = df.iloc[-3]
    

    # Three White Soldiers : Strong up trend
    if (df['close'].iloc[-3] > df['open'].iloc[-3] and  # 1st candle bullish
        df['close'].iloc[-2] > df['open'].iloc[-2] and  # 2nd candle bullish
        df['close'].iloc[-1] > df['open'].iloc[-1] and  # 3rd candle bullish
        df['close'].iloc[-2] > df['close'].iloc[-3] and  # Higher close
        df['close'].iloc[-1] > df['close'].iloc[-2] and  # Higher close
        df['open'].iloc[-2] > df['close'].iloc[-3] and  # Opens within prev body
        df['open'].iloc[-1] > df['close'].iloc[-2]):    # Opens within prev body
        return 'BUY'

    # Three Black Crows : Strong down trend
    if (df['close'].iloc[-3] < df['open'].iloc[-3] and  # 1st candle bearish
        df['close'].iloc[-2] < df['open'].iloc[-2] and  # 2nd candle bearish
        df['close'].iloc[-1] < df['open'].iloc[-1] and  # 3rd candle bearish
        df['close'].iloc[-2] < df['close'].iloc[-3] and  # Lower close
        df['close'].iloc[-1] < df['close'].iloc[-2] and  # Lower close
        df['open'].iloc[-2] < df['close'].iloc[-3] and  # Opens within prev body
        df['open'].iloc[-1] < df['close'].iloc[-2]):    # Opens within prev body
        return 'SELL'
    
    # Morning Star (Bullish reversal)
    if (
        prev_prev_candle['close'] < prev_prev_candle['open'] and  # First candle bearish
        abs(prev_candle['close'] - prev_candle['open']) < (prev_prev_candle['open'] - prev_prev_candle['close']) * 0.5 and  # Second candle indecisive
        last_candle['close'] > last_candle['open'] and  # Third candle bullish
        last_candle['close'] > prev_prev_candle['open']  # Third candle closes above the first candle's open
    ):
        return 'BUY'

    # Evening Star (Bearish reversal)
    if (
        prev_prev_candle['close'] > prev_prev_candle['open'] and  # First candle bullish
        abs(prev_candle['close'] - prev_candle['open']) < (prev_prev_candle['close'] - prev_prev_candle['open']) * 0.5 and  # Second candle indecisive
        last_candle['close'] < last_candle['open'] and  # Third candle bearish
        last_candle['close'] < prev_prev_candle['open']  # Third candle closes below the first candle's open
    ):
        return 'SELL'

    # Single-candle patterns
    open_price = last_candle['open']
    close_price = last_candle['close']
    high_price = last_candle['high']
    low_price = last_candle['low']

    # Candle body and shadow sizes
    body = abs(close_price - open_price)
    upper_shadow = high_price - max(open_price, close_price)
    lower_shadow = min(open_price, close_price) - low_price

    # Hammer (Bullish reversal)
    if lower_shadow > 2 * body and upper_shadow < body:
        if close_price > open_price:  # Bullish hammer
            return 'BUY'

    # Shooting Star (Bearish reversal)
    if upper_shadow > 2 * body and lower_shadow < body:
        if close_price < open_price:  # Bearish shooting star
            return 'SELL'

    # Hanging Man (Bearish reversal after uptrend)        
    if lower_shadow > 2 * body and upper_shadow < body:
        if close_price < open_price:  # Bearish Hanging Man
            if df['close'].iloc[-2] < df['close'].iloc[-1] and df['close'].iloc[-3] < df['close'].iloc[-2]:
                return 'SELL'

    # Inverted Hammer (Bullish reversal after downtrend)        
    if upper_shadow > 2 * body and lower_shadow < body:
        if close_price < open_price:  # Bearish Hanging Man
            if df['close'].iloc[-2] < df['close'].iloc[-1] and df['close'].iloc[-3] < df['close'].iloc[-2]:
                return 'BUY'

    # No clear pattern detected
    return 'HOLD'

    
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
    if 50 > adx_value > 25:
        print(f"* * * Avoiding trade {symbol} due to strong trend ADX ({adx_value} > 25)")
        return True

    return False

# Function to calculate ATR (Average True Range)
def calculate_atr(df, period=14):
    """Calculate ATR (Average True Range) for volatility measurement."""
    atr = pd_ta.atr(df['high'], df['low'], df['close'], length=period)
    return atr.iloc[-1]  # Return the most recent ATR value

# ++ DECISSION SECTION ++

# Calculate trend result to make decission. 
def get_trading_decision(symbol):
    """Combine RSI, Bollinger Band, and candlestick decisions for a final trend decision."""
    df = get_historical_data(symbol)
    if df is None or len(df) < 20:  # Ensure sufficient data
        print(f"Insufficient data for {symbol}")
        return 'HOLD'

    # Get decisions from individual indicators
    rsi_dec = rsi_decision(df)
    bollinger_dec = bollinger_bands_decision(df)
    candle_pattern = detect_candlestick_patterns(df)
    adx_value = adx_indicator(df)  # Get ADX to evaluate trend strength

    # Check avoid conditions
    #if avoid_conditions(symbol,df):
    #    return 'HOLD'  # Avoid entering any trades if conditions are met

    # Determine the suggested trend
    trend = 'HOLD'

    
        # Assign scores for RSI and Bollinger Bands
    candle_pattern_score = {'BUY': 1, 'SELL': -1, 'HOLD': 0}.get(candle_pattern, 0)
    rsi_score = {'BUY': 1, 'SELL': -1, 'HOLD': 0}.get(rsi_dec, 0)
    bollinger_score = {'BUY': 1, 'SELL': -1, 'HOLD': 0}.get(bollinger_dec, 0)

    
    rsi_weight = 0.4
    bollinger_weight = 0.3
    candle_weight = 0.3

    total_score = (rsi_score * rsi_weight) + (bollinger_score * bollinger_weight) + (candle_pattern_score * candle_weight)

        # Determine trend based on score thresholds
    if total_score > 0.65:
            trend = 'BUY'
    elif total_score < -0.65:
            trend = 'SELL'
    else:
            trend = 'HOLD'

    # Log detailed decision components
    print(
        f"Trend decision for {symbol}: "
        f"RSI={rsi_dec}, Bollinger={bollinger_dec}, Candle Pattern={candle_pattern}, "
        f"ADX={adx_value:.2f}, Trend={trend}"
    )
    return trend

import pandas_ta as pd_ta

# Function to calculate ATR (Average True Range)
def calculate_atr(df, period=14):
    """Calculate ATR (Average True Range) for volatility measurement."""
    atr = pd_ta.atr(df['high'], df['low'], df['close'], length=period)
    return atr.iloc[-1]  # Return the most recent ATR value

# calculate bolinge movement
def calculate_bollinger_bands(df, window=20):
    """Calculate Bollinger Bands (upper, middle, lower)."""
    middle_band = df['close'].rolling(window=window).mean()
    rolling_std = df['close'].rolling(window=window).std()

    upper_band = middle_band + (rolling_std * 2)  # 2 standard deviations above
    lower_band = middle_band - (rolling_std * 2)  # 2 standard deviations below

    return upper_band, middle_band, lower_band

# ++ SUGGESTION SECTION ++

# Get Trade Suggestion based on Trading Decision
def suggest_trade(symbol, fixed_stop_loss=0.01, fixed_take_profit=0.015):
    """
    Calculate trade suggestion including trend, entry price, stop loss, and take profit.
    Handles NaN values in the DataFrame.
    """
    # Fetch historical data
    df = get_historical_data(symbol)
    if df is None or len(df) < 20:
        print(f"Insufficient data for {symbol}")
        return {'trend': 'HOLD'}  # Return 'HOLD' by default

    # Initialize trend variable
    trend = 'HOLD'  # Default value
    # Clean NaN values
    df = df.dropna()  # Drop rows with any NaN values

    # Entry price is the latest close price
    entry_price = df['close'].iloc[-1]

    try:
        # Calculate trend
        trend = get_trading_decision(symbol)

        # Calculate Bollinger Bands
        bollinger_window = 20  # Bollinger Bands window (default: 20)
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df, bollinger_window)

        # Get the direction of the Bollinger Bands
        bb_slope = df['bb_upper'].iloc[-1] - df['bb_upper'].iloc[-2]  # Upper band slope
        bb_direction = "up" if bb_slope > 0 else "down"  # Bollinger Bands direction

        # Conditional 1: If final trend is BUY and Bollinger is moving down, HOLD
        #if trend == 'BUY' and bb_direction == 'down':
        #    print("HOLD due to Bollinger Bands moving down despite BUY signal.")
        #    trend = 'HOLD'

        # Conditional 2: If final trend is SELL and Bollinger is moving up, HOLD
        #if trend == 'SELL' and bb_direction == 'up':
        #    print("HOLD due to Bollinger Bands moving up despite SELL signal.")
        #    trend = 'HOLD'

        # Calculate ATR for volatility-based stop loss and take profit
        atr_value = calculate_atr(df, period=14)

        # Calculate dynamic stop loss and take profit based on ATR
        if trend == 'BUY':
            stop_loss = entry_price - (2 * atr_value)  # 2x ATR for stop loss
            take_profit = entry_price + (2 * atr_value)  # 2x ATR for take profit
        elif trend == 'SELL':
            stop_loss = entry_price + (2 * atr_value)  # 2x ATR for stop loss
            take_profit = entry_price - (2 * atr_value)  # 2x ATR for take profit
        else:
            stop_loss = entry_price
            take_profit = entry_price

        # If the ATR-based stop loss or take profit is too close, fall back to fixed values
        if abs(entry_price - stop_loss) < (entry_price * fixed_stop_loss):
            stop_loss = entry_price - (entry_price * fixed_stop_loss)
        if abs(entry_price - take_profit) < (entry_price * fixed_take_profit):
            take_profit = entry_price + (entry_price * fixed_take_profit)

        return {
            'trend': trend,
            'suggest_entry_price': entry_price,
            'suggest_stop_loss': stop_loss,
            'suggest_take_profit': take_profit,
        }

    except Exception as e:
        print(f"Error in suggest_trade: {e}")
        return {
            'trend': trend,
            'suggest_entry_price': entry_price,
            'suggest_stop_loss': entry_price,
            'suggest_take_profit': entry_price,
        }


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


# +++ Placing future Order without stop loss and take profit +++
def place_futures_order_xman(symbol, trend, entry_price, stop_loss, take_profit, leverage, quantity, usdt_to_trade, position_size):
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


    except Exception as e:
        print(f"Error placing order: {e}")

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
            
# ++ MAIN SECTION ++

# Calling the whole function to start trading process
import threading

# Initialize the list of coin pairs with a default value
coin_pairs = ['DOTUSDT','ATOMUSDT', 'LTCUSDT','ETHUSDT']  # Example with 4 pairs
# Flag to control whether the fetch_recent_orders task should be canceled
cancel_fetch_orders = False
# Function to handle trading for each symbol
def run_symbol_task(symbol):
    print(f"+ + + Running bot for == {symbol} ==")
    
    usdt_to_trade = Decimal('2')  # Example trade amount
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
        print(f"Trade Size: ",usdt_to_trade)
        print(f"Leverage: {leverage}")
        print(f"Entry Price: {entry_price}")
        print(f"Effective Position Size: {effective_position_size}")
        print(f"Quantity: {quantity}")

        # Place order
        place_futures_order_xman(symbol, trade_signal, entry_price,
                            Decimal(str(trade_suggestion['suggest_stop_loss'])),
                            Decimal(str(trade_suggestion['suggest_take_profit'])),
                            leverage, quantity, usdt_to_trade, effective_position_size)
    

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



