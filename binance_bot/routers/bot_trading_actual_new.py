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


# Initialize the blueprint for the bot
run_bot_real_bp = Blueprint('run_bot_atr_new', __name__, url_prefix='/api/bot/')

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

import pandas_ta as pd_ta

def rsi_decision(symbol, price_column='close'):
    """Fetch historical data, calculate RSI, and check trading conditions."""
    df = get_historical_data(symbol, interval='1m', lookback="12 hour ago UTC")
    
    # Calculate RSI using pandas-ta
    df['RSI'] = pd_ta.rsi(df['close'], length=14)
    
    # Add a midline (50) to help evaluate crossovers
    midline = 50
    lower_limit = midline * 0.95  # 5% below the midline (47.5)
    upper_limit = midline * 1.05  # 5% above the midline (52.5)
    
    # Get the latest RSI value and previous ones
    latest_rsi = df['RSI'].iloc[-1]
    previous_rsi = df['RSI'].iloc[-2]  # Previous RSI
    two_previous_rsi = df['RSI'].iloc[-3]  # RSI value two periods ago

    # Default trend is HOLD
    trend = 'HOLD'

    # Check if RSI is 5% below the midline and previously failed to break above midline (SELL)
    if latest_rsi < lower_limit and previous_rsi < lower_limit and two_previous_rsi < lower_limit:
        print(f"SELL condition met for {symbol}: RSI is below 47.5 and failed to break out previously.")
        # Trigger a SELL signal
        trend = 'SELL'
    
    # Check if RSI is 5% above the midline and previously failed to break below midline (BUY)
    elif latest_rsi > upper_limit and previous_rsi > upper_limit and two_previous_rsi > upper_limit:
        print(f"BUY condition met for {symbol}: RSI is above 52.5 and failed to break down previously.")
        # Trigger a BUY signal
        trend = 'BUY'
    
    else: 
        # Calculate RSI for 5, 7, and 14 periods
        rsi_5 = ta.momentum.RSIIndicator(df[price_column], window=5).rsi().iloc[-1]
        rsi_7 = ta.momentum.RSIIndicator(df[price_column], window=7).rsi().iloc[-1]
        rsi_14 = ta.momentum.RSIIndicator(df[price_column], window=14).rsi().iloc[-1]

        # Set thresholds
        buy_threshold = 40
        sell_threshold = 70

        # Count buy and sell signals
        buy_signals = sum(rsi < buy_threshold for rsi in [rsi_5, rsi_7, rsi_14])
        sell_signals = sum(rsi > sell_threshold for rsi in [rsi_5, rsi_7, rsi_14])
        
        # Decision logic: if 2 or more indicators signal BUY or SELL
        if buy_signals >= 2:
            trend = 'BUY'
        elif sell_signals >= 2:
            trend = 'SELL'
        else:
            trend = 'SELL'
    
    return trend


# Define the EMA decision logic
def ema_decision(df, price_column='close'):
    # Calculate EMA for 5 and 15 periods
    ema_5 = ta.trend.EMAIndicator(df[price_column], window=5).ema_indicator().iloc[-1]
    ema_15 = ta.trend.EMAIndicator(df[price_column], window=15).ema_indicator().iloc[-1]
    
    # Decision logic for EMA
    if ema_5 > ema_15:
        return 'BUY'
    elif ema_5 < ema_15:
        return 'SELL'
    else:
        return 'HOLD'

def get_ema_trend_decision(symbol):
    
    # Fetch historical data for each interval
    df_1m = get_historical_data(symbol, interval='1m')
    df_3m = get_historical_data(symbol, interval='3m')
    df_5m = get_historical_data(symbol, interval='5m')
    
    # Get EMA decisions for each interval
    ema_dec_1m = ema_decision(df_1m)
    ema_dec_3m = ema_decision(df_3m)
    ema_dec_5m = ema_decision(df_5m)
    
    # Count BUY and SELL signals
    buy_signals = sum(dec == 'BUY' for dec in [ema_dec_1m, ema_dec_3m, ema_dec_5m])
    sell_signals = sum(dec == 'SELL' for dec in [ema_dec_1m, ema_dec_3m, ema_dec_5m])
    
    # Final decision: at least 2 signals for BUY or SELL
    if buy_signals >= 2:
        return 'BUY'
    elif sell_signals >= 2:
        return 'SELL'
    else:
        return 'HOLD'
    

# Define the VWAP decision logic
def vwap_decision(df):
    """Calculate VWAP and compare with closing price for trend indication."""
    # Calculate VWAP: Cumulative sum of price * volume, divided by cumulative volume
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    latest_close = df['close'].iloc[-1]
    latest_vwap = vwap.iloc[-1]
    
    # Decision based on VWAP comparison with close price
    if latest_close > latest_vwap:
        return 'BUY'
    elif latest_close < latest_vwap:
        return 'SELL'
    else:
        return 'HOLD'

def get_vwap_trend_decision(symbol):
    """Get VWAP-based trend decision across multiple intervals."""
    # Fetch historical data for each interval
    df_1m = get_historical_data(symbol, interval='1m')
    df_3m = get_historical_data(symbol, interval='3m')
    df_5m = get_historical_data(symbol, interval='5m')
    
    # Get VWAP decisions for each interval
    vwap_dec_1m = vwap_decision(df_1m)
    vwap_dec_3m = vwap_decision(df_3m)
    vwap_dec_5m = vwap_decision(df_5m)
    
    # Count BUY and SELL signals
    buy_signals = sum(dec == 'BUY' for dec in [vwap_dec_1m, vwap_dec_3m, vwap_dec_5m])
    sell_signals = sum(dec == 'SELL' for dec in [vwap_dec_1m, vwap_dec_3m, vwap_dec_5m])
    
    # Final decision: at least 2 signals for BUY or SELL
    if buy_signals >= 2:
        return 'BUY'
    elif sell_signals >= 2:
        return 'SELL'
    else:
        return 'HOLD'

# Define the BOLLINGER BAND decision logic    
def bollinger_bands_decision(df, price_column='close', window=20, num_std_dev=2, momentum_threshold=1.5, band_slope_threshold=0.05, buy_threshold=0.2, sell_threshold=0.8):
    """Calculate Bollinger Bands and compare with closing price for trend indication, considering fast crossovers, band direction, and relative price position."""
    
    # Calculate the Bollinger Bands
    sma = df[price_column].rolling(window=window).mean()
    std = df[price_column].rolling(window=window).std()
    upper_band = sma + (std * num_std_dev)
    lower_band = sma - (std * num_std_dev)
    
    latest_close = df[price_column].iloc[-1]
    latest_lower_band = lower_band.iloc[-1]
    latest_upper_band = upper_band.iloc[-1]
    
    # Calculate the price change over the last two periods
    price_change = (df[price_column].iloc[-1] - df[price_column].iloc[-2]) / df[price_column].iloc[-2]
    
    # Calculate the slope of the bands (rate of change)
    upper_band_slope = (upper_band.iloc[-1] - upper_band.iloc[-2]) / upper_band.iloc[-2]
    lower_band_slope = (lower_band.iloc[-1] - lower_band.iloc[-2]) / lower_band.iloc[-2]
    
    # Calculate the percentage of the price's position within the Bollinger Bands range
    band_range = latest_upper_band - latest_lower_band
    price_position_percentage = (latest_close - latest_lower_band) / band_range
    
    # Decision based on Bollinger Bands with momentum, band direction, and price position
    if price_position_percentage > sell_threshold:
        # Price is above 80% of the band range, SELL
        return 'SELL'
    elif price_position_percentage < buy_threshold:
        # Price is below 20% of the band range, BUY
        return 'BUY'
    
    if latest_close < latest_lower_band and price_change > momentum_threshold:
        # Fast upward crossing after being below lower band (Buy signal)
        return 'BUY'
    elif latest_close > latest_upper_band and price_change < -momentum_threshold:
        # Fast downward crossing after being above upper band (Sell signal)
        return 'SELL'
    
    # Incorporate the movement of the bands themselves
    elif upper_band_slope > band_slope_threshold and latest_close > latest_upper_band:
        # If the upper band is moving up rapidly (strong price move) and price is above upper band, SELL
        return 'SELL'
    elif lower_band_slope < -band_slope_threshold and latest_close < latest_lower_band:
        # If the lower band is moving down rapidly (strong price move) and price is below lower band, BUY
        return 'BUY'
    
    else:
        # If no strong crossover, band movement, or price positioning detected, return HOLD
        return 'HOLD'

def get_bollinger_bands_trend_decision(symbol, momentum_threshold=1.5, band_slope_threshold=0.05, buy_threshold=0.2, sell_threshold=0.8):
    """Get Bollinger Bands-based trend decision across multiple intervals with fast crossovers, band movement, and price position."""
    
    # Fetch historical data for each interval
    df_1m = get_historical_data(symbol, interval='1m')
    df_3m = get_historical_data(symbol, interval='3m')
    df_5m = get_historical_data(symbol, interval='5m')
    
    # Get Bollinger Bands decisions for each interval
    bollinger_dec_1m = bollinger_bands_decision(df_1m, momentum_threshold=momentum_threshold, band_slope_threshold=band_slope_threshold, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    bollinger_dec_3m = bollinger_bands_decision(df_3m, momentum_threshold=momentum_threshold, band_slope_threshold=band_slope_threshold, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    bollinger_dec_5m = bollinger_bands_decision(df_5m, momentum_threshold=momentum_threshold, band_slope_threshold=band_slope_threshold, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    
    # Count BUY and SELL signals
    buy_signals = sum(dec == 'BUY' for dec in [bollinger_dec_1m, bollinger_dec_3m, bollinger_dec_5m])
    sell_signals = sum(dec == 'SELL' for dec in [bollinger_dec_1m, bollinger_dec_3m, bollinger_dec_5m])
    
    # Final decision: at least 2 signals for BUY or SELL
    if buy_signals >= 2:
        return 'BUY'
    elif sell_signals >= 2:
        return 'SELL'
    else:
        return 'HOLD'

# Define the MACD calculation logic with shortened parameters
def refined_macd_decision(df, price_column='close', window_short=12, window_long=26, signal_window=9):
    """Enhanced MACD decision-making with Histogram confirmation and divergence analysis."""
    
    # Calculate MACD and Signal line using the standard formula
    macd = df[price_column].ewm(span=window_short, adjust=False).mean() - df[price_column].ewm(span=window_long, adjust=False).mean()
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    
    # Calculate MACD Histogram
    macd_histogram = macd - signal_line
    
    # Latest values
    latest_macd = macd.iloc[-1]
    latest_signal = signal_line.iloc[-1]
    latest_histogram = macd_histogram.iloc[-1]
    
    # Bullish Crossover and Histogram Expansion
    if latest_macd > latest_signal and latest_histogram > 0:
        # Confirm that the price is also above the EMA to add precision
        if df[price_column].iloc[-1] > df[price_column].ewm(span=50, adjust=False).mean().iloc[-1]:  # Check if price is above 50-period EMA
            return 'BUY'
    
    # Bearish Crossover and Histogram Expansion
    elif latest_macd < latest_signal and latest_histogram < 0:
        # Confirm that the price is below the EMA to add precision
        if df[price_column].iloc[-1] < df[price_column].ewm(span=50, adjust=False).mean().iloc[-1]:  # Check if price is below 50-period EMA
            return 'SELL'
    
    # If no clear trend, hold
    return 'HOLD'

# Refined decision logic for multiple timeframes
def get_macd_trend_decision(symbol):
    """Get MACD-based trend decision across multiple intervals."""
    
    # Fetch historical data for multiple intervals
    df_1m = get_historical_data(symbol, interval='1m')
    df_5m = get_historical_data(symbol, interval='5m')
    df_15m = get_historical_data(symbol, interval='15m')
    
    # Get MACD decisions for each interval
    macd_decision_1m = refined_macd_decision(df_1m)
    macd_decision_5m = refined_macd_decision(df_5m)
    macd_decision_15m = refined_macd_decision(df_15m)
    
    # Count BUY and SELL signals
    buy_signals = sum(dec == 'BUY' for dec in [macd_decision_1m, macd_decision_5m, macd_decision_15m])
    sell_signals = sum(dec == 'SELL' for dec in [macd_decision_1m, macd_decision_5m, macd_decision_15m])
    
    # Final decision: at least 2 signals for BUY or SELL
    if buy_signals >= 2:
        return 'BUY'
    elif sell_signals >= 2:
        return 'SELL'
    else:
        return 'HOLD'


# Combine all trend decisions for the final trading decision
def get_trading_decision(symbol):
    
    # Fetch historical data
    df = get_historical_data(symbol, '1h')
    
    # Get RSI and EMA decisions
    rsi_dec = rsi_decision(symbol)
    if rsi_dec == 'BUY':
        rsi_score = 1
    elif rsi_dec == 'SELL':
        rsi_score = -1
    else:
        rsi_score = 0

    ema_dec = get_ema_trend_decision(symbol)
    if ema_dec == 'BUY':
        ema_score = 1
    elif ema_dec == 'SELL':
        ema_score = -1
    else:
        ema_score = 0

    vwap_dec = get_vwap_trend_decision(symbol)
    if vwap_dec == 'BUY':
        vwap_score = 1
    elif vwap_dec == 'SELL':
        vwap_score = -1
    else:
        vwap_score = 0

    bollinger_dec = get_bollinger_bands_trend_decision(symbol)
    if bollinger_dec == 'BUY':
        bollinger_score = 1
    elif bollinger_dec == 'SELL':
        bollinger_score = -1
    else:
        bollinger_score = 0

    macd_dec = get_macd_trend_decision(symbol)
    if macd_dec == 'BUY':
        macd_score = 1
    elif macd_dec == 'SELL':
        macd_score = -1
    else:
        macd_score = 0

    # Weighted scoring (adjusted for HFT strategy)
    rsi_weight = 0.15       # Lower weight for RSI as it's slower to react
    macd_weight = 0.3     # Higher weight for MACD due to its relevance in momentum
    bollinger_weight = 0.2 # Medium weight for Bollinger Bands to capture volatility
    vwap_weight = 0.25   # High weight for vwap as it confirms the strength of moves
    ema_weight = 0.1        # Lower weight for EMA in HFT to reduce lag

    # Calculate total score by applying weights
    total_score = (rsi_score * rsi_weight) + (macd_score * macd_weight) + \
                 (bollinger_score * bollinger_weight) + (vwap_score * vwap_weight) + (ema_score * ema_weight) 


    # Combine decisions: only buy if both are buy, and sell if both are sell
    if total_score > 0.5:
        trend = 'BUY'
    elif total_score < -0.5:
        trend = 'SELL'
    else:
        trend = 'HOLD'

    print ('==========================')
    print ('RSI Trend :',rsi_dec)
    print ('EMA Trend :',ema_dec)
    print ('VWAP trend :',vwap_dec)
    print ('Bollinger Trend :',bollinger_dec)
    print ('MACD Trend :',macd_dec)
    print ('==========================')

    return trend

#calculate price based on volatility 

def atr_decision(df, symbol):
    """Calculate ATR, stop-loss, take-profit, and current price, adjusted for trend direction and fees.
    
    Parameters:
    df (DataFrame): DataFrame containing price data (high, low, close).
    symbol (str): The symbol for the asset (e.g., 'BTCUSDT').
    window (int): The window size for ATR calculation (default 14).
    atr_multiplier_stop (float): Multiplier for stop-loss based on ATR.
    atr_multiplier_take_profit (float): Multiplier for take-profit based on ATR.
    fee (float): Trading fee as a fraction (e.g., 0.001 for 0.1%).
    
    Returns:
    tuple: The latest price, adjusted stop-loss, and take-profit considering the trend and fees.
    """
    
    # Get the trend direction (BUY or SELL) using the trading decision
    trend = get_trading_decision(symbol)
    
    latest_close = df['close'].iloc[-1]

    # Adjust stop-loss and take-profit based on the trend and fee
    
    if trend == 'BUY':
            # For BUY, stop-loss below current price, take-profit above current price
            stop_loss = latest_close - (latest_close * 0.0055)
            take_profit = latest_close + (latest_close * 0.01)
            
    elif trend == 'SELL':
            # For SELL, stop-loss above current price, take-profit below current price
            stop_loss = latest_close + (latest_close * 0.0055)
            take_profit = latest_close - (latest_close * 0.01)
        
    else:
            # If trend is neutral, you could return HOLD or similar
            stop_loss = latest_close
            take_profit = latest_close
    
    # Return the adjusted values, including fees
    return latest_close, stop_loss, take_profit


# Define combined ATR decision across multiple intervals (1-minute specific)
def get_atr_exit_decision(symbol):
    
    # Fetch historical data for the 1-minute timeframe
    df_1m = get_historical_data(symbol, interval='1m')
    
    # Get ATR-based exit levels for the 1-minute timeframe, considering trend
    current_price, stop_loss_1m, take_profit_1m = atr_decision(df_1m, symbol)
    
    # Return the current price, stop-loss, and take-profit levels for 1-minute
    decision = {
        'symbol': symbol,
        'current_price': current_price,
        'stop_loss_1m': stop_loss_1m,
        'take_profit_1m': take_profit_1m
    }
    
    return decision

    
#calculate the trend, entry price, stop loss and take profit
def suggest_trade(symbol):  # Default ATR multiplier set to 2.5
    """Return trade suggestion based on multi-timeframe indicator trends"""

    try:
        #get the trading signal
        final_trend = get_trading_decision(symbol)
        # Get current price and apply dynamic stop loss and take profit based on ATR
        decission = get_atr_exit_decision(symbol) 
        current_price = decission['current_price']
        stop_loss = decission['stop_loss_1m']
        take_profit = decission['take_profit_1m']

        if stop_loss == take_profit:
            return {
                'trend': 'HOLD',
                'suggest_entry_price': current_price,
                'suggest_stop_loss': stop_loss,
                'suggest_take_profit': take_profit
            }
        else:
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
coin_pairs = ['DOTUSDT', 'INJUSDT', 'MAGICUSDT', 'LTCUSDT']  # Example with 4 pairs
# Flag to control whether the fetch_recent_orders task should be canceled
cancel_fetch_orders = False
# Function to handle trading for each symbol
def run_symbol_task(symbol):
    print(f"+ + + Running bot for == {symbol} ==")
    
    usdt_to_trade = Decimal('7.5')  # Example trade amount
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
        place_futures_order(symbol, trade_signal, entry_price,
                            Decimal(str(trade_suggestion['suggest_stop_loss'])),
                            Decimal(str(trade_suggestion['suggest_take_profit'])),
                            leverage, quantity, usdt_to_trade, effective_position_size)
    else:
        print(f"Trade suggestion: {trade_suggestion['trend']}")

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



