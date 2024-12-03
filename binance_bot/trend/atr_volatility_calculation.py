import os
from flask import Blueprint, jsonify, request
from binance.client import Client
import certifi
import ta
import pandas as pd
from dconfig import read_db_config
from trend.trend import get_trading_decision

# Initialize the blueprint for the bot
atr_bp = Blueprint('atr', __name__, url_prefix='/api/trend/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification enabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

# Fetch historical data from Binance
def get_historical_data(symbol, interval, lookback="2 hour ago UTC"):
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

def atr_decision(df, symbol, interval, window=14, atr_multiplier_stop=1, atr_multiplier_take_profit=2):
    """Calculate ATR, stop-loss, take-profit, and current price, adjusted for trend direction."""
    
    # Get the trend direction (BUY or SELL) using the trading decision
    trend = get_trading_decision(symbol, interval)
    
    # Calculate ATR using ta library
    atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=window)
    
    latest_atr = atr.iloc[-1]
    latest_close = df['close'].iloc[-1]
    
    # Adjust stop-loss and take-profit based on the trend
    if trend == 'BUY':
        # For BUY, stop-loss below current price, take-profit above current price
        stop_loss = latest_close - (latest_atr * atr_multiplier_stop)  # Stop-loss at 1x ATR below the current price
        take_profit = latest_close + (latest_atr * atr_multiplier_take_profit)  # Take-profit at 2x ATR above the current price
    elif trend == 'SELL':
        # For SELL, stop-loss above current price, take-profit below current price
        stop_loss = latest_close + (latest_atr * atr_multiplier_stop)  # Stop-loss at 1x ATR above the current price
        take_profit = latest_close - (latest_atr * atr_multiplier_take_profit)  # Take-profit at 2x ATR below the current price
    else:
        # If trend is neutral, you could return HOLD or similar
        stop_loss = latest_close
        take_profit = latest_close
    
    return latest_close, stop_loss, take_profit

# Define combined ATR decision across multiple intervals (1-minute specific)
@atr_bp.route('/atr_exit_decision', methods=['GET'])
def get_atr_exit_decision():
    symbol = request.args.get('symbol', 'BTCUSDT')
    
    # Fetch historical data for the 1-minute timeframe
    df_1m = get_historical_data(symbol, interval='1m')
    
    # Get ATR-based exit levels for the 1-minute timeframe, considering trend
    current_price, stop_loss_1m, take_profit_1m = atr_decision(df_1m, symbol, '1m')
    
    # Return the current price, stop-loss, and take-profit levels for 1-minute
    decision = {
        'symbol': symbol,
        'current_price': current_price,
        'stop_loss_1m': stop_loss_1m,
        'take_profit_1m': take_profit_1m
    }
    
    return decision


