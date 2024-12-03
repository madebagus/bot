import os
from flask import Blueprint, jsonify, request
from binance.client import Client
import certifi
import ta
import pandas as pd
from dconfig import read_db_config

# Initialize the blueprint for the bot
trend_bp = Blueprint('trend', __name__, url_prefix='/api/trend/')

# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification enabled
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

# Define the RSI calculation logic
import pandas_ta as pd_ta

def rsi_decision(symbol, price_column='close', interval='1m', lookback="12 hour ago UTC"):
    """
    Fetch historical data, calculate RSI, and determine trend.
    
    Parameters:
        symbol (str): Trading pair (e.g., 'BTCUSDT').
        price_column (str): Column to use for RSI calculation (default: 'close').
        interval (str): Interval for data fetching (default: '1m').
        lookback (str): Timeframe for fetching historical data (default: '12 hour ago UTC').
    
    Returns:
        str: Trend ('BUY', 'SELL', 'STRONG BUY', 'STRONG SELL', or 'HOLD').
    """
    # Fetch historical data
    df = get_historical_data(symbol, interval=interval, lookback=lookback)
    
    # Check if we have enough data for RSI calculation (at least 14 data points)
    if len(df) < 14:
        print(f"Insufficient data for {symbol}. Need at least 14 data points for RSI calculation.")
        return 'HOLD'
    
    # Calculate RSI using pandas-ta
    df['RSI'] = pd_ta.rsi(df[price_column], length=14)
    
    # Ensure RSI values exist
    if df['RSI'].isnull().sum() > 0:
        print(f"Insufficient RSI data for {symbol}. Returning HOLD.")
        return 'HOLD'

    # Define RSI thresholds
    midline = 50
    lower_limit = midline * 0.95  # 47.5
    upper_limit = midline * 1.05  # 52.5

    # Get recent RSI values
    latest_rsi = df['RSI'].iloc[-1]
    previous_rsi = df['RSI'].iloc[-2]
    two_previous_rsi = df['RSI'].iloc[-3]

    # Default trend
    trend = 'HOLD'

    # 1. Extreme overbought/oversold conditions
    if latest_rsi >= 80:
        trend = 'STRONG SELL'
    elif latest_rsi <= 20:
        trend = 'STRONG BUY'
    # 2. Midline conditions for BUY and SELL
    elif (
        latest_rsi >= lower_limit and latest_rsi <= midline and
        previous_rsi >= lower_limit and previous_rsi <= midline and
        two_previous_rsi >= lower_limit and two_previous_rsi <= midline
    ):
        trend = 'SELL'
    elif (
        latest_rsi <= upper_limit and latest_rsi >= midline and
        previous_rsi <= upper_limit and previous_rsi >= midline and
        two_previous_rsi <= upper_limit and two_previous_rsi >= midline
    ):
        trend = 'BUY'
    # 3. Crossovers
    elif latest_rsi > 55 and previous_rsi <= 35:
        trend = 'BUY'
    elif latest_rsi < 45 and previous_rsi >= 55:
        trend = 'SELL'
    elif latest_rsi >= 35 and previous_rsi <= 20:
        trend = 'BUY'
    elif latest_rsi <= 60 and previous_rsi >= 70:
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

def get_ema_trend_decision():
    symbol = request.args.get('symbol', 'BTCUSDT')
    
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

# Define the VWAP calculation logic
def vwap_decision(df):
    """Calculate VWAP and compare with closing price for trend indication."""
    # Calculate VWAP
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    latest_close = df['close'].iloc[-1]
    latest_vwap = vwap.iloc[-1]
    
    # Decision based on VWAP
    if latest_close > latest_vwap:
        return 'BUY'
    elif latest_close < latest_vwap:
        return 'SELL'
    else:
        return 'HOLD'

def get_vwap_trend_decision():
    symbol = request.args.get('symbol', 'BTCUSDT')
    
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
        return  'BUY'
    elif sell_signals >= 2:
        return  'SELL'
    else:
        return  'HOLD'    
    
# Define the Bollinger Bands calculation logic
def bollinger_bands_decision(df, price_column='close', window=20, num_std_dev=2):
    """Calculate Bollinger Bands and compare with closing price for trend indication."""
    # Calculate the Bollinger Bands
    sma = df[price_column].rolling(window=window).mean()
    std = df[price_column].rolling(window=window).std()
    upper_band = sma + (std * num_std_dev)
    lower_band = sma - (std * num_std_dev)
    
    latest_close = df[price_column].iloc[-1]
    latest_lower_band = lower_band.iloc[-1]
    latest_upper_band = upper_band.iloc[-1]
    
    # Decision based on Bollinger Bands
    if latest_close < latest_lower_band:
        return 'BUY'
    elif latest_close > latest_upper_band:
        return 'SELL'
    else:
        return 'HOLD'
    
def get_bollinger_bands_trend_decision():
    symbol = request.args.get('symbol', 'BTCUSDT')
    
    # Fetch historical data for each interval
    df_1m = get_historical_data(symbol, interval='1m')
    df_3m = get_historical_data(symbol, interval='3m')
    df_5m = get_historical_data(symbol, interval='5m')
    
    # Get Bollinger Bands decisions for each interval
    bollinger_dec_1m = bollinger_bands_decision(df_1m)
    bollinger_dec_3m = bollinger_bands_decision(df_3m)
    bollinger_dec_5m = bollinger_bands_decision(df_5m)
    
    # Count BUY and SELL signals
    buy_signals = sum(dec == 'BUY' for dec in [bollinger_dec_1m, bollinger_dec_3m, bollinger_dec_5m])
    sell_signals = sum(dec == 'SELL' for dec in [bollinger_dec_1m, bollinger_dec_3m, bollinger_dec_5m])
    
    # Final decision: at least 2 signals for BUY or SELL
    if buy_signals >= 2:
        return  'BUY'
    elif sell_signals >= 2:
        return  'SELL'
    else:
        return  'HOLD'
    
# Define the MACD calculation logic with shortened parameters
import ta

def macd_decision(df, fast_period=6, slow_period=13, signal_period=4):
    """Calculate MACD with shortened parameters and provide a decision."""
    
    # Calculate the MACD line
    macd = ta.trend.MACD(df['close'], window_slow=slow_period, window_fast=fast_period)
    macd_line = macd.macd()  # MACD line
    macd_signal = macd.macd_signal()  # Signal line
    
    # Calculate the MACD Histogram (Optional for extra insights)
    macd_histogram = macd_line - macd_signal
    
    # Get the latest MACD, Signal line values, and Histogram
    latest_macd = macd_line.iloc[-1]
    latest_signal = macd_signal.iloc[-1]
    latest_histogram = macd_histogram.iloc[-1]
    
    # Determine if we have a Bullish or Bearish crossover
    
    # Bearish Crossover: MACD crosses below Signal Line
    if latest_macd > latest_signal and macd_line.iloc[-2] < macd_signal.iloc[-2]:
        crossover = 'bullish'
    elif latest_macd < latest_signal and macd_line.iloc[-2] > macd_signal.iloc[-2]:
        crossover = 'bearish'
    else:
        crossover = 'neutral'

    # Provide a trading decision based on the crossover
    if crossover == 'bullish':
        return 'BUY'
    elif crossover == 'bearish':
        return 'SELL'
    else:
        return 'HOLD'



def get_macd_decision():
    symbol = request.args.get('symbol', 'BTCUSDT')
    
    # Fetch historical data for each interval
    df_1m = get_historical_data(symbol, interval='1m')
    df_3m = get_historical_data(symbol, interval='3m')
    df_5m = get_historical_data(symbol, interval='5m')
    
    # Get MACD decision for each interval
    macd_decision_1m = macd_decision(df_1m)
    macd_decision_3m = macd_decision(df_3m)
    macd_decision_5m = macd_decision(df_5m)
    
    # Combine the decisions and apply logic: Minimum 2 "BUY" or "SELL" for final decision
    decisions = [macd_decision_1m, macd_decision_3m, macd_decision_5m]
    
    if decisions.count('BUY') >= 2:
        return  'BUY'
    elif decisions.count('SELL') >= 2:
        return  'SELL'
    else:
        return  'HOLD'
    
    
def stoch_rsi_decision(df, stoch_rsi_period=14, stoch_rsi_k_period=3, stoch_rsi_d_period=3):
    """Analyze Stochastic RSI for overbought/oversold, momentum, and midline crossovers."""
    
    # Calculate Stochastic RSI using pandas-ta (stochastic RSI uses %K and %D)
    stoch_rsi = pd_ta.stochrsi(df['close'], length=stoch_rsi_period, rsi_length=stoch_rsi_k_period, stoch_length=stoch_rsi_d_period)
    
    # Check available columns in the DataFrame
    print(stoch_rsi.columns)

    # Use the correct column from the Stochastic RSI DataFrame
    df['stoch_rsi'] = stoch_rsi['STOCHRSI_k']  # Adjust the column name based on what you find
    
    # Drop rows with NaN values in 'stoch_rsi' column
    df = df.dropna(subset=['stoch_rsi'])

    # Ensure enough data points
    if len(df) < stoch_rsi_period:
        return "HOLD"

    # Get the latest StochRSI values
    latest_stoch_rsi = df['stoch_rsi'].iloc[-1]
    previous_stoch_rsi = df['stoch_rsi'].iloc[-2]

    # **1. Overbought and Oversold Conditions**
    if latest_stoch_rsi > 0.8:
        return "STRONG SELL"  # Overbought condition
    elif latest_stoch_rsi < 0.2:
        return "STRONG BUY"  # Oversold condition

    # **2. Momentum (Sharp Movements)**
    stoch_rsi_change = latest_stoch_rsi - previous_stoch_rsi
    momentum_threshold = 0.1  # Adjust based on market behavior
    if stoch_rsi_change > momentum_threshold:
        return "STRONG BUY"  # Strong upward momentum
    elif stoch_rsi_change < -momentum_threshold:
        return "STRONG SELL"  # Strong downward momentum

    # **3. Midline Crossovers**
    if previous_stoch_rsi < 0.5 and latest_stoch_rsi > 0.5:
        return "BUY"  # Bullish crossover (reversal)
    elif previous_stoch_rsi > 0.5 and latest_stoch_rsi < 0.5:
        return "SELL"  # Bearish crossover (reversal)

    # **4. Sudden Reversal from Overbought/Oversold**
    if 0.8 > latest_stoch_rsi > 0.6 and previous_stoch_rsi > 0.8:
        return "SELL"  # Reversal from overbought
    elif 0.2 < latest_stoch_rsi < 0.4 and previous_stoch_rsi < 0.2:
        return "BUY"  # Reversal from oversold

    # Default to HOLD
    return "HOLD"

# Combine RSI and EMA decisions for the final trading decision
@trend_bp.route('/decision', methods=['GET'])
def get_trading_decision():
    symbol = request.args.get('symbol', 'BTCUSDT')
    interval = request.args.get('interval', '1h')
    
    # Fetch historical data
    df = get_historical_data(symbol, interval)
    
    # Get RSI and EMA decisions
    rsi_dec = rsi_decision(df)
    if rsi_dec == 'BUY':
        rsi_score = 1
    elif rsi_dec == 'SELL':
        rsi_score = -1
    else:
        rsi_score = 0

    ema_dec = get_ema_trend_decision()
    if ema_dec == 'BUY':
        ema_score = 1
    elif ema_dec == 'SELL':
        ema_score = -1
    else:
        ema_score = 0

    vwap_dec = get_vwap_trend_decision()
    if vwap_dec == 'BUY':
        vwap_score = 1
    elif vwap_dec == 'SELL':
        vwap_score = -1
    else:
        vwap_score = 0

    bollinger_dec = get_bollinger_bands_trend_decision()
    if bollinger_dec == 'BUY':
        bollinger_score = 1
    elif bollinger_dec == 'SELL':
        bollinger_score = -1
    else:
        bollinger_score = 0

    macd_dec = get_macd_decision()
    if macd_dec == 'BUY':
        macd_score = 1
    elif macd_dec == 'SELL':
        macd_score = -1
    else:
        macd_score = 0

    stoc_rsi_dec = stoch_rsi_decision(df)
    if stoc_rsi_dec == 'BUY':
        stoc_rsi_score = 1
    elif stoc_rsi_dec == 'SELL':
        stoc_rsi_score = -1
    else:
        stoc_rsi_score = 0

    # Weighted scoring (adjusted for HFT strategy)
    rsi_weight = 0.1       # Lower weight for RSI as it's slower to react
    macd_weight = 0.4     # Higher weight for MACD due to its relevance in momentum
    bollinger_weight = 0.2 # Medium weight for Bollinger Bands to capture volatility
    vwap_weight = 0.3   # High weight for Volume as it confirms the strength of moves
    ema_weight = 0.1        # Lower weight for MA in HFT to reduce lag
    stoc_rsi_weight = 0.1

    # Calculate total score by applying weights
    total_score = (rsi_score * rsi_weight) + (macd_score * macd_weight) + \
                 (bollinger_score * bollinger_weight) + (vwap_score * vwap_weight) + (ema_score * ema_weight) + \
                 (stoc_rsi_score * stoc_rsi_weight)

    print("RSI Decision : ",rsi_dec)
    print("Stoc RSI Decision : ",stoc_rsi_score)
    print("EMA Decision :",ema_dec)
    print("VWAP Decision :",vwap_dec)
    print("Bollinger Decision :",bollinger_dec)
    print("MACD Decision :",macd_dec)

    # Combine decisions: only buy if both are buy, and sell if both are sell
    if total_score > 0.2:
        trend = 'BUY'
    elif total_score < -0.2:
        trend = 'SELL'
    else:
        trend = 'HOLD'
    
    # Return combined decision as JSON
    return jsonify({"symbol": symbol, "interval": interval, "decision": trend})
