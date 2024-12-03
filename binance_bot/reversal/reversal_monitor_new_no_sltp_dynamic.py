import time
import os
from bot.new_bot_xman import suggest_trade
from matplotlib.units import DecimalConverter
import pandas as pd
from binance.client import Client
import ta
import certifi
from dconfig import read_db_config
import pandas_ta as pd_ta 
from conn_ssh import create_conn


# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

# Function to fetch historical data and calculate indicators
import time

def get_data(symbol, interval, retries=5, delay=2):
    """
    Fetch data for the specified symbol and interval, with retry logic.

    Args:
        symbol (str): Trading symbol (e.g., 'BTCUSDT').
        interval (str): Timeframe interval (e.g., '1m', '5m').
        retries (int): Maximum number of retries.
        delay (int): Delay in seconds between retries.

    Returns:
        pd.DataFrame: DataFrame containing the data, or None if unsuccessful.
    """
    attempt = 0

    while attempt < retries:
        try:
            #print(f"Attempt {attempt + 1} to fetch data for {symbol}...")
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=50)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'trades', 'tbav', 'tbqav', 'ignore'
            ])
            # Convert close prices to numeric
            df['close'] = pd.to_numeric(df['close'])

            # Calculate EMA and RSI indicators
            df['ema_short'] = ta.trend.ema_indicator(df['close'], window=7)
            df['ema_long'] = ta.trend.ema_indicator(df['close'], window=14)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # Drop NaN values
            df = df.dropna()

            # Check if the DataFrame is valid
            if df.empty:
                raise ValueError("DataFrame is empty after processing.")

            print(f"Data fetched successfully for {symbol} on attempt {attempt + 1}")
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


# Function to close a position
def close_position(symbol, side):
    try:
        positions = client.futures_account()['positions']
        for pos in positions:
            if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                quantity = round(abs(float(pos['positionAmt'])), 8)  # Ensure precision

                client.futures_create_order(
                        symbol=symbol,
                        side='SELL' if side == 'BUY' else 'BUY',
                        positionSide='LONG' if side == 'BUY' else 'SHORT',
                        type='MARKET',
                        quantity=quantity
                )
                print(f"[CLOSE ORDER] Position closed for {symbol} ({side}).")
                return
    except Exception as e:
        print(f"Error closing position for {symbol}: {e}")
   
#calculate RSI
def calculate_rsi(series, period=4):
    """
    Calculate RSI using a pandas Series of closing prices.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# trac the open orders and decide the exit base on some condition

def track_trade(
    symbol,
    side,
    amount,
    entry_price,
    max_loss=0.35,  # Maximum allowable loss in percentage
    max_profit=0.45,  # Minimum target profit in percentage
    bollinger_window=20,  # Bollinger Bands window
    bollinger_std_dev=2,  # Bollinger Bands standard deviation
    rsi_oversold_zone=25,  # RSI oversold for reversal
    rsi_overbough_zone=75  # RSI overbough for reversal
):
    """
    Monitor price movements and handle dynamic closing conditions, including Bollinger Bands and RSI-based reversals.
    """
    try:
        # Fetch data
        df = get_data(symbol, '1m')
        if df is None or df.empty:
            print(f"No data available for {symbol}")
            return {"close_position": False, "reason": "No data"}

        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=bollinger_window).mean()
        df['std_dev'] = df['close'].rolling(window=bollinger_window).std()
        df['upper_band'] = df['sma'] + (df['std_dev'] * bollinger_std_dev)
        df['lower_band'] = df['sma'] - (df['std_dev'] * bollinger_std_dev)

        # Calculate RSI
        df['rsi'] = calculate_rsi(df['close'])

        # Ensure Bollinger Bands and RSI are valid
        if df[['upper_band', 'lower_band']].isna().all().any() or df['rsi'].isna().all():
            print(f"Insufficient indicator data for {symbol}.")
            return {"close_position": False, "reason": "Insufficient indicator data"}

        # Fetch latest values
        latest_close = get_current_price(symbol)
        previous_close = df['close'].iloc[-2]

        latest_upper_band = df['upper_band'].iloc[-1]
        latest_lower_band = df['lower_band'].iloc[-1]
        latest_mid_band = df['sma'].iloc[-1]
        latest_rsi = df['rsi'].iloc[-1]
        previous_rsi = df['rsi'].iloc[-2]
        

        rsi_delta_1 = previous_rsi - latest_rsi # delta previous and latest rsi 

        # Inline PnL Calculation
        if side == 'BUY':
            profit_relative = ((latest_close - entry_price) / entry_price) * 100
            loss_relative = ((entry_price - latest_close) / entry_price) * 100
        elif side == 'SELL':
            profit_relative = ((entry_price - latest_close) / entry_price) * 100
            loss_relative = ((latest_close - entry_price) / entry_price) * 100
        else:
            print(f"Invalid side: {side}")
            return {"close_position": False, "reason": "Invalid trade side"}

        print(f"[PnL Check] {symbol} {side}: Profit/Loss= {profit_relative}%")

        #1. Exit when overbough or oversold

        if (side == 'BUY' and latest_rsi >= rsi_overbough_zone) or (side == 'SELL' and latest_rsi <= rsi_oversold_zone):
            close_position(symbol, side)
            print(f"[* * * CLOSE ORDER ] {symbol} {side} Strong RSI reversal detected at {latest_rsi}")
            return {"close_position": True, "reason": "RSI strong reversal"}

        #2. Exit when loss more then tracehold and trend reverse

        if loss_relative >= max_loss and (
            (side == 'BUY' and (rsi_delta_1 > 5 or latest_rsi < rsi_oversold_zone - 5)) or
            (side == 'SELL' and (rsi_delta_1 < -5 or latest_rsi > rsi_overbough_zone + 5))
        ):
            close_position(symbol, side)
            print(f"[* * * CLOSE ORDER ] {symbol} {side} Loss > {max_loss}% and RSI reversal detected at {latest_rsi}")
            return {"close_position": True, "reason": "Loss threshold with RSI reversal"}
        
        #3. Exit when loss more then tracehold and trend reverse

        if profit_relative >= max_profit and (
            (side == 'BUY' and (latest_rsi >= rsi_overbough_zone or 
                                (latest_rsi < rsi_overbough_zone and previous_rsi > rsi_overbough_zone) or 
                                latest_rsi < previous_rsi)
                                ) or
            (side == 'SELL' and (latest_rsi <= rsi_oversold_zone or 
                                (latest_rsi > rsi_oversold_zone and previous_rsi < rsi_oversold_zone) or 
                                latest_rsi > previous_rsi))
        ):
            close_position(symbol, side)
            print(f"[* * * CLOSE ORDER ] {symbol} {side} Profit > {max_profit}% or RSI reversal detected at {latest_rsi}")
            return {"close_position": True, "reason": "Profit threshold or RSI condition"}

        #4. Exit based on Bollinger Bands

        if side == 'BUY':
            # Corrected BUY exit conditions
            if latest_close >= latest_upper_band:# touch upper band
                close_position(symbol, side)
                print(f"[* * * CLOSE ORDER ] {symbol} {side} Price touched upper Bollinger Band at {latest_close}")
                return {"close_position": True, "reason": "Price touched upper Bollinger Band"}
            if latest_close < latest_mid_band and previous_close >= latest_mid_band:  # fail breakout midline and rsi overbough
                close_position(symbol, side)
                print(f"[* * * CLOSE ORDER ] {symbol} {side} Price crossed below midline")
                return {"close_position": True, "reason": "Price crossed midline with RSI reversal"}

        elif side == 'SELL':
            # Corrected SELL exit conditions
            if latest_close <= latest_lower_band:# touch lower band
                close_position(symbol, side)
                print(f"[* * * CLOSE ORDER ] {symbol} {side} Price touched lower Bollinger Band at {latest_close}")
                return {"close_position": True, "reason": "Price touched lower Bollinger Band"}
            if latest_close > latest_mid_band and previous_close <= latest_mid_band: # fail down midline and rsi oversold
                close_position(symbol, side)
                print(f"[* * * CLOSE ORDER ] {symbol} {side} Price crossed above midline and RSI reversal detected")
                return {"close_position": True, "reason": "Price crossed midline with RSI reversal"}

    except Exception as e:
        print(f"Error tracking trade for {symbol}: {e}")
        return {"close_position": False, "reason": "Error"}

    return {"close_position": False, "reason": "No conditions met"}


# Function to fetch all open positions
def get_open_positions():
    try:
        # Fetch all positions from the futures account
        positions = client.futures_account()['positions']
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return []
    
    open_positions = []
    for pos in positions:
        try:
            position_amt = float(pos['positionAmt'])  # Position amount
            entry_price = float(pos['entryPrice'])  # Entry price
            
            if position_amt != 0:  # Check if position is open
                open_positions.append({
                    'symbol': pos['symbol'],
                    'positionAmt': position_amt,
                    'entryPrice': entry_price,  # Include entry price
                    'side': 'BUY' if position_amt > 0 else 'SELL'  # Determine side
                })
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing position: {e}, Data: {pos}")
    
    return open_positions


# Get current traded price
def get_current_price(symbol):
    """
    Fetch the latest traded price for a given symbol.
    :param symbol: The trading symbol (e.g., 'BTCUSDT').
    :return: The latest traded price (float).
    """
    try:
        ticker_data = client.futures_symbol_ticker(symbol=symbol)
        return float(ticker_data['price'])  # Latest traded price
    except Exception as e:
        print(f"Error fetching current price for {symbol}: {e}")
        return 0  # Return 0 as a fallback


# Main function to monitor all positions

import time

def monitor_positions():
    """
    Monitors open positions dynamically and tracks their profit/loss in real-time.
    """
    while True:
        try:
            # Fetch open positions
            orders = get_open_positions()
            
            if not orders:  # If no orders are found
                print("[MONITORING] No open positions available.")
                time.sleep(30)
                continue

            for order in orders:
                # Extract details for the current order
                symbol = order['symbol']
                side = order['side']
                amount = float(order['positionAmt'])  # Ensure the amount is a float for calculations
                entry_price = float(order['entryPrice'])  # Ensure entry price is a float

                #print(f"[PROFIT MONITORING] Monitoring {symbol} {side} position with amount: {amount}")

                # Validate position details
                if entry_price == 0 or amount == 0:
                    print(f"[INCOMPLETE POSITION] Missing details for {symbol}: "
                          f"Entry price: {entry_price}, Amount: {amount}")
                    continue

                # Monitor the trade dynamically
                result = track_trade(symbol, side, amount, entry_price)

                # Log the result of the monitoring
                if result['close_position']:
                    print(f"[CLOSE SIGNAL] {symbol} {side} closed. Reason: {result['reason']}")
                #else:
                #    print(f"[MONITORING CONTINUE] {symbol} {side} - No exit condition met.")

        except Exception as e:
            print(f"[ERROR] Error in monitoring positions: {e}")
        finally:
            # Adjust based on API rate limits
            time.sleep(5)











