from math import e
import time
import os
from matplotlib.units import DecimalConverter
import pandas as pd
from binance.client import Client
import ta
import certifi
from dconfig import read_db_config
import pandas_ta as pd_ta 
from conn_ssh import create_conn
import time
#from data.database_management import update_order_in_db


# Get All Config SSH and DB
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification disabled
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

conn, tannel = create_conn()

def sync_binance_time(client):
    server_time = client.get_server_time()
    local_time = int(time.time() * 1000)
    client.time_offset = server_time['serverTime'] - local_time
    #print(f"Time offset set to {client.time_offset} ms")

# Function to fetch historical data and calculate indicators


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


def get_data_rev(symbol, interval='1m', retries=5, delay=2):
    """
    Fetch data for the specified symbol and interval, with retry logic.
    """
    attempt = 0

    while attempt < retries:
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=50)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'trades', 'tbav', 'tbqav', 'ignore'
            ])

            # Ensure numeric columns are properly converted
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            # Calculate indicators
            df['ema_short'] = ta.trend.ema_indicator(df['close'], window=7)
            df['ema_long'] = ta.trend.ema_indicator(df['close'], window=14)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)

            # Drop rows with NaN values
            df = df.dropna()

            # Validate DataFrame before returning
            if df.empty:
                raise ValueError("DataFrame is empty after processing.")
            
            # Debug print: check the first few rows
            #print(df.head())
            
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
        positions = client.futures_account(recvWindow=10000)['positions']
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


# Function to close a position
def averaging_order(symbol, side, amount):
    try:
        
        quantity = round(abs(amount), 8)   # double the amou

        client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        positionSide='LONG' if side == 'BUY' else 'SHORT',
                        type='MARKET',
                        quantity=quantity
                        )
        print(f"[AVERAGING ORDER] Placing Averaging Order for {symbol} ({side}).")
        return
    except Exception as e:
        print(f"Error placing averaging order for {symbol}: {e}")


   
def calculate_rsi(series, period):
    """
    Calculate RSI using a pandas Series of closing prices.

    Args:
        series (pd.Series): The Series of closing prices.
        period (int): The lookback period for RSI calculation.

    Returns:
        pd.Series: A Series containing the RSI values.
    """
    # Calculate price differences
    delta = series.diff()

    # Calculate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate rolling averages
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Calculate RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle cases where avg_loss is zero (set RSI to 100)
    rsi[avg_loss == 0] = 100

    # Handle cases where avg_gain is zero (set RSI to 0)
    rsi[avg_gain == 0] = 0

    return rsi

# calculate reversal for exit, if side = BUY and this function return SELL then reversal spoted 

def detect_reversal_trend(symbol):
    """
    Detect volume spikes and confirm with RSI(9), Bollinger Bands(9), and ADX.
    Returns BUY, SELL, or HOLD based on conditions.
    """
    df = get_data_rev(symbol)
    if df is None or df.empty:
        print(f"[{symbol}] No data available for tracking.")
        return {"close_position": False, "reason": "No data"}
    
    try:
        # Validate input DataFrame
        required_columns = {'close', 'high', 'low', 'volume'}
        if not required_columns.issubset(df.columns):
            missing_columns = required_columns - set(df.columns)
            return {"signal": "NONE-NONE", "reason": f"Missing required columns: {', '.join(missing_columns)}"}

        if len(df) < 10:  # Ensure enough data points
            return {"signal": "NONE-10", "reason": "Insufficient data points"}

        # Calculate indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=9)
        sma = df['close'].rolling(window=9).mean()
        std = df['close'].rolling(window=9).std()
        df['upper_band'] = sma + (2 * std)
        df['lower_band'] = sma - (2 * std)
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)

        # Volume spike detection
        mean_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        volume_spike = current_volume > 1.5 * mean_volume

        # Get latest indicator values
        rsi = df['rsi'].iloc[-1]
        close_price = df['close'].iloc[-1]
        upper_band = df['upper_band'].iloc[-1]
        lower_band = df['lower_band'].iloc[-1]
        
        # Decision logic
        if (rsi < 30 or close_price <= lower_band) and volume_spike:
            return {"signal": "BUY", "reason": "RSI oversold, price near lower band, and volume spike"}
        elif (rsi > 70 or close_price >= upper_band) and volume_spike:
            return {"signal": "SELL", "reason": "RSI overbought, price near upper band, and volume spike"}
        else:
            return {"signal": "HOLD", "reason": "No strong signal"}

    except Exception as e:
        return {"signal": "NONE-ERR", "reason": f"Error: {e}"}



# track the market movement and decide the perfect timing for exit.

import time

def track_trade(
    symbol,
    side,
    amount,
    entry_price,
    max_loss=10,  # Maximum allowable loss in percentage
    max_profit=1,  # Minimum target profit in percentage
    bollinger_window=9,  # Bollinger Bands window
    bollinger_std_dev=2,  # Bollinger Bands standard deviation
    rsi_oversold_zone=40,  # RSI oversold zone
    rsi_overbought_zone=60,  # RSI overbought zone
    rsi_length=9,  # RSI calculation period
    sleep_time=1  # Time to sleep between checks (in seconds)
):
    """
    Monitor price movements and handle dynamic closing conditions,
    including Bollinger Bands and RSI-based reversals.
    """
    #sync_binance_time(client)

    try:
        # Fetch data
        df = get_data(symbol, '1m')
        if df is None or df.empty:
            print(f"[{symbol}] No data available for tracking.")
            return {"close_position": False, "reason": "No data"}
            

        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=bollinger_window).mean()
        df['std_dev'] = df['close'].rolling(window=bollinger_window).std()
        df['upper_band'] = df['sma'] + (df['std_dev'] * bollinger_std_dev)
        df['lower_band'] = df['sma'] - (df['std_dev'] * bollinger_std_dev)

        # Calculate RSI
        df['rsi'] = calculate_rsi(df['close'], rsi_length)

        # Validate indicator data
        if df[['upper_band', 'lower_band']].isna().all().any() or df['rsi'].isna().all():
            print(f"[{symbol}] Insufficient indicator data.")
            return {"close_position": False, "reason": "Insufficient indicator data"}
        
        
        # Fetch latest values
        latest_close = get_current_price(symbol)

        previous_close = df['close'].iloc[-2]
        latest_upper_band = df['upper_band'].iloc[-1]
        latest_lower_band = df['lower_band'].iloc[-1]
        latest_mid_band = df['sma'].iloc[-1]
        latest_rsi = df['rsi'].iloc[-1]
        previous_rsi = df['rsi'].iloc[-2]
        rsi_delta = previous_rsi - latest_rsi
        closing_price = latest_close
        
        # define minimum target profit for exit
        min_profit = 0.1 * max_profit
        micro_profit = 0.45 * max_profit
        # check price reversal for exit
        price_reversal = get_price_reversal(symbol,side)
        #price_reversal = get_price_reversal_new(symbol,side)
        

        # Calculate PnL
        if side == 'BUY':
            # For BUY position, profit is when current price > entry price
            profit_relative = ((latest_close - entry_price) / entry_price) * 100
            # For BUY position, loss is when current price < entry price
            loss_relative = ((entry_price - latest_close) / entry_price) * 100

            # RSI reversal
            rsi_reversal_profit = (
                (latest_rsi >= rsi_overbought_zone) or # overbought
                (latest_rsi < rsi_overbought_zone and previous_rsi >= rsi_overbought_zone) or # overbought reversal
                (latest_rsi < 50 and previous_rsi >= 50) or # fail breakout
                (latest_rsi < 50 and previous_rsi <=50 and latest_rsi < previous_rsi) # fail breakdown with low price action
            )
            # Bollinger Reversal
            boll_reversal_profit = (
                (latest_close >= latest_upper_band) or # overbought
                (latest_close < latest_mid_band and previous_close > latest_mid_band) or #fail breakout
                (latest_close < previous_close and latest_close < latest_mid_band)  #low price action
            )
        elif side == 'SELL':
            # For SELL position, profit is when current price < entry price
            profit_relative = ((entry_price - latest_close) / entry_price) * 100
            # For SELL position, loss is when current price > entry price
            loss_relative = ((latest_close - entry_price) / entry_price) * 100

            # RSI reversal
            rsi_reversal_profit = (
                (latest_rsi <= rsi_oversold_zone) or # oversold met
                (latest_rsi > rsi_oversold_zone and previous_rsi < rsi_oversold_zone) or # oversold reversal met
                (latest_rsi > 50 and previous_rsi < 50) or # fail breakdown
                (latest_rsi > 50 and previous_rsi <= 50 and latest_rsi > previous_rsi) # fail breakdown with low price action
            )
            #Bollinger reversal
            boll_reversal_profit =  (
                (latest_close <= latest_lower_band) or # touching lower band 
                (latest_close > latest_mid_band and previous_close < latest_mid_band) or # fail breakdown
                (latest_close > previous_close and latest_close > latest_mid_band) # fail breakdown low price action
            )
        #print(f"[Martingle] Martingle = {martingle}, Reversal_averaging:{reversal_for_averaging}, Amount:{amount}, original_amount: {amount_2}")
        # Log PnL
        profit_label = '* * PROFIT' if profit_relative > 0 else '~ ~ LOSS'
        print(f"[Tracking PnL] {symbol} {side} {profit_label} = {profit_relative:.2f}%")

        
        # 1. opportunis taking profit
        if price_reversal and profit_relative >= micro_profit:
            close_position(symbol, side)
            print(f"[* * * * CLOSED] {symbol} Closed due to price reversal with profit of {profit_relative:.2f}%.")
            return {"close_position": True, "reason": "Price reversal with profit"}   
        
        # 2. Check if profit but not met the max profit and reversal detected
        
        if min_profit <= profit_relative <= max_profit:
            print(f"Checking exit conditions for {symbol}: Profit={profit_relative:.2f}%")
            
            # List of exit conditions
            exit_conditions = [
                (boll_reversal_profit, "Price crossed Bollinger Bands after profit"),
                (rsi_reversal_profit, f"Profit > {min_profit:.2f}% with Potential RSI reversal"),
                (price_reversal, f"Profit > {min_profit:.2f}% with Potential Price reversal"),
            ]

            for condition, reason in exit_conditions:
                if condition:
                    close_position(symbol, side)
                    print(f"[* * * * CLOSED] {symbol} due to {reason} at {latest_close:.2f}")
                    return {"close_position": True, "reason": reason}

            print(f"No exit condition met for {symbol}.")
            return {"close_position": False, "reason": "No exit condition met"}    
        
        # 3. Check if profit met the max profit and no reversal detected

        if profit_relative >= max_profit: 
            print(f"Checking exit conditions for {symbol}: Profit={profit_relative:.2f}%")
            
            # List of exit conditions
            exit_conditions = [
                (boll_reversal_profit, "Price crossed Bollinger Bands after profit"),
                (rsi_reversal_profit, f"Profit > {min_profit:.2f}% with Potential RSI reversal"),
                (price_reversal, f"Profit > {min_profit:.2f}% with Potential Price reversal"),
            ]

            for condition, reason in exit_conditions:
                if condition:
                    close_position(symbol, side)
                    print(f"[* * * * CLOSED] {symbol} due to {reason} at {latest_close:.2f}")
                    return {"close_position": True, "reason": reason}
        
            print(f"No exit condition met for {symbol}.")
            return {"close_position": False, "reason": "No exit condition met"}
        
    except Exception as e:
        print(f"[{symbol}] Error in trade tracking: {e}")
        return {"close_position": False, "reason": str(e)}

    return {"close_position": False, "reason": "No exit conditions met"}


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

# get the price riversal

def get_price_reversal(symbol, side):
    """
    Monitors price movements and determines if a price reversal has occurred
    based on ATR-based thresholds. Returns True for a reversal, False otherwise.
    """
    try:
        # Fetch initial data and calculate ATR
        df = get_data(symbol, '1m')  # Fetch recent 1-minute candlestick data
        if df is None or df.empty:
            return False

        atr = get_atr(symbol, period=9)  # Calculate ATR(9)
        if atr is None:
            return False

        # Set initial prices
        last_price = df['close'].iloc[-2]  # Second-to-last closed price
        current_price = df['close'].iloc[-1]  # Most recent closed price

        while True:
            # Fetch the latest price data
            df = get_data(symbol, '1m')  # Refresh data
            if df is None or df.empty:
                return False

            # Update prices
            current_price = df['close'].iloc[-1]  # Most recent closed price
            previous_price = last_price  # Store last recorded close price
            last_price = current_price  # Update last price for next iteration
            
            
            atr_tracehold = atr * 0.15

            # Check for reversals based on ATR
            if side == 'BUY' and current_price < previous_price - atr_tracehold:
                return True  # Reversal detected for BUY

            if side == 'SELL' and current_price > previous_price + atr_tracehold:
                return True  # Reversal detected for SELL

    except Exception:
        return False
    

# calculate ATR

def get_atr(symbol, period=9):
    """
    Calculates the Average True Range (ATR) for the given symbol.
    
    Parameters:
        symbol (str): Trading pair (e.g., 'BTCUSDT').
        period (int): Lookback period for ATR (default is 9).
    
    Returns:
        float: ATR value.
    """
    df = get_data(symbol, '1m')  # Fetch recent 1-minute candlestick data
    if df is None or len(df) < period:
        return None

    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    atr = df['true_range'].rolling(window=period).mean().iloc[-1]
    return atr

# Function to fetch all open positions
def get_open_positions():
    try:
        # Synchronize Binance time before making the API call
        sync_binance_time(client)

        # Fetch all positions from the futures account
        positions = client.futures_account(recvWindow=10000)['positions']
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return []
    
    open_positions = []
    for pos in positions:
        try:
            position_amt = float(pos['positionAmt'])  # Position amount
            entry_price = float(pos['entryPrice'])  # Entry price
            
            if position_amt != 0:  # Check if position is open
                side = 'BUY' if position_amt > 0 else 'SELL'  # Determine side
                open_positions.append({
                    'symbol': pos['symbol'],
                    'positionAmt': position_amt,
                    'entryPrice': entry_price,  # Include entry price
                    'side': side
                })
                #print(f"[DEBUG] Found open position: {pos['symbol']} | Amount: {position_amt} | Entry Price: {entry_price} | Side: {side}")
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing position: {e}, Data: {pos}")
    
    return open_positions

# get realtime price
def get_current_price(symbol):
    """
    Fetch the current mark price for a given symbol.
    :param symbol: The trading symbol (e.g., 'BTCUSDT').
    :return: The latest mark price (float) or 0 on error.
    """
    try:
        # Ensure time synchronization periodically
        sync_binance_time(client)

        # Fetch the current mark price
        ticker_data = client.futures_symbol_ticker(recvWindow=10000, symbol=symbol)
        return float(ticker_data['price'])
    except Exception as e:
        print(f"Error fetching current price for {symbol}: {e}")
        return 0  # Fallback value


# new get open position insluding get order id 

def get_open_positions_with_order_id():
    """
    Fetches open positions and includes associated order IDs.
    """
    try:
        # Synchronize Binance time before making the API call
        sync_binance_time(client)

        # Fetch all positions
        positions = client.futures_account(recvWindow=10000)['positions']

        # Fetch all open orders
        open_orders = client.futures_get_open_orders(recvWindow=10000)

    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

    open_positions = []
    for pos in positions:
        try:
            position_amt = float(pos['positionAmt'])  # Position amount
            entry_price = float(pos['entryPrice'])  # Entry price

            if position_amt != 0:  # Check if position is open
                # Find matching orders for the position
                related_orders = [
                    order for order in open_orders if order['symbol'] == pos['symbol']
                ]

                # Extract the first order ID (if any) for simplicity
                order_id = related_orders[0]['orderId'] if related_orders else None

                open_positions.append({
                    'symbol': pos['symbol'],
                    'positionAmt': position_amt,
                    'entryPrice': entry_price,
                    'side': 'BUY' if position_amt > 0 else 'SELL',
                    'order_id': order_id  # Include order_id
                })
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing position: {e}, Data: {pos}")

    return open_positions

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
                print("[Monitoring] No open positions available.")
                time.sleep(10)
                continue

            for order in orders:
                # Extract details for the current order
                symbol = order['symbol']
                side = order['side']
                amount = float(order['positionAmt'])  # Ensure the amount is a float for calculations
                entry_price = float(order['entryPrice'])  # Ensure entry price is a float

                # Debugging: Print order details
                #print(f"[INFO] Checking order for {symbol}: {side}, Amount: {amount}, Entry Price: {entry_price}")

                # Validate position details
                if entry_price == 0 or amount == 0:
                    print(f"[INCOMPLETE POSITION] Missing details for {symbol}. Skipping...")
                    continue

                # Debugging: Ensure we're calling the track_trade function
                #print(f"[INFO] Calling track_trade for {symbol} with {side}, {amount}, {entry_price}")
                result = track_trade(symbol, side, amount, entry_price)

                if result['close_position']:
                    print(f"[CLOSE SIGNAL] {symbol} {side} closed. Reason: {result['reason']}")
                #else:
                #    print(f"[Monitoring] {symbol} {side} - No exit condition met.")

        except Exception as e:
            print(f"[ERROR] Error in monitoring positions: {e}")
        finally:
            # Adjust based on API rate limits
            time.sleep(3)













