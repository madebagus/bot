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
from binance_bot.messaging.chat_bot import send_telegram_message
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


# Function to close a position
def close_position(symbol, side, profit_relative, reason):
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
                message = f"[* * * Closing Order] {symbol}\nSide: {side}\nQuantity: {quantity} {symbol}\nProfit: {profit_relative:.2f}%\nReason: {reason} "
                send_telegram_message(message)
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


# Exit condition functions
def condition_rsi_overbought_oversold(rsi_current, rsi_overbought, rsi_oversold, position_side):
    if position_side == 'BUY' and rsi_current >= rsi_overbought:
        return True
    elif position_side == 'SELL' and rsi_current <= rsi_oversold:
        return True
    return False

def condition_rsi_breakout_sudden(rsi_current, rsi_previous, rsi_overbought, rsi_oversold, position_side):
    if position_side == 'BUY' and rsi_previous < rsi_overbought and rsi_current >= rsi_overbought:
        return True
    elif position_side == 'SELL' and rsi_previous > rsi_oversold and rsi_current <= rsi_oversold:
        return True
    return False

def condition_rsi_momentum(rsi_current, rsi_previous, position_side,rsi_overbought, rsi_oversold):
    if position_side == 'BUY':
        if rsi_current >= rsi_oversold and rsi_current < 50 and rsi_previous > 50:
            return True
    elif position_side == 'SELL':
        if rsi_current <= rsi_overbought and rsi_current > 50 and rsi_previous < 50:
            return True
    return False

def check_exit_conditions(symbol, rsi_overbought, rsi_oversold, position_side):
    # Fetch data for the specified symbol and interval
    df = get_data(symbol, '1m')
    if df is not None:
        # Get the last two RSI values from the DataFrame
        rsi_current = df['rsi'].iloc[-1]
        rsi_previous = df['rsi'].iloc[-2]

    exit_signal = False
    reason = None  # Default reason when no exit signal is triggered

    # Check the exit conditions
    if condition_rsi_overbought_oversold(rsi_current, rsi_overbought, rsi_oversold, position_side):
        reason = 'RSI in over bought/sold'
        exit_signal = True
    elif condition_rsi_breakout_sudden(rsi_current, rsi_previous, rsi_overbought, rsi_oversold, position_side):
        reason = 'RSI suddent fail break out/down'
        exit_signal = True
    elif condition_rsi_momentum(rsi_current, rsi_previous, position_side,rsi_overbought, rsi_oversold):
        reason = 'RSI weak momentum'
        exit_signal = True

    return {
        'exit_signal': exit_signal,
        'reason': reason if reason is not None else 'No exit signal detected'
    }

# track the market movement and decide the perfect timing for exit.

def track_trade(
    symbol,
    side,
    amount,
    entry_price,
    max_loss=10,  # Maximum allowable loss in percentage
    max_profit=1,  # Minimum target profit in percentage
    bollinger_window=9,  # Bollinger Bands window
    bollinger_std_dev=2,  # Bollinger Bands standard deviation
    rsi_oversold_zone=38,  # RSI oversold zone
    rsi_overbought_zone=62,  # RSI overbought zone
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

        # Define minimum target profit for exit
        min_profit = 0.1 * max_profit
        rush_profit = 0.025 * max_profit
        micro_profit = 0.70 * max_profit
        price_reversal = get_price_reversal(symbol, side)

        # 3. Call detect_exit
        rsi_exit_decision = check_exit_conditions(symbol,rsi_overbought_zone,rsi_oversold_zone, side)

        # Calculate PnL
        if side == 'BUY':
            profit_relative = ((latest_close - entry_price) / entry_price) * 100
            loss_relative = ((entry_price - latest_close) / entry_price) * 100

            boll_reversal_profit = (
                (latest_close >= latest_upper_band) or 
                (latest_close <= latest_upper_band and previous_close > latest_upper_band)
            )
        elif side == 'SELL':
            profit_relative = ((entry_price - latest_close) / entry_price) * 100
            loss_relative = ((latest_close - entry_price) / entry_price) * 100

            boll_reversal_profit = (
                (latest_close <= latest_lower_band) or 
                (latest_close >= latest_lower_band and previous_close < latest_lower_band)
            )

        # Log PnL
        profit_label = '* * PROFIT' if profit_relative > 0 else '~ ~ LOSS'
        print(f"[Tracking PnL - v1.62] {symbol} {side} {profit_label} = {profit_relative:.2f}%")

        # Define a flag for whether the position has already been closed
        position_closed = False

        # 1. Take profit opportunity
        if profit_relative >= micro_profit:
            reason = 'Price > Micro Profit'
            close_position(symbol, side, profit_relative, reason)
            print(f"[* * * * CLOSED] {symbol} Closed due to micro profit > {micro_profit:.2f}% with profit of {profit_relative:.2f}%.")
            position_closed = True
            return {"close_position": True, "reason": "Micro Profit Met"}
        

        # 2. Meet Bollinger and RSI reversal conditions
        if profit_relative > rush_profit:
            if rsi_exit_decision['exit_signal']:
                print(f"[* * * * CLOSED] {symbol}: {side} due to RSI Exit Decission Met.")
                close_position(symbol, side, profit_relative, rsi_exit_decision["reason"])
                position_closed = True
                return {"close_position": True, "reason": rsi_exit_decision["reason"]}
            elif boll_reversal_profit:
                print(f"[* * * * CLOSED] {symbol}: {side} due to Bollinger Exit Decission Met.")
                reason = 'Bollinger Exit Decission Met'
                close_position(symbol, side, profit_relative, reason)
                position_closed = True
                return {"close_position": True, "reason": reason} 
            

        # 3. Check if profit but not yet max profit and reversal detected
        if min_profit <= profit_relative <= max_profit:
            exit_conditions = [
                (boll_reversal_profit, "Price crossed Bollinger Bands after profit"),
                (rsi_exit_decision['exit_signal'], rsi_exit_decision["reason"]),
                (price_reversal, "Profit > min profit with potential price reversal"),
            ]

            for condition, reason in exit_conditions:
                if condition and not position_closed:
                    close_position(symbol, side, profit_relative, reason)
                    print(f"[* * * * CLOSED] {symbol} due to {reason} at {latest_close:.2f}")
                    position_closed = True
                    return {"close_position": True, "reason": reason}

        # 4. Check if profit met max profit and no reversal detected
        if profit_relative >= max_profit and not position_closed:
            exit_conditions = [
                (boll_reversal_profit, "Price crossed Bollinger Bands after profit"),
                (rsi_exit_decision['exit_signal'], rsi_exit_decision["reason"]),
                (price_reversal, "Profit > max profit with potential price reversal"),
            ]

            for condition, reason in exit_conditions:
                if condition:
                    close_position(symbol, side, profit_relative, reason)
                    print(f"[* * * * CLOSED] {symbol} due to {reason} at {latest_close:.2f}")
                    position_closed = True
                    return {"close_position": True, "reason": reason}

        # No exit condition met
        if not position_closed:
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
    adx_df = pd_ta.adx(df['high'], df['low'], df['close'], length=9)  # ADX returns a DataFrame
    
    # Select only the ADX column from the resulting DataFrame
    df['ADX'] = adx_df['ADX_9']

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













