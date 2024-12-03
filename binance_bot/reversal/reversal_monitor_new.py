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

# Function to fetch all open positions
def get_open_positions():
    try:
        positions = client.futures_account()['positions']
        #(f"Fetched positions: {positions}")
    except Exception as e:
        print(f"Error fetching positions: {e}")
        return []
    
    open_positions = []
    for pos in positions:
        try:
            position_amt = float(pos['positionAmt'])
            if position_amt != 0:
                #print(f"Adding position: {pos}")
                open_positions.append({
                    'symbol': pos['symbol'],
                    'positionAmt': position_amt,
                    'side': 'BUY' if position_amt > 0 else 'SELL'
                })
        except (ValueError, KeyError, TypeError) as e:
            print(f"Error processing position: {e}, Data: {pos}")
    #print(f"Open positions: {open_positions}")
    return open_positions


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


def calculate_pnl(entry_price, current_price, side, exp_profit, exp_loss):
    """
    Calculate the percentage of realized profit or loss compared to the expected profit or loss.

    Parameters:
    - entry_price: Decimal, price at which the position was opened.
    - current_price: Decimal, current market price of the asset.
    - side: str, 'BUY' or 'SELL' to indicate the trade direction.
    - exp_profit: Decimal, expected profit in USDT.
    - exp_loss: Decimal, expected loss in USDT.

    Returns:
    - profit_relative: float, percentage of profit realized compared to expected profit.
    - loss_relative: float, percentage of loss realized compared to expected loss.
    """
    # Ensure expected profit and loss are positive
    if exp_profit <= 0 or exp_loss <= 0:
        raise ValueError("Expected profit and loss must be greater than zero.")

    # Initialize relative profit and loss percentages
    profit_relative = 0
    loss_relative = 0

    if side == 'BUY':
        # Calculate profit for BUY position
        if current_price > entry_price:
            current_profit = current_price - entry_price
            profit_relative = (current_profit / exp_profit) * 100

        # Calculate loss for BUY position
        elif current_price < entry_price:
            current_loss = entry_price - current_price
            loss_relative = (current_loss / exp_loss) * 100

    elif side == 'SELL':
        # Calculate profit for SELL position
        if current_price < entry_price:
            current_profit = entry_price - current_price
            profit_relative = (current_profit / exp_profit) * 100

        # Calculate loss for SELL position
        elif current_price > entry_price:
            current_loss = current_price - entry_price
            loss_relative = (current_loss / exp_loss) * 100

    else:
        raise ValueError("Invalid side. Side must be 'BUY' or 'SELL'.")

    return profit_relative, loss_relative

        

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

def track_trade(
    symbol,
    side,
    amount,
    entry_price,
    stop_loss,
    take_profit,
    max_loss=0.5,
    min_profit=0.5,
    bollinger_window=20,  # Bollinger Bands window
    bollinger_std_dev=2  # Bollinger Bands standard deviation
):
    """
    Monitor price movements and handle dynamic closing conditions, including Bollinger Bands.
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

        # Ensure Bollinger Bands are valid
        if df[['upper_band', 'lower_band']].isna().all().any():
            print(f"Bollinger Bands not available for {symbol}.")
            return {"close_position": False, "reason": "Insufficient Bollinger Band data"}

        # Fetch latest price and Bollinger Band values
        latest_close = get_current_price(symbol)
        latest_upper_band = df['upper_band'].iloc[-1]
        latest_lower_band = df['lower_band'].iloc[-1]

        # Calculate PnL
        profit_relative, loss_relative = calculate_pnl(entry_price, latest_close, side, take_profit, stop_loss)
        print(f"[PnL Check] {symbol} {side}: Current={latest_close}, Profit={profit_relative}, Loss={loss_relative}")

        # Closing logic
        if side == 'BUY':
            if latest_close >= latest_upper_band:
                close_position(symbol, side)
                print(f"[* * * close_position] {symbol} {side} Price touched UPPER BAND at {latest_close}")
                return {"close_position": True, "reason": "Price touched upper Bollinger Band"}
            if loss_relative >= max_loss:
                close_position(symbol, side)
                print(f"[* * * close_position] {symbol} {side} Max loss met at -{loss_relative}%")
                return {"close_position": True, "reason": "Max loss reached"}
            if profit_relative >= min_profit:
                close_position(symbol, side)
                print(f"[* * * close_position] {symbol} {side} Target profit met at {profit_relative}%")
                return {"close_position": True, "reason": "Profit target reached"}

        elif side == 'SELL':
            if latest_close <= latest_lower_band:
                close_position(symbol, side)
                print(f"[* * * close_position] {symbol} {side} Price touched LOWER BAND at {latest_close}")
                return {"close_position": True, "reason": "Price touched lower Bollinger Band"}
            if loss_relative >= max_loss:
                close_position(symbol, side)
                print(f"[* * * close_position] {symbol} {side} Max loss met at -{loss_relative}%")
                return {"close_position": True, "reason": "Max loss reached"}
            if profit_relative >= min_profit:
                close_position(symbol, side)
                print(f"[* * * close_position] {symbol} {side} Target profit met at {profit_relative}%")
                return {"close_position": True, "reason": "Profit target reached"}

    except Exception as e:
        print(f"Error tracking trade for {symbol}: {e}")
        return {"close_position": False, "reason": "Error"}

    return {"close_position": False, "reason": "No conditions met"}



def track_trade_by_bollinger(symbol, side, leverage):
    """
    Track an ongoing trade and dynamically exit based on Bollinger Bands criteria.

    :param symbol: The trading symbol (e.g., 'BTCUSDT').
    :param side: The side of the trade ('BUY' or 'SELL').
    :param leverage: Leverage used for the trade.
    """
    try:
        while True:
            # Fetch latest data
            df = get_data(symbol)
            
            if df is not None:
                # Calculate Bollinger Bands
                df['mean'] = df['close'].rolling(window=20).mean()
                df['stddev'] = df['close'].rolling(window=20).std()
                df['upper_band'] = df['mean'] + (2 * df['stddev'])
                df['lower_band'] = df['mean'] - (2 * df['stddev'])

                # Get the latest closing price and bands
                latest_close = df['close'].iloc[-1]
                upper_band = df['upper_band'].iloc[-1]
                lower_band = df['lower_band'].iloc[-1]

                print(f"[{symbol}] Latest Close: {latest_close}, Upper Band: {upper_band}, Lower Band: {lower_band}")

                # Close condition for BUY
                if side.upper() == 'BUY' and latest_close >= upper_band:
                    print(f"[{symbol}] Closing BUY trade: Price reached the upper Bollinger Band.")
                    close_trade(symbol, side, leverage)
                    break

                # Close condition for SELL
                elif side.upper() == 'SELL' and latest_close <= lower_band:
                    print(f"[{symbol}] Closing SELL trade: Price reached the lower Bollinger Band.")
                    close_trade(symbol, side, leverage)
                    break

            else:
                print(f"[{symbol}] Failed to fetch data. Retrying...")

            time.sleep(5)  # Monitor at regular intervals

    except Exception as e:
        print(f"Error in track_trade for {symbol}: {e}")


# Get stop loss target
# Combined Function with Side Filtering
def get_order_targets(symbol, side):
    """
    Fetch the stop loss and take profit targets for a given symbol and position side.
    
    :param symbol: The trading symbol (e.g., 'BTCUSDT').
    :param side: The side of the position ('BUY' for LONG or 'SELL' for SHORT).
    :return: stop_loss, take_profit (defaults to 0 if not available).
    """
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        stop_loss = 0
        take_profit = 0
        
        # Map 'BUY'/'SELL' to 'LONG'/'SHORT' for positionSide
        position_side = 'LONG' if side.upper() == 'BUY' else 'SHORT'
        
        for order in open_orders:
            # Ensure the order is associated with the correct position side
            if order.get('positionSide', '').upper() != position_side:
                continue  # Skip orders not matching the position side
            
            if order.get('reduceOnly', False):
                if order.get('type') == 'STOP_MARKET':  # Stop-loss order
                    stop_loss = float(order.get('stopPrice', 0))
                elif order.get('type') == 'TAKE_PROFIT_MARKET':  # Take-profit order
                    take_profit = float(order.get('stopPrice', 0))
        
        return stop_loss, take_profit
    except Exception as e:
        print(f"Error fetching targets for {symbol} [{side}]: {e}")
        return 0, 0

# Updated get_position_details Function
import time

def get_position_details(symbol, side, retries=5, delay=2):
    """
    Get the entry price, stop loss, and take profit for a given position, with retry logic.
    
    :param symbol: The trading symbol (e.g., 'BTCUSDT').
    :param side: The side of the position ('BUY' for LONG or 'SELL' for SHORT).
    :param retries: Maximum number of retries.
    :param delay: Delay in seconds between retries.
    :return: entry_price, stop_loss, take_profit (defaults to 0 if not available).
    """
    attempt = 0

    while attempt < retries:
        try:
            #print(f"Attempt {attempt + 1} to get position details for {symbol} [{side}]...")
            # Fetch account positions
            positions = client.futures_account()['positions']
            for pos in positions:
                # Find the relevant position by symbol
                if pos['symbol'] == symbol and float(pos['positionAmt']) != 0:
                    # Determine the actual side (BUY or SELL) based on positionAmt
                    actual_side = "BUY" if float(pos['positionAmt']) > 0 else "SELL"
                    if actual_side == side.upper():
                        entry_price = float(pos.get('entryPrice', 0))  # Get the entry price
                        stop_loss, take_profit = get_order_targets(symbol, actual_side)  # Fetch targets
                        #print(f"Position details fetched successfully for {symbol} [{side}].")
                        return entry_price, stop_loss, take_profit

            print(f"No position found for {symbol} [{side}].")
            return 0, 0, 0  # If no matching position found

        except Exception as e:
            print(f"Error getting position details for {symbol} [{side}]: {e}")
            attempt += 1
            if attempt < retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Failed to fetch position details for {symbol} [{side}] after {retries} attempts.")
                return 0, 0, 0


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
                amount = order['positionAmt']

                print(f"[PROFIT MONITORING] Monitoring {symbol} {side} position with amount: {amount}")

                # Fetch position details
                entry_price, stop_loss, take_profit = get_position_details(symbol, side)

                if entry_price == 0 or stop_loss == 0 or take_profit == 0:
                    print(f"[INCOMPLETE POSITION] Missing details for {symbol}: "
                          f"Entry price: {entry_price}, Stop loss: {stop_loss}, Take profit: {take_profit}")
                    continue

                # Debug print for position details
                #print(f"[DEBUG] {symbol} | Side: {side} | Entry: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")

                # Monitor the trade
                track_trade(symbol, side, amount, entry_price, stop_loss, take_profit)

        except Exception as e:
            print(f"[ERROR] Error in monitoring positions: {e}")
        finally:
            # Adjust based on API rate limits
            time.sleep(10)










