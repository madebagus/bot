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
def get_data(symbol, interval):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=50)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                       'close_time', 'qav', 'trades', 'tbav', 'tbqav', 'ignore'])
    df['close'] = pd.to_numeric(df['close'])
    df['ema_short'] = ta.trend.ema_indicator(df['close'], window=7)
    df['ema_long'] = ta.trend.ema_indicator(df['close'], window=14)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    return df

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


# Function to calculate profit/loss percentage
def calculate_pnl(entry_price, current_price, side, exp_profit, exp_loss):
    if side == 'BUY':
        if current_price > entry_price:
            current_profit = current_price - entry_price
            profit_relative = current_profit/exp_profit
        else:
            profit_relative = 0
        
        if current_price < entry_price:
            current_loss = entry_price - current_price
            loss_relative = current_loss/exp_loss
        else:
            loss_relative = 0

    elif side == 'SELL':
        if current_price < entry_price:
            current_profit = entry_price - current_price
            profit_relative = current_profit/exp_profit
        else:
            profit_relative = 0
        
        if current_price > entry_price:
            current_loss = current_price - entry_price
            loss_relative = current_loss/exp_loss
        else:
            loss_relative = 0

    return profit_relative,loss_relative
        

# Function to track position dynamically
def track_trade(symbol, side, amount, entry_price, stop_loss, take_profit, max_loss=0.6, min_profit=0.45, rsi_overbought=70, rsi_oversold=30):
    """
    Monitor price movements and handle dynamic closing conditions.
    """ 
    # Calculate expected profit or loss by position result, with checks for None
    if side == 'BUY':
        # Ensure take_profit and stop_loss are valid numbers
        exp_profit = take_profit - entry_price if take_profit else 0
        exp_loss = entry_price - stop_loss if stop_loss else 0
    elif side == 'SELL':
        # Ensure take_profit and stop_loss are valid numbers
        exp_profit = entry_price - take_profit if take_profit else 0
        exp_loss = stop_loss - entry_price if stop_loss else 0

    # Fetch data
    df = get_data(symbol, '3m')
    if df is None or df.empty:
        print(f"No data available for {symbol}")
        return {"close_position": False, "reason": "No data"}

    # Latest price and PnL
    latest_close = get_current_price(symbol)  # current price
    profit_relative, loss_relative = calculate_pnl(entry_price, latest_close, side, exp_profit, exp_loss)
    

    print(f"[** Profit Check **] for {symbol} {side}: current_price: {latest_close}, profit_relative: {profit_relative}, loss_relative: {loss_relative}") 

    # 1. Maximum Loss Condition
    #if loss_relative <= max_loss:
    if loss_relative >= max_loss:
        if side == 'BUY':
            close_position(symbol, side)
            #update_order_closing(symbol, side, latest_close)
            print(f"[LOSS CLOSING] Closing {symbol} {side} in LOSS : {profit_relative}")
            return {"close_position": True, "reason": "RSI Reversal (Overbought)"}

        elif side == 'SELL':
            close_position(symbol, side)
            #update_order_closing(symbol, side, latest_close)
            print(f"[LOSS CLOSING] Closing {symbol} {side} in LOSS : {profit_relative}")
            return {"close_position": True, "reason": "RSI Reversal (Oversold)"}   

    # 2. Profit Reversal Condition
    if profit_relative >= min_profit:
        if side == 'BUY':
            close_position(symbol, side)
            #update_order_closing(symbol, side, latest_close)
            print(f"[PROFIT CLOSING] Closing {symbol} {side} in PROFIT : {profit_relative}")
            return {"close_position": True, "reason": "RSI Reversal (Overbought)"}

        elif side == 'SELL':
            close_position(symbol, side)
            #update_order_closing(symbol, side, latest_close)
            print(f"[PROFIT CLOSING] Closing {symbol} {side} in PROFIT : {profit_relative}")
            return {"close_position": True, "reason": "RSI Reversal (Oversold)"}

    # No conditions met
    return {"close_position": False, "reason": "No conditions met"}




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
def get_position_details(symbol, side):
    """
    Get the entry price, stop loss, and take profit for a given position.
    
    :param symbol: The trading symbol (e.g., 'BTCUSDT').
    :param side: The side of the position ('BUY' for LONG or 'SELL' for SHORT).
    :return: entry_price, stop_loss, take_profit (defaults to 0 if not available).
    """
    try:
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
                    return entry_price, stop_loss, take_profit
        return 0, 0, 0  # If no position found
    except Exception as e:
        print(f"Error getting position details for {symbol} [{side}]: {e}")
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

                if entry_price is None or stop_loss is None or take_profit is None:
                    print(f"[INCOMPLETE POSITION] Missing details for {symbol}: "
                          f"Entry price: {entry_price}, Stop loss: {stop_loss}, Take profit: {take_profit}")
                    continue

                # Debug print for position details
                print(f"[DEBUG] {symbol} | Side: {side} | Entry: {entry_price}, Stop Loss: {stop_loss}, Take Profit: {take_profit}")

                # Monitor the trade
                track_trade(symbol, side, amount, entry_price, stop_loss, take_profit)

        except Exception as e:
            print(f"[ERROR] Error in monitoring positions: {e}")
        finally:
            # Adjust based on API rate limits
            time.sleep(10)










