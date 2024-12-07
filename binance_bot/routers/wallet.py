# blueprints/wallet.py
import os
from flask import Blueprint, jsonify
from binance.client import Client
import certifi
from decimal import Decimal
from dconfig import read_db_config

# Initialize Binance client with your API key and secret
binance_key = read_db_config(section='user_credential')

# Configure your Binance API using environment variables for security
API_KEY = os.getenv('BINANCE_API_KEY', binance_key['api_key'])
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key['secret_key'])

# Initialize the Binance client with SSL verification
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

# Check if API credentials are properly loaded
if not API_KEY or not API_SECRET:
    print("Error: Binance API key and secret must be provided.")
    
# Create a Blueprint for wallet operations
wallet_bp = Blueprint('wallet', __name__, url_prefix='/api/wallet/')

import time

def get_total_wallet_balance(max_retries=10, delay=2):
    """
    Fetch the total USDT balance in the futures wallet (available + used margin + unrealized PnL).
    Retries until a valid balance is fetched or maximum retries are reached.

    Parameters:
        max_retries (int): Maximum number of retries.
        delay (int): Delay (in seconds) between retries.

    Returns:
        Decimal: The total balance of USDT in the futures wallet.
    """
    for attempt in range(max_retries):
        try:
            # Fetch futures account balance
            futures_account_info = client.futures_account_balance()
            
            # Find USDT asset and fetch its total balance
            usdt_info = next(
                (asset for asset in futures_account_info if asset['asset'] == 'USDT'),
                None
            )
            
            if usdt_info:
                # Extract and sum relevant fields for total balance
                total_balance = Decimal(usdt_info.get('balance', '0.0'))  # Total balance
                return total_balance
            else:
                print(f"USDT balance not found in futures account. Attempt {attempt + 1}/{max_retries}.")
        
        except Exception as e:
            print(f"Error fetching total wallet balance (Attempt {attempt + 1}/{max_retries}): {e}")
        
        # Wait before retrying
        time.sleep(delay)

    print("Failed to fetch USDT balance after maximum retries.")
    return Decimal('0.0')



def calculate_dynamic_safe_trade_amount(available_balance, num_symbols, two_sided):
    """
    Calculate a dynamic safe trade amount for each symbol, considering two-sided orders.
    
    :param available_balance: The available balance in the wallet.
    :param num_symbols: The number of symbols to trade.
    :param two_sided: Whether two-sided orders (buy and sell) are being placed for each symbol.
    :return: Calculated safe trade amount per symbol.
    """
    # Define the maximum safe percentage based on risk tolerance
    safety_percentage = Decimal('0.9')  # Example: 30% risk tolerance
    
    # Total safe trade amount for all symbols based on the safety percentage
    total_safe_trade_amount = (available_balance * safety_percentage)
    
    # If two-sided orders are considered, the allocation per side will be halved
    if two_sided:
        total_safe_trade_amount /= Decimal(2)
    
    # Calculate the safe trade amount per symbol
    safe_trade_amount_per_symbol = total_safe_trade_amount / Decimal(num_symbols)
    
    return safe_trade_amount_per_symbol

@wallet_bp.route('/safe_trade_amount', methods=['GET'])

def safe_trade_amount(num_symbols,two_side):
    """
    API endpoint to calculate a safe trade amount based on available balance and leverage.
    """
    try:
        # Get the available balance
        available_balance = get_total_wallet_balance()
        # Calculate safety percentage dynamically
        safe_trade_amount = calculate_dynamic_safe_trade_amount(available_balance, num_symbols, two_side)
        
        return safe_trade_amount
        
    except Exception as e:
        print(f"Error calculating safe trade amount: {e}")
        return 0.0
