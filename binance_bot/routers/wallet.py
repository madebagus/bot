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

def get_wallet_balance():
    """Fetch the current available USDT balance in the futures wallet."""
    try:
        # Fetch futures account balance
        futures_account_info = client.futures_account_balance()
        
        # Find USDT balance from futures account
        usdt_balance = next(
            (asset['balance'] for asset in futures_account_info if asset['asset'] == 'USDT'), None
        )
        
        if usdt_balance is not None:
            usdt_balance = Decimal(usdt_balance)
            return usdt_balance
        else:
            print("USDT balance not found in futures account.")
            return Decimal('0.0')
        
    except Exception as e:
        print(f"Error fetching futures wallet balance: {e}")
        return Decimal('0.0')
        
        
    except Exception as e:
        print(f"Error fetching wallet balance: {e}")
        # Default balance for testing or fallback if error occurs
        return Decimal('0.0')

def calculate_dynamic_safe_trade_amount(available_balance, num_symbols, two_sided=False):
    """
    Calculate a dynamic safe trade amount for each symbol, considering two-sided orders.
    
    :param available_balance: The available balance in the wallet.
    :param num_symbols: The number of symbols to trade.
    :param two_sided: Whether two-sided orders (buy and sell) are being placed for each symbol.
    :return: Calculated safe trade amount per symbol.
    """
    # Define the maximum safe percentage based on risk tolerance
    safety_percentage = Decimal('50.0')  # Example: 30% risk tolerance
    
    # Total safe trade amount for all symbols based on the safety percentage
    total_safe_trade_amount = (available_balance * safety_percentage) / Decimal(100)
    
    # If two-sided orders are considered, the allocation per side will be halved
    if two_sided:
        total_safe_trade_amount /= Decimal(2)
    
    # Calculate the safe trade amount per symbol
    safe_trade_amount_per_symbol = total_safe_trade_amount / Decimal(num_symbols)
    
    return safe_trade_amount_per_symbol

@wallet_bp.route('/safe_trade_amount', methods=['GET'])
def safe_trade_amount():
    """
    API endpoint to calculate a safe trade amount based on available balance and leverage.
    """
    try:
        # Leverage value to be used in the calculation (can be set dynamically if needed)
        leverage = Decimal('5')
        num_symbols = 5
        # Get the available balance
        available_balance = get_wallet_balance()
        
        # Calculate safety percentage dynamically
        safety_percentage = calculate_dynamic_safe_trade_amount(available_balance, num_symbols)
        
        # Calculate the safe trade amount
        safe_trade_amount = (available_balance * safety_percentage) / Decimal(100)
        
        return jsonify({
            "available_balance": str(available_balance),  # Convert to string for precise representation
            "leverage": str(leverage),
            "dynamic_safety_percentage": str(safety_percentage),
            "safe_trade_amount": str(safe_trade_amount)
        })
        
    except Exception as e:
        print(f"Error calculating safe trade amount: {e}")
        return jsonify({"error": "Could not calculate safe trade amount"}), 500
