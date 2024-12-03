import os
import certifi
from flask import Blueprint, request, jsonify
from binance.client import Client
import requests
from dconfig import read_db_config
import time

# Initialize the blueprint for the proxy
binance_proxy_bp = Blueprint('proxy', __name__, url_prefix='/api/proxy')

# Load environment variables from a .env file
binance_key = read_db_config()

# Configure your Binance API using environment variables or fallback config
API_KEY = os.getenv('BINANCE_API_KEY', binance_key.get('api_key'))
API_SECRET = os.getenv('BINANCE_API_SECRET', binance_key.get('secret_key'))

# Initialize the Binance client with SSL verification
client = Client(API_KEY, API_SECRET, {"verify": certifi.where()})

@binance_proxy_bp.route('/start', methods=['GET', 'POST'])
def binance_proxy():
    # Get the endpoint and parameters from the incoming request
    endpoint = request.args.get('endpoint')
    params = request.args.to_dict(flat=False)  # Get all query parameters

    if not endpoint:
        return jsonify({"error": "Endpoint is required"}), 400
    
    # Remove the 'endpoint' key from params
    params.pop('endpoint', None)

    # Send the request to the Binance API
    url = f'https://api.binance.com/api/v3/{endpoint}'  # Adjust API version/endpoint as needed
    
    try:
        # Forward the request to Binance
        if request.method == 'GET':
            response = requests.get(url, params=params, headers={'X-MBX-APIKEY': API_KEY})
        elif request.method == 'POST':
            response = requests.post(url, data=params, headers={'X-MBX-APIKEY': API_KEY})
        
        # Check the response
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({"error": "Binance API request failed", "details": response.json()}), response.status_code
    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Request to Binance failed", "details": str(e)}), 500
