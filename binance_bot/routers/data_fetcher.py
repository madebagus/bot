# data_fetcher.py
import ccxt
from datetime import datetime, timedelta
from conn_ssh import create_conn

# Initialize MySQL and Binance connections

conn, tunnel = create_conn()
cursor = conn.cursor()

binance = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1m'  # 1-minute data for high frequency

def fetch_historical_data():
    """Fetch 3 months of historical data."""
    since = binance.parse8601((datetime.now() - timedelta(days=90)).isoformat())
    while since < binance.milliseconds():
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if len(ohlcv) == 0:
            break
        for candle in ohlcv:
            print(f"Inserting data for {symbol} at {datetime.fromtimestamp(candle[0] / 1000)}")
            data = (
                symbol.replace("/", ""),
                datetime.fromtimestamp(candle[0] / 1000),
                candle[1],  # open
                candle[2],  # high
                candle[3],  # low
                candle[4],  # close
                candle[5]   # volume
            )
            cursor.execute(
                """
                INSERT IGNORE INTO historical_data (symbol, open_time, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                data
            )
        conn.commit()
        since = ohlcv[-1][0] + 1
        print("Data inserted and commit successful.")


def fetch_latest_data():
    """Fetch latest 1-minute data and update database."""
    # Query for the latest available entry in the database
    latest_time_query = "SELECT MAX(open_time) FROM historical_data WHERE symbol = %s"
    cursor.execute(latest_time_query, (symbol.replace("/", ""),))
    last_entry_time = cursor.fetchone()[0]

    # If no data exists, fetch initial historical data (last 3 months)
    if last_entry_time is None:
        print("No data found in the database, fetching historical data...")
        fetch_historical_data()  # Fetch historical data if nothing is present
        return  # Exit here if no data is found initially
    
    # Continue fetching new data if data already exists
    since = int(last_entry_time.timestamp() * 1000) + 1  # Start from the last entry time
    ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    for candle in ohlcv:
        data = (
            symbol.replace("/", ""),
            datetime.fromtimestamp(candle[0] / 1000),
            candle[1], candle[2], candle[3], candle[4], candle[5]
        )
        cursor.execute(
            """
            INSERT IGNORE INTO historical_data (symbol, open_time, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            data
        )
    conn.commit()
