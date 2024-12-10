import os
import requests
from dconfig import read_db_config 

tele_config = read_db_config(section='telegram_cedential')
# Replace with your bot token and chat ID

BOT_TOKEN = '7655250313:AAEhoj_J3Gy8m-kFgsbofUeVr6sNaGWUrJI'
CHAT_ID = '1111135002'

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"  # Optional: Enables formatting (e.g., bold, italics)
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Telegram message sent successfully!")
    else:
        print(f"Failed to send message: {response.status_code}, {response.text}")



# Function to send a status message to Telegram
def send_status_message():
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": "...I Am Alive [v2.2.2]..."
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Status message sent successfully!")
        else:
            print(f"Failed to send status message: {response.text}")
    except Exception as e:
        print(f"Error sending status message: {e}")