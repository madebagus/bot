from configparser import ConfigParser

def read_db_config(filename='database.ini', section=None):
    """read config generally"""
    # Create a parser object
    parser = ConfigParser(interpolation=None)
    # Read the config file
    parser.read(filename)
    
    # Get the section from the config file
    db_config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db_config[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in the {filename} file')
    
    return db_config


def read_twitter_config(filename='database.ini', section=None):
    """read config generally"""
    # Create a parser object
    parser = ConfigParser(interpolation=None)
    # Read the config file
    parser.read(filename)
    
    # Get the section from the config file
    db_config = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db_config[param[0]] = param[1]
    else:
        raise Exception(f'Section {section} not found in the {filename} file')
    
    return db_config

# Fetching the Twitter credentials from the config file
twitter_key = read_twitter_config(section='twitter_credential')

# Assigning values from the dictionary
TWITTER_CONSUMER_KEY = twitter_key.get('api_key')
TWITTER_CONSUMER_SECRET = twitter_key.get('secret_key')
TWITTER_ACCESS_TOKEN = twitter_key.get('token_key')
TWITTER_ACCESS_TOKEN_SECRET = twitter_key.get('token_secret')
TWITTER_BEARER_TOKEN = twitter_key['bearer_token']
NEWSAPI_KEY = twitter_key['news_api_key']
