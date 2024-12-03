"""connection using engine for pandas operation"""
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from dconfig import read_db_config

# Get All Config SSH and DB
ssh_config = read_db_config(section='ssh')
mysql_config = read_db_config(section='mysql_ssh')


# Class to manage the SSH tunnel and database engine
class DBConnection:
    """inisiate connection"""
    def __init__(self):
        # Set up SSH Tunnel
        self.tunnel = SSHTunnelForwarder(
            (ssh_config['host'], int(ssh_config['port'])),
            ssh_username=ssh_config['user'],
            ssh_password=ssh_config['password'],
            remote_bind_address=(mysql_config['host'], int(mysql_config['port']))
        )

    def start_tunnel(self):
        """start ssh tunnel"""
        self.tunnel.start()

    def stop_tunnel(self):
        """stop ssh tunnel"""
        self.tunnel.stop()

    def get_engine(self):
        """db connection engine"""
        local_port = self.tunnel.local_bind_port
        engine = create_engine(f'mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@{'127.0.0.1'}:{local_port}/{mysql_config['database']}',
                               pool_recycle=3600,pool_size=10,max_overflow=20)
      

        return engine

# Singleton instance of DatabaseConnection
conn = DBConnection()

def get_db_engine():
    """get connected"""
    # Start the SSH tunnel and return the SQLAlchemy engine
    if not conn.tunnel.is_active:
        conn.start_tunnel()
    return conn.get_engine()
