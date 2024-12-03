import pymysql
from sshtunnel import SSHTunnelForwarder
from dconfig import read_db_config

def create_conn():
    # Read SSH and MySQL config from database.ini
    ssh_config = read_db_config(section='ssh')
    mysql_config = read_db_config(section='mysql_ssh')

    # Set up the SSH tunnel
    tunnel = SSHTunnelForwarder(
        (ssh_config['host'], int(ssh_config['port'])),
        ssh_username=ssh_config['user'],
        ssh_password=ssh_config['password'],
        remote_bind_address=(mysql_config['host'], int(mysql_config['port']))
    )
    
    tunnel.start()

    # Create a connection to the MySQL server via SSH
    conn = pymysql.connect(
        host='127.0.0.1',
        port=tunnel.local_bind_port,
        user=mysql_config['user'],
        password=mysql_config['password'],
        db=mysql_config['database'],
        connect_timeout=10,
        read_timeout=30,
        write_timeout=30
    )

    return conn, tunnel  # Return both the connection and tunnel object for reuse
