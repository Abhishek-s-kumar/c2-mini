#!/usr/bin/env python3
"""
Import Zeek conn.log files into PostgreSQL
"""

import pandas as pd
import psycopg2
import sys
import os
from datetime import datetime
from configparser import ConfigParser

def import_zeek_log(log_file):
    """Import a Zeek conn.log file into PostgreSQL"""
    
    # Read the log file
    try:
        df = pd.read_csv(
            log_file,
            sep='\t',
            comment='#',
            header=None,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            names=[
                'ts', 'uid', 'id_orig_h', 'id_orig_p', 'id_resh_h', 'id_resp_p',
                'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
                'conn_state', 'local_orig', 'local_resp'
            ]
        )
        
        # Convert timestamp
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
        
        # Fill missing values
        df.fillna({
            'service': '',
            'duration': 0,
            'orig_bytes': 0,
            'resp_bytes': 0,
            'conn_state': ''
        }, inplace=True)
        
        print(f"[+] Loaded {len(df)} records from {log_file}")
        return df
        
    except Exception as e:
        print(f"[-] Failed to read {log_file}: {e}")
        return None

def save_to_database(df):
    """Save DataFrame to PostgreSQL"""
    config = ConfigParser()
    config.read('config/database.conf')
    
    try:
        conn = psycopg2.connect(
            host=config['postgresql']['host'],
            database=config['postgresql']['database'],
            user=config['postgresql']['user'],
            password=config['postgresql']['password']
        )
        
        # Use pandas to_sql with chunksize for large files
        df.to_sql('conn_log', conn, if_exists='append', index=False, method='multi', chunksize=1000)
        
        print(f"[+] Successfully imported {len(df)} records to database")
        conn.close()
        return True
        
    except Exception as e:
        print(f"[-] Database import failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 import_zeek_logs.py <conn.log file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not os.path.exists(log_file):
        print(f"[-] File not found: {log_file}")
        sys.exit(1)
    
    print(f"[*] Importing {log_file}...")
    df = import_zeek_log(log_file)
    
    if df is not None and not df.empty:
        save_to_database(df)
