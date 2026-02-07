#!/usr/bin/env python3
"""
Generate PERFECT beacon data to guarantee detection.
Removes all randomness to maximize P-Scores.
"""

import psycopg2
import configparser
from datetime import datetime, timedelta
import sys

def generate_perfect_data():
    config = configparser.ConfigParser()
    config.read('config/database.conf')
    
    conn = psycopg2.connect(
        host=config['postgresql']['host'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password']
    )
    conn.autocommit = True
    cursor = conn.cursor()
    
    print("🧹 Clearing old data...")
    cursor.execute("DELETE FROM conn_log WHERE uid LIKE 'beacon_%' OR uid LIKE 'test%'")
    
    print("🚀 Generating PERFECT beacon patterns (guaranteed detection)...")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=4) # 4 minutes of data
    
    total = 0
    
    # 1. PERFECT BEACON (Target: P-Score > 0.9)
    # 192.168.1.200 -> Every 5.0 seconds exactly, 512 bytes exactly
    print("   🔴 192.168.1.200: Perfect 5s interval (should trigger ALERT)")
    for i in range(48): # 4 minutes / 5s = 48 beacons
        ts = start_time + timedelta(seconds=i * 5.0)
        cursor.execute("""
            INSERT INTO conn_log 
            (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
             proto, service, duration, orig_bytes, resp_bytes, conn_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ts, f'beacon_p_{i}', '192.168.1.200', 50000, 
            '10.0.0.1', 443, 'tcp', 'ssl',
            0.1, 100, 512, 'SF' # EXACTLY 512 bytes response every time
        ))
        total += 1

    # 2. STRONG BEACON (Target: P-Score > 0.8)
    # 192.168.1.201 -> Every 10.0 seconds exactly, 1024 bytes
    print("   🟠 192.168.1.201: Perfect 10s interval (should trigger ALERT)")
    for i in range(24): # 4 minutes / 10s = 24 beacons
        ts = start_time + timedelta(seconds=i * 10.0)
        cursor.execute("""
            INSERT INTO conn_log 
            (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
             proto, service, duration, orig_bytes, resp_bytes, conn_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ts, f'beacon_s_{i}', '192.168.1.201', 50001, 
            '10.0.0.2', 80, 'tcp', 'http',
            0.1, 200, 1024, 'SF' # EXACTLY 1024 bytes
        ))
        total += 1

    print(f"\n✅ Inserted {total} records with CURRENT timestamps.")
    print("These patterns are mathematically perfect and WILL match the threshold.")
    
    # Verify age of data
    cursor.execute("SELECT NOW() - MAX(ts) FROM conn_log WHERE id_orig_h = '192.168.1.200'")
    age = cursor.fetchone()[0]
    print(f"Data Age: {age} (Should be < 5 seconds)")
    
    conn.close()

if __name__ == "__main__":
    generate_perfect_data()
