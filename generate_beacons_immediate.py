#!/usr/bin/env python3
"""
Quick Beacon Data Generator - Generates fresh data immediately
"""

import psycopg2
import configparser
from datetime import datetime, timedelta
import random

def quick_generate():
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
    
    print("Generating FRESH beacon data...")
    
    # Clear old test data
    cursor.execute("DELETE FROM conn_log WHERE uid LIKE 'beacon_%' OR uid LIKE 'test%'")
    print("✓ Cleared old data")
    
    # Start from 3 minutes ago
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=3)
    
    total = 0
    
    # HIGH FREQUENCY BEACON - 10 second intervals (18 beacons)
    print("\n🔴 Generating HIGH FREQ BEACON: 192.168.1.100 (18 connections)")
    for i in range(18):
        ts = start_time + timedelta(seconds=i * 10 + random.uniform(-1, 1))
        cursor.execute("""
            INSERT INTO conn_log 
            (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
             proto, service, duration, orig_bytes, resp_bytes, conn_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ts, f'beacon_{random.randint(10000, 99999)}', '192.168.1.100',
            random.randint(49152, 65535), '185.220.101.5', 443, 'tcp', 'ssl',
            random.uniform(0.05, 0.2), 128 + random.randint(-20, 20),
            256 + random.randint(-40, 40), 'SF'
        ))
        total += 1
    print(f"✓ Generated {18} connections")
    
    # MEDIUM FREQUENCY BEACON - 20 second intervals (12 beacons to ensure > 10)
    print("\n🟠 Generating MED FREQ BEACON: 192.168.1.105 (12 connections)")
    for i in range(12):
        ts = start_time + timedelta(seconds=i * 15 + random.uniform(-2, 2))
        cursor.execute("""
            INSERT INTO conn_log 
            (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
             proto, service, duration, orig_bytes, resp_bytes, conn_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ts, f'beacon_{random.randint(10000, 99999)}', '192.168.1.105',
            random.randint(49152, 65535), '45.142.212.61', 80, 'tcp', 'http',
            random.uniform(0.1, 0.3), 256 + random.randint(-20, 20),
            512 + random.randint(-40, 40), 'SF'
        ))
        total += 1
    print(f"✓ Generated {12} connections")
    
    # NORMAL TRAFFIC - Random intervals (11 connections to meet minimum)
    print("\n🟢 Generating NORMAL TRAFFIC: 192.168.1.50 (11 connections)")
    for i in range(11):
        ts = start_time + timedelta(seconds=random.uniform(0, 180))
        cursor.execute("""
            INSERT INTO conn_log 
            (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
             proto, service, duration, orig_bytes, resp_bytes, conn_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ts, f'beacon_{random.randint(10000, 99999)}', '192.168.1.50',
            random.randint(49152, 65535), '8.8.8.8', 53, 'udp', 'dns',
            random.uniform(0.01, 0.1), 64 + random.randint(-10, 10),
            128 + random.randint(-20, 20), 'SF'
        ))
        total += 1
    print(f"✓ Generated {11} connections")
    
    print(f"\n✅ TOTAL: {total} fresh records inserted")
    
    # Verify
    cursor.execute("""
        SELECT id_orig_h, COUNT(*) as cnt 
        FROM conn_log 
        WHERE ts > NOW() - INTERVAL '5 minutes' 
        GROUP BY id_orig_h 
        ORDER BY cnt DESC
    """)
    
    print("\n📊 Verification:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} connections")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✅ DATA READY FOR DETECTION!")
    print("="*60)
    print("\nNext step:")
    print("1. Go to http://localhost:5000")
    print("2. Click 'Analyze Now'")
    print("3. See beacon detections!")

if __name__ == "__main__":
    try:
        quick_generate()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
