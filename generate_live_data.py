#!/usr/bin/env python3
"""
Generate FRESH beacon data with current timestamps
This ensures data is within the detection window (last 5 minutes)
"""

import psycopg2
import configparser
from datetime import datetime, timedelta
import random

def generate_live_beacons():
    """Generate beacon data with timestamps starting NOW"""
    
    config = configparser.ConfigParser()
    config.read('config/database.conf')
    
    conn = psycopg2.connect(
        host=config['postgresql']['host'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password']
    )
    
    cursor = conn.cursor()
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║    GENERATING LIVE BEACON DATA (CURRENT TIME)         ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    # Clear old data
    print("[1/3] Clearing old test data...")
    cursor.execute("DELETE FROM conn_log WHERE uid LIKE 'beacon_%' OR uid LIKE 'test%'")
    conn.commit()
    print("      ✓ Cleared")
    
    print()
    print("[2/3] Generating LIVE beacon traffic (last 3 minutes)...")
    
    # Start from 3 minutes ago to NOW
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=3)
    
    total = 0
    
    # HIGH FREQUENCY BEACON - 10 second intervals
    print("\n      🔴 HIGH FREQ BEACON: 192.168.1.100 (every 10s)")
    beacon_count = 18  # 3 min = 18 beacons at 10s interval
    for i in range(beacon_count):
        ts = start_time + timedelta(seconds=i * 10 + random.uniform(-1, 1))
        
        cursor.execute("""
            INSERT INTO conn_log 
            (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
             proto, service, duration, orig_bytes, resp_bytes, conn_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ts,
            f'beacon_{random.randint(10000, 99999)}',
            '192.168.1.100',
            random.randint(49152, 65535),
            '185.220.101.5',  # Suspicious IP
            443,
            'tcp',
            'ssl',
            random.uniform(0.05, 0.2),
            128 + random.randint(-20, 20),
            256 + random.randint(-40, 40),
            'SF'
        ))
        total += 1
    
    print(f"      ✓ {beacon_count} connections")
    
    # MEDIUM FREQUENCY BEACON - 30 second intervals  
    print("\n      🟠 MED FREQ BEACON: 192.168.1.105 (every 30s)")
    beacon_count = 6  # 3 min = 6 beacons at 30s interval
    for i in range(beacon_count):
        ts = start_time + timedelta(seconds=i * 30 + random.uniform(-2, 2))
        
        cursor.execute("""
            INSERT INTO conn_log 
            (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
             proto, service, duration, orig_bytes, resp_bytes, conn_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ts,
            f'beacon_{random.randint(10000, 99999)}',
            '192.168.1.105',
            random.randint(49152, 65535),
            '45.142.212.61',  # Another suspicious IP
            80,
            'tcp',
            'http',
            random.uniform(0.1, 0.3),
            256 + random.randint(-20, 20),
            512 + random.randint(-40, 40),
            'SF'
        ))
        total += 1
    
    print(f"      ✓ {beacon_count} connections")
    
    # NORMAL TRAFFIC - Random intervals
    print("\n      🟢 NORMAL TRAFFIC: 192.168.1.50 (random)")
    beacon_count = 10
    for i in range(beacon_count):
        ts = start_time + timedelta(seconds=random.uniform(0, 180))
        
        cursor.execute("""
            INSERT INTO conn_log 
            (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
             proto, service, duration, orig_bytes, resp_bytes, conn_state)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ts,
            f'beacon_{random.randint(10000, 99999)}',
            '192.168.1.50',
            random.randint(49152, 65535),
            '8.8.8.8',
            53,
            'udp',
            'dns',
            random.uniform(0.01, 0.1),
            64 + random.randint(-10, 10),
            128 + random.randint(-20, 20),
            'SF'
        ))
        total += 1
    
    print(f"      ✓ {beacon_count} connections")
    
    conn.commit()
    
    print()
    print(f"      ✓ Total: {total} FRESH records")
    
    # Verify
    print()
    print("[3/3] Verifying...")
    cursor.execute("""
        SELECT COUNT(*), 
               MIN(ts) as oldest, 
               MAX(ts) as newest,
               NOW() - MAX(ts) as age
        FROM conn_log 
        WHERE ts > NOW() - INTERVAL '5 minutes'
    """)
    count, oldest, newest, age = cursor.fetchone()
    print(f"      ✓ {count} records in last 5 minutes")
    print(f"      ✓ Newest: {age} ago (CURRENT!)")
    
    conn.close()
    
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║           LIVE DATA READY - DETECTION ACTIVE!          ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print("Expected Results:")
    print()
    print("  🔴 192.168.1.100 → P-Score > 0.8  ⚠️  BEACON!")
    print("  🟠 192.168.1.105 → P-Score > 0.7  ⚠️  BEACON!")
    print("  🟢 192.168.1.50  → P-Score < 0.5  ✓  Normal")
    print()
    print("─" * 56)
    print()
    print("Dashboard: http://localhost:5000")
    print("Click 'Analyze Now' - Results will appear instantly!")
    print()

if __name__ == "__main__":
    try:
        generate_live_beacons()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
