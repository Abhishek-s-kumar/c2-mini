#!/usr/bin/env python3
"""
Complete C2 Detection Demo
Generates fresh beacon data and triggers analysis
"""

import psycopg2
import configparser
from datetime import datetime, timedelta
import random
import time
import requests

def generate_fresh_beacon_data():
    """Generate fresh beacon data in database"""
    
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
    print("║         C2 BEACON DETECTION - LIVE DEMO                ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    # Clear old data
    print("[1/4] Clearing old test data...")
    cursor.execute("DELETE FROM conn_log WHERE uid LIKE 'beacon_%' OR uid LIKE 'test%'")
    conn.commit()
    print("      ✓ Old data cleared")
    
    # Generate fresh beacon patterns
    print()
    print("[2/4] Generating fresh beacon traffic...")
    
    beacons = [
        {
            'name': '🔴 Malicious C2 Beacon',
            'source': '192.168.1.100',
            'dest': '185.220.101.5',
            'interval': 10,
            'count': 30,
            'bytes': 128,
            'jitter': 1
        },
        {
            'name': '🟠 Suspicious Callback',
            'source': '192.168.1.105', 
            'dest': '45.142.212.61',
            'interval': 30,
            'count': 20,
            'bytes': 256,
            'jitter': 2
        },
        {
            'name': '🟢 Normal DNS Traffic',
            'source': '192.168.1.50',
            'dest': '8.8.8.8',
            'interval': 0,  # Random
            'count': 15,
            'bytes': 64,
            'jitter': 60
        }
    ]
    
    total = 0
    base_time = datetime.now() - timedelta(minutes=3)
    
    for beacon in beacons:
        print(f"\n      {beacon['name']}")
        print(f"      {beacon['source']} → {beacon['dest']}")
        
        for i in range(beacon['count']):
            if beacon['interval'] == 0:
                # Random intervals
                offset = random.uniform(0, 180)
            else:
                # Periodic with jitter
                offset = (i * beacon['interval']) + random.uniform(-beacon['jitter'], beacon['jitter'])
            
            ts = base_time + timedelta(seconds=offset)
            
            cursor.execute("""
                INSERT INTO conn_log 
                (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, 
                 proto, service, duration, orig_bytes, resp_bytes, conn_state)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                ts,
                f'beacon_{random.randint(10000, 99999)}',
                beacon['source'],
                random.randint(49152, 65535),
                beacon['dest'],
                443 if 'Malicious' in beacon['name'] else (80 if 'Suspicious' in beacon['name'] else 53),
                'tcp' if beacon['interval'] > 0 else 'udp',
                'ssl' if 'Malicious' in beacon['name'] else ('http' if 'Suspicious' in beacon['name'] else 'dns'),
                random.uniform(0.05, 0.3),
                beacon['bytes'] + random.randint(-20, 20),
                beacon['bytes'] * 2 + random.randint(-40, 40),
                'SF'
            ))
            total += 1
        
        print(f"      ✓ {beacon['count']} connections")
    
    conn.commit()
    print()
    print(f"      ✓ Total: {total} records generated")
    
    # Verify
    print()
    print("[3/4] Verifying database...")
    cursor.execute("SELECT COUNT(*) FROM conn_log WHERE ts > NOW() - INTERVAL '10 minutes'")
    count = cursor.fetchone()[0]
    print(f"      ✓ {count} recent records in database")
    
    conn.close()
    
    # Trigger analysis
    print()
    print("[4/4] Triggering analysis...")
    try:
        response = requests.get('http://localhost:5000/api/analyze', timeout=10)
        if response.status_code == 200:
            print("      ✓ Analysis triggered successfully")
        else:
            print(f"      ⚠ Analysis returned status {response.status_code}")
    except Exception as e:
        print(f"      ⚠ Could not trigger analysis: {e}")
        print("      (Dashboard might not be running)")
    
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║              DEMO READY!                               ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print("Expected Detection Results:")
    print()
    print("  🔴 192.168.1.100 (10s interval)")
    print("     P-Score: > 0.8  →  ⚠️  BEACON DETECTED")
    print()
    print("  🟠 192.168.1.105 (30s interval)")
    print("     P-Score: > 0.7  →  ⚠️  BEACON DETECTED")
    print()
    print("  🟢 192.168.1.50 (random DNS)")
    print("     P-Score: < 0.5  →  ✓  Normal Traffic")
    print()
    print("─" * 56)
    print()
    print("Open dashboard: http://localhost:5000")
    print("Click 'Analyze Now' to see detections!")
    print()

if __name__ == "__main__":
    try:
        generate_fresh_beacon_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is PostgreSQL running? sudo systemctl start postgresql")
        print("  2. Is the database configured? Check config/database.conf")
        print("  3. Is the dashboard running? ./start_system.sh")
