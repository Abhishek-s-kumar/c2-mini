#!/usr/bin/env python3
"""
Generate Synthetic Beacon Data for Testing
Creates realistic C2 beacon patterns in the database
"""

import psycopg2
import configparser
from datetime import datetime, timedelta
import random

def generate_beacon_traffic():
    """Generate synthetic beacon traffic with periodic patterns"""
    
    # Read config
    config = configparser.ConfigParser()
    config.read('config/database.conf')
    
    # Connect to database
    conn = psycopg2.connect(
        host=config['postgresql']['host'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password']
    )
    
    cursor = conn.cursor()
    
    print("╔════════════════════════════════════════════════════════╗")
    print("║     SYNTHETIC BEACON DATA GENERATOR                   ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    
    # Generate beacon patterns
    beacon_configs = [
        {
            'name': 'Malicious Beacon (10s interval)',
            'source_ip': '192.168.1.100',
            'dest_ip': '185.220.101.5',  # Suspicious IP
            'interval': 10,  # seconds
            'count': 30,
            'bytes': 128,
            'jitter': 1  # +/- 1 second
        },
        {
            'name': 'C2 Callback (30s interval)',
            'source_ip': '192.168.1.105',
            'dest_ip': '45.142.212.61',  # Another suspicious IP
            'interval': 30,
            'count': 20,
            'bytes': 256,
            'jitter': 2
        },
        {
            'name': 'Normal DNS Traffic',
            'source_ip': '192.168.1.50',
            'dest_ip': '8.8.8.8',
            'interval': random.randint(5, 60),  # Random intervals
            'count': 15,
            'bytes': 64,
            'jitter': 30  # Lots of jitter (normal behavior)
        }
    ]
    
    total_records = 0
    
    for beacon in beacon_configs:
        print(f"\n[+] Generating: {beacon['name']}")
        print(f"    Source: {beacon['source_ip']} → {beacon['dest_ip']}")
        print(f"    Pattern: Every {beacon['interval']}s (±{beacon['jitter']}s), {beacon['count']} connections")
        
        base_time = datetime.now() - timedelta(minutes=5)
        
        for i in range(beacon['count']):
            # Calculate timestamp with jitter
            jitter = random.uniform(-beacon['jitter'], beacon['jitter'])
            timestamp = base_time + timedelta(seconds=(i * beacon['interval'] + jitter))
            
            # Random but consistent packet sizes
            orig_bytes = beacon['bytes'] + random.randint(-20, 20)
            resp_bytes = beacon['bytes'] * 2 + random.randint(-40, 40)
            
            # Insert into database
            cursor.execute("""
                INSERT INTO conn_log 
                (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, service, 
                 duration, orig_bytes, resp_bytes, conn_state)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                timestamp,
                f'beacon_{i}_{random.randint(1000, 9999)}',
                beacon['source_ip'],
                random.randint(49152, 65535),  # Ephemeral port
                beacon['dest_ip'],
                443 if 'Malicious' in beacon['name'] else 53,
                'tcp' if 'Malicious' in beacon['name'] else 'udp',
                'ssl' if 'Malicious' in beacon['name'] else 'dns',
                random.uniform(0.05, 0.2),
                orig_bytes,
                resp_bytes,
                'SF'
            ))
            
            total_records += 1
        
        print(f"    ✓ Generated {beacon['count']} connections")
    
    # Commit all inserts
    conn.commit()
    
    print()
    print("═" * 56)
    print(f"✓ Successfully generated {total_records} synthetic records")
    print("═" * 56)
    print()
    
    # Verify data
    cursor.execute("SELECT COUNT(*) FROM conn_log WHERE ts > NOW() - INTERVAL '10 minutes'")
    recent_count = cursor.fetchone()[0]
    print(f"Database now contains {recent_count} recent records")
    print()
    
    # Show sample
    print("Sample connections:")
    cursor.execute("""
        SELECT ts, id_orig_h, id_resp_h, orig_bytes, resp_bytes 
        FROM conn_log 
        WHERE ts > NOW() - INTERVAL '10 minutes'
        ORDER BY ts DESC 
        LIMIT 5
    """)
    
    for row in cursor.fetchall():
        print(f"  {row[0]} | {row[1]} → {row[2]} | {row[3]}↑ {row[4]}↓ bytes")
    
    conn.close()
    
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║              NEXT STEPS                                ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    print("1. Check the dashboard: http://localhost:5000")
    print("2. Click 'Analyze Now' button")
    print("3. Look for beacons from 192.168.1.100 and 192.168.1.105")
    print()
    print("Expected Results:")
    print("  • 192.168.1.100 (10s beacon)  → P-Score > 0.8 ⚠️ DETECTED")
    print("  • 192.168.1.105 (30s beacon)  → P-Score > 0.7 ⚠️ DETECTED")
    print("  • 192.168.1.50  (random DNS)  → P-Score < 0.5 ✓ Normal")
    print()

if __name__ == "__main__":
    try:
        generate_beacon_traffic()
    except Exception as e:
        print(f"\n[ERROR] Failed to generate data: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. Database is properly configured")
        print("  3. config/database.conf exists")
