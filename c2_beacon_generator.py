#!/usr/bin/env python3
"""
C2 Beacon Simulation Script
Generates periodic beacon traffic to simulate command-and-control activity
This should be detected by Suricata with appropriate rules
"""

import time
import random
import socket
import struct
import sys
import json
from datetime import datetime
from threading import Thread
import requests
import logging

class C2Beacon:
    def __init__(self, c2_server="192.168.1.100", beacon_interval=10, jitter=2):
        """
        Initialize C2 Beacon
        
        Args:
            c2_server: IP address or domain of C2 server
            beacon_interval: Base interval between beacons in seconds
            jitter: Random jitter to add to intervals
        """
        self.c2_server = c2_server
        self.base_interval = beacon_interval
        self.jitter = jitter
        self.running = False
        self.victim_ip = "192.168.1." + str(random.randint(100, 200))
        
        # Setup logging to simulate Suricata detection
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging to create synthetic Suricata alerts"""
        self.logger = logging.getLogger('C2Beacon')
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        fh = logging.FileHandler('c2_beacon.log')
        fh.setLevel(logging.INFO)
        
        # Create JSON formatter to simulate Suricata eve.json format
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "alert",
                    "src_ip": getattr(record, 'src_ip', '192.168.1.150'),
                    "dest_ip": getattr(record, 'dest_ip', self.c2_server),
                    "src_port": random.randint(1024, 65535),
                    "dest_port": getattr(record, 'dest_port', 443),
                    "proto": getattr(record, 'proto', 'TCP'),
                    "alert": {
                        "action": getattr(record, 'action', 'allowed'),
                        "signature": getattr(record, 'signature', 'ET MALWARE C2 Beacon'),
                        "category": getattr(record, 'category', 'Malware Beacon'),
                        "severity": getattr(record, 'severity', 1)
                    },
                    "bytes_toserver": getattr(record, 'bytes_toserver', random.randint(100, 500)),
                    "bytes_toclient": getattr(record, 'bytes_toclient', random.randint(1000, 5000)),
                    "flow": {
                        "bytes_toserver": getattr(record, 'bytes_toserver', random.randint(100, 500)),
                        "bytes_toclient": getattr(record, 'bytes_toclient', random.randint(1000, 5000)),
                        "pkts_toserver": random.randint(1, 5),
                        "pkts_toclient": random.randint(3, 10)
                    }
                }
                return json.dumps(log_data)
        
        formatter = JsonFormatter()
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
    def http_beacon(self):
        """Simulate HTTP beacon (common C2 technique)"""
        try:
            # Simulate HTTP request to C2
            url = f"http://{self.c2_server}/api/v1/checkin"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'X-Client-ID': 'victim-' + self.victim_ip
            }
            
            # Add some beacon data
            data = {
                'status': 'active',
                'id': random.randint(1000, 9999),
                'timestamp': time.time()
            }
            
            # Simulate the request without actually sending it
            # In real scenario, you would use: response = requests.post(url, headers=headers, json=data)
            
            # Log the simulated beacon
            self.logger.info('HTTP Beacon', extra={
                'src_ip': self.victim_ip,
                'dest_ip': self.c2_server,
                'dest_port': 80,
                'proto': 'TCP',
                'bytes_toserver': len(json.dumps(data)),
                'bytes_toclient': 2048,
                'signature': 'ET MALWARE HTTP Beacon Checkin'
            })
            
            print(f"[{datetime.now()}] HTTP Beacon sent from {self.victim_ip} to {self.c2_server}")
            
        except Exception as e:
            print(f"HTTP Beacon error: {e}")
    
    def dns_beacon(self):
        """Simulate DNS beacon (covert channel)"""
        try:
            # Simulate DNS query for C2
            domain = f"{random.randint(100000, 999999)}.malicious-domain.com"
            
            self.logger.info('DNS Beacon', extra={
                'src_ip': self.victim_ip,
                'dest_ip': '8.8.8.8',  # DNS server
                'dest_port': 53,
                'proto': 'UDP',
                'bytes_toserver': 100,
                'bytes_toclient': 200,
                'signature': 'ET MALWARE DNS Beacon'
            })
            
            print(f"[{datetime.now()}] DNS Beacon: {domain}")
            
        except Exception as e:
            print(f"DNS Beacon error: {e}")
    
    def icmp_beacon(self):
        """Simulate ICMP beacon (ping-based C2)"""
        try:
            self.logger.info('ICMP Beacon', extra={
                'src_ip': self.victim_ip,
                'dest_ip': self.c2_server,
                'proto': 'ICMP',
                'bytes_toserver': 64,
                'bytes_toclient': 64,
                'signature': 'ET MALWARE ICMP Tunnel'
            })
            
            print(f"[{datetime.now()}] ICMP Beacon to {self.c2_server}")
            
        except Exception as e:
            print(f"ICMP Beacon error: {e}")
    
    def generate_background_traffic(self):
        """Generate random background traffic to hide the beacon"""
        destinations = [
            '8.8.8.8', '1.1.1.1', '208.67.222.222',
            'www.google.com', 'www.microsoft.com', 'www.apple.com'
        ]
        
        for _ in range(random.randint(1, 3)):
            dest = random.choice(destinations)
            self.logger.info('Background Traffic', extra={
                'src_ip': self.victim_ip,
                'dest_ip': dest,
                'dest_port': random.choice([80, 443, 53]),
                'proto': random.choice(['TCP', 'UDP']),
                'bytes_toserver': random.randint(500, 2000),
                'bytes_toclient': random.randint(1000, 5000),
                'signature': random.choice(['ET WEB_CLIENT', 'ET DNS', 'ET POLICY'])
            })
    
    def run(self, duration=300):
        """Run the beacon simulation"""
        print(f"Starting C2 Beacon Simulation")
        print(f"Victim IP: {self.victim_ip}")
        print(f"C2 Server: {self.c2_server}")
        print(f"Base Interval: {self.base_interval}s")
        print(f"Duration: {duration}s")
        print("-" * 50)
        
        self.running = True
        start_time = time.time()
        
        beacon_types = [self.http_beacon, self.dns_beacon, self.icmp_beacon]
        
        while self.running and (time.time() - start_time) < duration:
            # Send beacon
            beacon_func = random.choice(beacon_types)
            beacon_func()
            
            # Generate some background traffic
            self.generate_background_traffic()
            
            # Calculate next beacon interval with jitter
            interval = self.base_interval + random.uniform(-self.jitter, self.jitter)
            
            # Sleep until next beacon
            time.sleep(max(1, interval))
        
        print("\nBeacon simulation completed.")
        print(f"Logs saved to: c2_beacon.log")

def create_suricata_rules():
    """Generate sample Suricata rules to detect the beacons"""
    rules = """
# HTTP Beacon Rule
alert http $HOME_NET any -> $EXTERNAL_NET any (msg:"ET MALWARE C2 HTTP Beacon"; flow:established,to_server; content:"/api/v1/checkin"; http_uri; content:"status"; content:"active"; fast_pattern; classtype:trojan-activity; sid:2024123; rev:1;)

# DNS Beacon Rule  
alert dns $HOME_NET any -> any 53 (msg:"ET MALWARE DNS Beacon"; dns.query; content:".malicious-domain.com"; nocase; isdataat:!1,relative; classtype:domain-c2; sid:2024124; rev:1;)

# ICMP Beacon Rule
alert icmp $HOME_NET any -> $EXTERNAL_NET any (msg:"ET MALWARE ICMP Tunnel"; itype:8; icode:0; dsize:>50; content:"|00 00 00 00|"; depth:4; offset:4; classtype:bad-unknown; sid:2024125; rev:1;)
"""
    
    with open('suricata_beacon_rules.rules', 'w') as f:
        f.write(rules)
    
    print("Suricata rules saved to: suricata_beacon_rules.rules")
    print("Place in /etc/suricata/rules/ and reload Suricata")

if __name__ == "__main__":
    # Create Suricata detection rules
    create_suricata_rules()
    
    # Run beacon simulation
    beacon = C2Beacon(
        c2_server="192.168.1.100",
        beacon_interval=10,  # 10 second base interval
        jitter=2  # ±2 seconds jitter
    )
    
    # Run for 5 minutes
    beacon.run(duration=300)
