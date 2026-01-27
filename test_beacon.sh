#!/bin/bash
# C2 Beacon Traffic Generator
# Generates simulated beacon traffic for testing detection

INTERFACE="enp0s8"
BEACON_IP=$(ip -4 addr show $INTERFACE | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
TARGET_IP="8.8.8.8"
INTERVAL=10
DURATION=300
COUNT=1

echo "========================================"
echo "C2 BEACON TRAFFIC GENERATOR"
echo "========================================"
echo "Interface: $INTERFACE"
echo "Beacon IP: $BEACON_IP"
echo "Target IP: $TARGET_IP"
echo "Interval: $INTERVAL seconds"
echo "Duration: $DURATION seconds"
echo ""
echo "This will simulate C2 beacon traffic for testing."
echo "Run this in one terminal and the detector in another."
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

echo -e "\n[$BEACON_IP] Starting beacon to $TARGET_IP every ${INTERVAL}s..."

for ((i=1; i<=DURATION/INTERVAL; i++)); do
    # Send beacon packet
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] Beacon $i: $BEACON_IP → $TARGET_IP"
    
    # Send ICMP echo (ping) as beacon
    ping -c 1 -I $INTERFACE -W 1 $TARGET_IP > /dev/null 2>&1
    
    # Also send a small TCP packet (using netcat if available)
    if command -v nc &> /dev/null; then
        echo "beacon_$i" | timeout 1 nc -w 1 $TARGET_IP 80 2>/dev/null
    fi
    
    sleep $INTERVAL
done

echo -e "\n[✓] Beacon simulation complete"
echo "Check your detector for alerts!"
