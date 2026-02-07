#!/bin/bash

# ============================================================================
# C2 Beacon Traffic Generator - Improved Version
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}C2 BEACON TRAFFIC GENERATOR${NC}"
echo -e "${BLUE}========================================${NC}"

# Try to detect interface and IP
INTERFACE=""
BEACON_IP=""

# Try common interface names
for iface in enp0s8 eth0 eth1 wlan0 ens33 enp0s3; do
    if ip link show $iface &>/dev/null; then
        INTERFACE=$iface
        BEACON_IP=$(ip -4 addr show $iface 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
        if [ ! -z "$BEACON_IP" ]; then
            break
        fi
    fi
done

# If still not found, use default interface
if [ -z "$INTERFACE" ]; then
    INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
    if [ ! -z "$INTERFACE" ]; then
        BEACON_IP=$(ip -4 addr show $INTERFACE 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
    fi
fi

# Fallback to localhost if nothing works
if [ -z "$BEACON_IP" ]; then
    BEACON_IP="127.0.0.1"
    INTERFACE="lo"
fi

TARGET_IP="8.8.8.8"
INTERVAL=10
DURATION=300

echo -e "${GREEN}Interface: $INTERFACE${NC}"
echo -e "${GREEN}Source IP: $BEACON_IP${NC}"
echo -e "${YELLOW}Target IP: $TARGET_IP${NC}"
echo -e "${BLUE}Interval: $INTERVAL seconds${NC}"
echo -e "${BLUE}Duration: $DURATION seconds${NC}"
echo ""
echo "This will generate regular 'beacon-like' network traffic."
echo "The C2 detector should identify this as suspicious periodic activity."
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

echo ""
echo -e "${GREEN}[$BEACON_IP] Starting beacon to $TARGET_IP every ${INTERVAL}s...${NC}"
echo ""

BEACON_COUNT=$((DURATION / INTERVAL))

for ((i=1; i<=BEACON_COUNT; i++)); do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${YELLOW}[$timestamp]${NC} Beacon $i/$BEACON_COUNT: ${BLUE}$BEACON_IP${NC} → ${GREEN}$TARGET_IP${NC}"
    
    # Send ICMP echo (ping) as beacon
    if [ "$INTERFACE" != "lo" ]; then
        ping -c 1 -I $INTERFACE -W 1 $TARGET_IP > /dev/null 2>&1
    else
        ping -c 1 -W 1 $TARGET_IP > /dev/null 2>&1
    fi
    
    # Also try DNS query (more realistic C2 beacon)
    dig +short @$TARGET_IP google.com > /dev/null 2>&1 || true
    
    # Small HTTP-like request if nc is available
    if command -v nc &> /dev/null; then
        echo "GET / HTTP/1.0" | timeout 1 nc -w 1 $TARGET_IP 80 2>/dev/null || true
    fi
    
    # Show progress
    REMAINING=$((DURATION - (i * INTERVAL)))
    echo -e "  ${BLUE}Progress: $i/$BEACON_COUNT beacons sent, ${REMAINING}s remaining${NC}"
    
    # Wait for next beacon
    if [ $i -lt $BEACON_COUNT ]; then
        sleep $INTERVAL
    fi
done

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Beacon simulation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Sent $BEACON_COUNT beacons in $DURATION seconds"
echo ""
echo "Check your dashboard at: http://localhost:5000"
echo "Or run: python3 real_time_analyzer.py"
echo ""
echo -e "${YELLOW}Note: For the detector to work, this traffic must be:${NC}"
echo "  1. Captured by Zeek or similar network monitor"
echo "  2. Imported into the database using import_zeek_logs.py"
echo ""
