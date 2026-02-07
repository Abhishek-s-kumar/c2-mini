#!/bin/bash
# Enhanced Stop Script - Kills all C2 detection processes and frees port 5000

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping C2 Detection System...${NC}"
echo "========================================"

# Kill Python processes
echo "Stopping Python services..."
pkill -f "dashboard.py" 2>/dev/null && echo "  ✓ Stopped dashboard" || echo "  - No dashboard running"
pkill -f "monitor_c2.py" 2>/dev/null && echo "  ✓ Stopped monitor" || echo "  - No monitor running"
pkill -f "real_time_analyzer.py" 2>/dev/null && echo "  ✓ Stopped analyzer" || echo "  - No analyzer running"

# Kill anything on port 5000
echo ""
echo "Checking port 5000..."
PORT_PID=$(lsof -ti:5000 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    kill -9 $PORT_PID 2>/dev/null
    echo -e "  ${GREEN}✓ Killed process on port 5000 (PID: $PORT_PID)${NC}"
else
    echo "  ✓ Port 5000 is free"
fi

# Double check port is free
sleep 1
if lsof -ti:5000 >/dev/null 2>&1; then
    echo -e "${RED}  ✗ Port 5000 still in use. Trying force kill...${NC}"
    fuser -k 5000/tcp 2>/dev/null
    sleep 1
fi

echo ""
echo -e "${GREEN}========================================"
echo "System Stopped"
echo "========================================${NC}"
