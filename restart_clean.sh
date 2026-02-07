#!/bin/bash

# ============================================================================
# Complete System Restart - Forces reload of all Python code
# ============================================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         FULL SYSTEM RESTART                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${YELLOW}[1/4] Stopping all processes...${NC}"
pkill -f "dashboard.py" 2>/dev/null
pkill -f "monitor_c2.py" 2>/dev/null
pkill -f "real_time_analyzer.py" 2>/dev/null
lsof -ti:5000 2>/dev/null | xargs kill -9 2>/dev/null
sleep 2
echo -e "${GREEN}✓ All processes stopped${NC}"

echo ""
echo -e "${YELLOW}[2/4] Verifying fixed code...${NC}"
if grep -q "resample('1s')" real_time_analyzer.py; then
    echo -e "${GREEN}✓ Code has lowercase 's' - pandas 3.x compatible${NC}"
else
    echo -e "${YELLOW}⚠ Code still has uppercase 'S' - fixing now...${NC}"
    sed -i "s/resample('1S')/resample('1s')/g" real_time_analyzer.py
    echo -e "${GREEN}✓ Code fixed${NC}"
fi

echo ""
echo -e "${YELLOW}[3/4] Clearing old log entries...${NC}"
if [ -f "logs/c2_monitor.log" ]; then
    # Backup old log
    cp logs/c2_monitor.log logs/c2_monitor.log.backup_$(date +%Y%m%d_%H%M%S)
    # Clear log
    > logs/c2_monitor.log
    echo -e "${GREEN}✓ Log cleared (backup created)${NC}"
else
    echo -e "${YELLOW}○ No log file to clear${NC}"
fi

echo ""
echo -e "${YELLOW}[4/4] Starting services...${NC}"

# Make sure we're in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start dashboard in background
echo "  Starting dashboard..."
python3 dashboard.py &
DASH_PID=$!
sleep 2

# Start monitor in background
echo "  Starting monitor..."
python3 monitor_c2.py &
MON_PID=$!
sleep 1

echo -e "${GREEN}✓ Services started${NC}"
echo ""

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              SYSTEM RESTARTED                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Services:"
echo "  Dashboard (PID: $DASH_PID) → http://localhost:5000"
echo "  Monitor   (PID: $MON_PID)   → logs/c2_monitor.log"
echo ""
echo "Watch for new logs (should have no errors):"
echo -e "${BLUE}  tail -f logs/c2_monitor.log${NC}"
echo ""
echo "To stop:"
echo -e "${BLUE}  ./stop_system_enhanced.sh${NC}"
echo ""
