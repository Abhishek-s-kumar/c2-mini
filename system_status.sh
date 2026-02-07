#!/bin/bash

# ============================================================================
# C2 Detection System - Complete Status Check
# ============================================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

clear
echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       C2 BEACON DETECTION SYSTEM - STATUS REPORT          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# System Services
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}1. SYSTEM SERVICES${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# PostgreSQL
if sudo systemctl is-active --quiet postgresql; then
    echo -e "${GREEN}✓${NC} PostgreSQL Database    : ${GREEN}Running${NC}"
else
    echo -e "${RED}✗${NC} PostgreSQL Database    : ${RED}Stopped${NC}"
fi

# Flask Dashboard
if pgrep -f "dashboard.py" > /dev/null; then
    DASH_PID=$(pgrep -f "dashboard.py")
    echo -e "${GREEN}✓${NC} Flask Dashboard        : ${GREEN}Running${NC} (PID: $DASH_PID)"
else
    echo -e "${YELLOW}○${NC} Flask Dashboard        : ${YELLOW}Not running${NC}"
fi

# Monitor Service
if pgrep -f "monitor_c2.py" > /dev/null; then
    MON_PID=$(pgrep -f "monitor_c2.py")
    echo -e "${GREEN}✓${NC} C2 Monitor Service     : ${GREEN}Running${NC} (PID: $MON_PID)"
else
    echo -e "${YELLOW}○${NC} C2 Monitor Service     : ${YELLOW}Not running${NC}"
fi

# Port 5000
if lsof -i:5000 > /dev/null 2>&1; then
    PORT_INFO=$(lsof -i:5000 | grep LISTEN | awk '{print $1, $2}')
    echo -e "${GREEN}✓${NC} Port 5000              : ${GREEN}In use${NC} ($PORT_INFO)"
else
    echo -e "${YELLOW}○${NC} Port 5000              : ${YELLOW}Available${NC}"
fi

echo ""

# Database Status
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}2. DATABASE STATUS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

DB_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='c2db'" 2>/dev/null)
if [ "$DB_EXISTS" = "1" ]; then
    echo -e "${GREEN}✓${NC} Database 'c2db'        : ${GREEN}Exists${NC}"
    
    # Table check
    TABLE_EXISTS=$(sudo -u postgres psql -d c2db -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'conn_log')" 2>/dev/null)
    if [ "$TABLE_EXISTS" = "t" ]; then
        echo -e "${GREEN}✓${NC} Table 'conn_log'       : ${GREEN}Exists${NC}"
        
        # Record count
        ROW_COUNT=$(sudo -u postgres psql -d c2db -tAc "SELECT COUNT(*) FROM conn_log" 2>/dev/null)
        if [ $ROW_COUNT -gt 0 ]; then
            echo -e "${GREEN}✓${NC} Records                : ${GREEN}$ROW_COUNT rows${NC}"
        else
            echo -e "${YELLOW}○${NC} Records                : ${YELLOW}Empty (no data yet)${NC}"
        fi
        
        # Recent data
        RECENT_COUNT=$(sudo -u postgres psql -d c2db -tAc "SELECT COUNT(*) FROM conn_log WHERE ts > NOW() - INTERVAL '5 minutes'" 2>/dev/null)
        if [ $RECENT_COUNT -gt 0 ]; then
            echo -e "${GREEN}✓${NC} Recent data (5min)     : ${GREEN}$RECENT_COUNT records${NC}"
        else
            echo -e "${YELLOW}○${NC} Recent data (5min)     : ${YELLOW}No recent data${NC}"
        fi
    else
        echo -e "${RED}✗${NC} Table 'conn_log'       : ${RED}Missing${NC}"
    fi
else
    echo -e "${RED}✗${NC} Database 'c2db'        : ${RED}Not found${NC}"
fi

echo ""

# Python Environment
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}3. PYTHON ENVIRONMENT${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Virtual Environment    : ${GREEN}Exists${NC}"
    
    # Check if activated
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo -e "${GREEN}✓${NC} Environment Status     : ${GREEN}Activated${NC}"
    else
        echo -e "${YELLOW}○${NC} Environment Status     : ${YELLOW}Not activated${NC}"
    fi
else
    echo -e "${RED}✗${NC} Virtual Environment    : ${RED}Not found${NC}"
fi

# Test Python connection
if [ -f "config/database.conf" ]; then
    echo -e "${GREEN}✓${NC} Config File            : ${GREEN}Found${NC}"
    
    python3 -c "
import sys
import psycopg2
import configparser

try:
    config = configparser.ConfigParser()
    config.read('config/database.conf')
    
    conn = psycopg2.connect(
        host=config['postgresql']['host'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password']
    )
    
    print('${GREEN}✓${NC} Database Connection    : ${GREEN}Successful${NC}')
    conn.close()
    
except Exception as e:
    print('${RED}✗${NC} Database Connection    : ${RED}Failed${NC}')
    sys.exit(1)
" 2>/dev/null
else
    echo -e "${RED}✗${NC} Config File            : ${RED}Not found${NC}"
fi

echo ""

# Detection Status
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}4. DETECTION STATUS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ -f "output/alerts.json" ]; then
    ALERT_COUNT=$(wc -l < output/alerts.json 2>/dev/null || echo 0)
    if [ $ALERT_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓${NC} Alerts File            : ${GREEN}$ALERT_COUNT alerts${NC}"
    else
        echo -e "${YELLOW}○${NC} Alerts File            : ${YELLOW}No alerts yet${NC}"
    fi
else
    echo -e "${YELLOW}○${NC} Alerts File            : ${YELLOW}Not created${NC}"
fi

if [ -f "logs/c2_monitor.log" ]; then
    LOG_SIZE=$(du -h logs/c2_monitor.log 2>/dev/null | cut -f1)
    LOG_LINES=$(wc -l < logs/c2_monitor.log 2>/dev/null)
    echo -e "${GREEN}✓${NC} Monitor Log            : ${GREEN}$LOG_LINES lines ($LOG_SIZE)${NC}"
else
    echo -e "${YELLOW}○${NC} Monitor Log            : ${YELLOW}Not created${NC}"
fi

echo ""

# Access URLs
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}5. ACCESS INFORMATION${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if pgrep -f "dashboard.py" > /dev/null; then
    echo -e "${GREEN}Dashboard URL:${NC}"
    echo -e "  ${CYAN}http://localhost:5000${NC}"
    
    # Get local IP
    LOCAL_IP=$(hostname -I | awk '{print $1}')
    if [ ! -z "$LOCAL_IP" ]; then
        echo -e "  ${CYAN}http://$LOCAL_IP:5000${NC}"
    fi
else
    echo -e "${YELLOW}Dashboard not running. Start with: ./start_system.sh${NC}"
fi

echo ""

# Quick Actions
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}6. QUICK ACTIONS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if ! pgrep -f "dashboard.py" > /dev/null; then
    echo -e "${YELLOW}▶${NC}  Start system          : ${CYAN}./start_system.sh${NC}"
fi

if pgrep -f "dashboard.py" > /dev/null || pgrep -f "monitor_c2.py" > /dev/null; then
    echo -e "${YELLOW}■${NC}  Stop system           : ${CYAN}./stop_system_enhanced.sh${NC}"
fi

echo -e "${YELLOW}⟳${NC}  Generate test traffic : ${CYAN}./test_beacon_improved.sh${NC}"
echo -e "${YELLOW}↑${NC}  Import Zeek logs      : ${CYAN}python3 import_zeek_logs.py <file>${NC}"
echo -e "${YELLOW}📊${NC}  View monitor log      : ${CYAN}tail -f logs/c2_monitor.log${NC}"
echo -e "${YELLOW}🔍${NC}  Manual analysis       : ${CYAN}python3 real_time_analyzer.py${NC}"

echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Overall Health
echo ""
DB_OK=0
DASHBOARD_OK=0

if [ "$DB_EXISTS" = "1" ] && [ "$TABLE_EXISTS" = "t" ]; then
    DB_OK=1
fi

if pgrep -f "dashboard.py" > /dev/null; then
    DASHBOARD_OK=1
fi

if [ $DB_OK -eq 1 ] && [ $DASHBOARD_OK -eq 1 ]; then
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║           ✓ SYSTEM STATUS: OPERATIONAL                    ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
elif [ $DB_OK -eq 1 ]; then
    echo -e "${YELLOW}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║      ○ SYSTEM STATUS: READY (Services not running)       ║${NC}"
    echo -e "${YELLOW}╚════════════════════════════════════════════════════════════╝${NC}"
else
    echo -e "${RED}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║         ✗ SYSTEM STATUS: NEEDS ATTENTION                  ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Run ./repair_database_table.sh to fix database issues${NC}"
fi

echo ""
