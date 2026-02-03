#!/bin/bash

# ============================================================================
# Database Status Check
# ============================================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Database Status Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check PostgreSQL
echo -e "${YELLOW}[1/5] PostgreSQL Service${NC}"
if sudo systemctl is-active --quiet postgresql; then
    echo -e "${GREEN}✓ PostgreSQL is running${NC}"
else
    echo -e "${RED}✗ PostgreSQL is not running${NC}"
    echo "  Run: sudo systemctl start postgresql"
fi

# Check database exists
echo ""
echo -e "${YELLOW}[2/5] Database Exists${NC}"
DB_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='c2db'" 2>/dev/null)
if [ "$DB_EXISTS" = "1" ]; then
    echo -e "${GREEN}✓ Database 'c2db' exists${NC}"
else
    echo -e "${RED}✗ Database 'c2db' not found${NC}"
fi

# Check user exists
echo ""
echo -e "${YELLOW}[3/5] User Exists${NC}"
USER_EXISTS=$(sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='c2user'" 2>/dev/null)
if [ "$USER_EXISTS" = "1" ]; then
    echo -e "${GREEN}✓ User 'c2user' exists${NC}"
else
    echo -e "${RED}✗ User 'c2user' not found${NC}"
fi

# Check table exists
echo ""
echo -e "${YELLOW}[4/5] Table Structure${NC}"
TABLE_EXISTS=$(sudo -u postgres psql -d c2db -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'conn_log')" 2>/dev/null)
if [ "$TABLE_EXISTS" = "t" ]; then
    echo -e "${GREEN}✓ Table 'conn_log' exists${NC}"
    
    # Check for ts column
    TS_EXISTS=$(sudo -u postgres psql -d c2db -tAc "SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'conn_log' AND column_name = 'ts')" 2>/dev/null)
    if [ "$TS_EXISTS" = "t" ]; then
        echo -e "${GREEN}  ✓ Column 'ts' exists${NC}"
    else
        echo -e "${RED}  ✗ Column 'ts' missing!${NC}"
    fi
    
    # Show column count
    COL_COUNT=$(sudo -u postgres psql -d c2db -tAc "SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'conn_log'" 2>/dev/null)
    echo -e "${BLUE}  ℹ Table has $COL_COUNT columns${NC}"
    
    # Show record count
    ROW_COUNT=$(sudo -u postgres psql -d c2db -tAc "SELECT COUNT(*) FROM conn_log" 2>/dev/null)
    echo -e "${BLUE}  ℹ Table has $ROW_COUNT records${NC}"
else
    echo -e "${RED}✗ Table 'conn_log' not found${NC}"
fi

# Test Python connection
echo ""
echo -e "${YELLOW}[5/5] Python Connection Test${NC}"
if [ -f "config/database.conf" ]; then
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
    
    cursor = conn.cursor()
    
    # Test query
    cursor.execute('SELECT COUNT(*) FROM conn_log')
    count = cursor.fetchone()[0]
    
    print('${GREEN}✓ Python connection successful${NC}')
    print('${BLUE}  ℹ Found', count, 'records${NC}')
    
    # Test ts column
    cursor.execute('SELECT ts FROM conn_log LIMIT 1')
    result = cursor.fetchone()
    if result:
        print('${GREEN}  ✓ ts column accessible${NC}')
    
    conn.close()
    
except Exception as e:
    print('${RED}✗ Python connection failed${NC}')
    print('${RED}  Error:', str(e), '${NC}')
    sys.exit(1)
" 2>&1
else
    echo -e "${RED}✗ Config file not found: config/database.conf${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo "If you see errors above, run:"
echo -e "${BLUE}  ./repair_database_table.sh${NC}"
echo -e "${BLUE}========================================${NC}"
