#!/bin/bash

# ============================================================================
# Complete System Fix Script
# Fixes port conflicts and database table issues
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}C2 Detection System - Complete Fix${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================================================
# 1. Stop all running processes
# ============================================================================
echo -e "${YELLOW}[1/4] Stopping all running processes...${NC}"

# Kill Flask processes
pkill -f "dashboard.py" 2>/dev/null && echo "  ✓ Stopped dashboard" || echo "  - No dashboard running"
pkill -f "monitor_c2.py" 2>/dev/null && echo "  ✓ Stopped monitor" || echo "  - No monitor running"

# Find and kill anything on port 5000
PORT_PID=$(lsof -ti:5000 2>/dev/null)
if [ ! -z "$PORT_PID" ]; then
    kill -9 $PORT_PID 2>/dev/null
    echo "  ✓ Killed process on port 5000 (PID: $PORT_PID)"
else
    echo "  - Port 5000 is free"
fi

sleep 2
echo -e "${GREEN}✓ All processes stopped${NC}"

# ============================================================================
# 2. Check and fix database table
# ============================================================================
echo ""
echo -e "${YELLOW}[2/4] Checking database table...${NC}"

# Check if table exists and has correct structure
TABLE_CHECK=$(sudo -u postgres psql -d c2db -tAc "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'conn_log');")

if [ "$TABLE_CHECK" = "t" ]; then
    echo "  ℹ Table exists, checking structure..."
    
    # Check if 'ts' column exists
    COLUMN_CHECK=$(sudo -u postgres psql -d c2db -tAc "SELECT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'conn_log' AND column_name = 'ts');")
    
    if [ "$COLUMN_CHECK" = "f" ]; then
        echo -e "  ${RED}✗ Column 'ts' is missing!${NC}"
        echo "  Dropping and recreating table..."
        
        sudo -u postgres psql -d c2db <<EOF
DROP TABLE IF EXISTS conn_log CASCADE;
EOF
        TABLE_CHECK="f"
    else
        echo "  ✓ Table structure looks good"
    fi
fi

if [ "$TABLE_CHECK" = "f" ]; then
    echo "  Creating conn_log table..."
    
    sudo -u postgres psql -d c2db <<EOF
CREATE TABLE IF NOT EXISTS conn_log (
    id SERIAL PRIMARY KEY,
    ts TIMESTAMP NOT NULL,
    uid VARCHAR(50),
    id_orig_h INET,
    id_orig_p INTEGER,
    id_resp_h INET,
    id_resp_p INTEGER,
    proto VARCHAR(10),
    service VARCHAR(50),
    duration FLOAT,
    orig_bytes INTEGER,
    resp_bytes INTEGER,
    conn_state VARCHAR(30),
    local_orig BOOLEAN,
    local_resp BOOLEAN,
    missed_bytes INTEGER,
    history VARCHAR(50),
    orig_pkts INTEGER,
    orig_ip_bytes INTEGER,
    resp_pkts INTEGER,
    resp_ip_bytes INTEGER,
    tunnel_parents VARCHAR(100),
    imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

GRANT ALL PRIVILEGES ON TABLE conn_log TO c2user;
GRANT USAGE, SELECT ON SEQUENCE conn_log_id_seq TO c2user;

CREATE INDEX IF NOT EXISTS idx_conn_log_ts ON conn_log(ts);
CREATE INDEX IF NOT EXISTS idx_conn_log_src ON conn_log(id_orig_h);
CREATE INDEX IF NOT EXISTS idx_conn_log_dst ON conn_log(id_resp_h);
EOF
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Table created successfully${NC}"
    else
        echo -e "${RED}  ✗ Failed to create table${NC}"
        exit 1
    fi
fi

# ============================================================================
# 3. Add some test data
# ============================================================================
echo ""
echo -e "${YELLOW}[3/4] Adding test data...${NC}"

sudo -u postgres psql -d c2db <<EOF
INSERT INTO conn_log (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, service, duration, orig_bytes, resp_bytes, conn_state)
VALUES 
    (NOW() - INTERVAL '1 minute', 'test1', '192.168.1.100', 12345, '8.8.8.8', 53, 'udp', 'dns', 0.1, 64, 128, 'SF'),
    (NOW() - INTERVAL '2 minutes', 'test2', '192.168.1.100', 12346, '1.1.1.1', 443, 'tcp', 'ssl', 1.5, 512, 1024, 'SF'),
    (NOW() - INTERVAL '3 minutes', 'test3', '192.168.1.101', 54321, '8.8.8.8', 53, 'udp', 'dns', 0.2, 64, 128, 'SF')
ON CONFLICT DO NOTHING;
EOF

echo -e "${GREEN}  ✓ Test data added${NC}"

# ============================================================================
# 4. Test database connection
# ============================================================================
echo ""
echo -e "${YELLOW}[4/4] Testing database connection...${NC}"

python3 -c "
import psycopg2
import configparser
import sys

config = configparser.ConfigParser()
config.read('config/database.conf')

try:
    conn = psycopg2.connect(
        host=config['postgresql']['host'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password']
    )
    
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM conn_log')
    count = cursor.fetchone()[0]
    
    print(f'  ✓ Database connection successful!')
    print(f'  ✓ Found {count} records in conn_log table')
    
    conn.close()
except Exception as e:
    print(f'  ✗ Connection failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Database connection test failed!${NC}"
    exit 1
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}System Fixed Successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "What's been fixed:"
echo "  ✓ Killed processes on port 5000"
echo "  ✓ Fixed database table structure"
echo "  ✓ Added test data"
echo "  ✓ Verified database connection"
echo ""
echo "Next steps:"
echo ""
echo "1. Start the system:"
echo "   ${BLUE}./start_system.sh${NC}"
echo ""
echo "2. Access the dashboard:"
echo "   ${BLUE}http://localhost:5000${NC}"
echo ""
echo "3. Or run components separately:"
echo "   ${BLUE}python3 monitor_c2.py${NC}     # Monitoring service"
echo "   ${BLUE}python3 dashboard.py${NC}      # Web dashboard"
echo "   ${BLUE}python3 real_time_analyzer.py${NC}  # Real-time analysis"
echo ""
