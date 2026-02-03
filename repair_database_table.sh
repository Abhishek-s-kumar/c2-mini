#!/bin/bash

# ============================================================================
# Direct Database Table Fix - Nuclear Option
# This will DROP and recreate the table with correct structure
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${RED}========================================${NC}"
echo -e "${RED}DATABASE TABLE REPAIR${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo -e "${YELLOW}This will DROP and recreate the conn_log table.${NC}"
echo -e "${YELLOW}All existing data will be lost!${NC}"
echo ""
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}[1/5] Checking PostgreSQL...${NC}"
if ! sudo systemctl is-active --quiet postgresql; then
    echo "Starting PostgreSQL..."
    sudo systemctl start postgresql
fi
echo -e "${GREEN}✓ PostgreSQL is running${NC}"

echo ""
echo -e "${BLUE}[2/5] Dropping old table...${NC}"
sudo -u postgres psql -d c2db <<'EOF'
DROP TABLE IF EXISTS conn_log CASCADE;
EOF
echo -e "${GREEN}✓ Old table dropped${NC}"

echo ""
echo -e "${BLUE}[3/5] Creating new table with correct structure...${NC}"
sudo -u postgres psql -d c2db <<'EOF'
CREATE TABLE conn_log (
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

-- Grant permissions
GRANT ALL PRIVILEGES ON TABLE conn_log TO c2user;
GRANT USAGE, SELECT ON SEQUENCE conn_log_id_seq TO c2user;

-- Create indexes
CREATE INDEX idx_conn_log_ts ON conn_log(ts);
CREATE INDEX idx_conn_log_src ON conn_log(id_orig_h);
CREATE INDEX idx_conn_log_dst ON conn_log(id_resp_h);
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Table created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create table${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}[4/5] Verifying table structure...${NC}"
sudo -u postgres psql -d c2db -c "\d conn_log"

echo ""
echo -e "${BLUE}[5/5] Adding sample data for testing...${NC}"
sudo -u postgres psql -d c2db <<'EOF'
INSERT INTO conn_log (ts, uid, id_orig_h, id_orig_p, id_resp_h, id_resp_p, proto, service, duration, orig_bytes, resp_bytes, conn_state)
VALUES 
    (NOW() - INTERVAL '30 seconds', 'test001', '192.168.1.100', 54321, '8.8.8.8', 53, 'udp', 'dns', 0.05, 64, 128, 'SF'),
    (NOW() - INTERVAL '1 minute', 'test002', '192.168.1.100', 54322, '1.1.1.1', 443, 'tcp', 'ssl', 1.2, 512, 2048, 'SF'),
    (NOW() - INTERVAL '90 seconds', 'test003', '192.168.1.101', 12345, '8.8.4.4', 53, 'udp', 'dns', 0.08, 64, 128, 'SF'),
    (NOW() - INTERVAL '2 minutes', 'test004', '192.168.1.100', 54323, '93.184.216.34', 80, 'tcp', 'http', 2.5, 1024, 4096, 'SF'),
    (NOW() - INTERVAL '150 seconds', 'test005', '192.168.1.102', 65432, '8.8.8.8', 53, 'udp', 'dns', 0.06, 64, 128, 'SF');
EOF
echo -e "${GREEN}✓ Sample data added${NC}"

echo ""
echo -e "${BLUE}Testing data retrieval...${NC}"
sudo -u postgres psql -d c2db -c "SELECT COUNT(*) as total_records FROM conn_log;"
sudo -u postgres psql -d c2db -c "SELECT ts, id_orig_h, id_resp_h FROM conn_log ORDER BY ts DESC LIMIT 3;"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Database Table Repaired!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Now test the connection from Python:"
echo ""
echo -e "${BLUE}python3 -c \"${NC}"
echo "import psycopg2"
echo "import configparser"
echo "config = configparser.ConfigParser()"
echo "config.read('config/database.conf')"
echo "conn = psycopg2.connect(host=config['postgresql']['host'], database=config['postgresql']['database'], user=config['postgresql']['user'], password=config['postgresql']['password'])"
echo "cursor = conn.cursor()"
echo "cursor.execute('SELECT COUNT(*) FROM conn_log')"
echo "print(f'Records: {cursor.fetchone()[0]}')"
echo "conn.close()"
echo -e "${BLUE}\"${NC}"
echo ""
echo "Then restart your system:"
echo -e "${BLUE}./stop_system_enhanced.sh${NC}"
echo -e "${BLUE}./start_system.sh${NC}"
