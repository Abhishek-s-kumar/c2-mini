#!/bin/bash

# ============================================================================
# PostgreSQL Authentication Fix Script
# ============================================================================
# This script fixes the password authentication issue for the c2user account

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PostgreSQL Authentication Fix${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if PostgreSQL is running
if ! sudo systemctl is-active --quiet postgresql; then
    echo -e "${YELLOW}Starting PostgreSQL...${NC}"
    sudo systemctl start postgresql
fi

echo "This script will reset the password for the c2user database account."
echo ""
read -sp "Enter NEW password for c2user: " NEW_PASSWORD
echo ""
read -sp "Confirm password: " CONFIRM_PASSWORD
echo ""

if [ "$NEW_PASSWORD" != "$CONFIRM_PASSWORD" ]; then
    echo -e "${RED}Passwords do not match!${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Updating PostgreSQL password...${NC}"

# Update the password in PostgreSQL
sudo -u postgres psql -c "ALTER USER c2user WITH PASSWORD '$NEW_PASSWORD';"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PostgreSQL password updated${NC}"
else
    echo -e "${RED}✗ Failed to update PostgreSQL password${NC}"
    exit 1
fi

# Update the config file
echo -e "${YELLOW}Updating config file...${NC}"

if [ -f "config/database.conf" ]; then
    # Create backup
    cp config/database.conf config/database.conf.backup
    
    # Update the password in the config file
    cat > config/database.conf <<CONF
[postgresql]
host = localhost
database = c2db
user = c2user
password = ${NEW_PASSWORD}
port = 5432
CONF
    
    chmod 600 config/database.conf
    echo -e "${GREEN}✓ Config file updated${NC}"
else
    echo -e "${RED}✗ Config file not found at config/database.conf${NC}"
    echo "Creating new config file..."
    mkdir -p config
    cat > config/database.conf <<CONF
[postgresql]
host = localhost
database = c2db
user = c2user
password = ${NEW_PASSWORD}
port = 5432
CONF
    chmod 600 config/database.conf
    echo -e "${GREEN}✓ Config file created${NC}"
fi

echo ""
echo -e "${YELLOW}Testing database connection...${NC}"

# Test the connection
python3 -c "
import psycopg2
import configparser

config = configparser.ConfigParser()
config.read('config/database.conf')

try:
    conn = psycopg2.connect(
        host=config['postgresql']['host'],
        database=config['postgresql']['database'],
        user=config['postgresql']['user'],
        password=config['postgresql']['password'],
        port=int(config['postgresql']['port'])
    )
    print('${GREEN}✓ Database connection successful!${NC}')
    conn.close()
except Exception as e:
    print('${RED}✗ Connection failed:', e, '${NC}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Authentication Fixed Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "You can now restart your C2 detection system:"
    echo "  1. Stop current processes (Ctrl+C in the terminal where Flask is running)"
    echo "  2. Run: ./start_system.sh"
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Connection Test Failed${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo "Please check:"
    echo "  1. PostgreSQL is running: sudo systemctl status postgresql"
    echo "  2. Database exists: sudo -u postgres psql -c '\l' | grep c2db"
    echo "  3. User exists: sudo -u postgres psql -c '\du' | grep c2user"
fi
