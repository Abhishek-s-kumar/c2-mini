#!/bin/bash

# ============================================================================
# C2 Beacon Detection - Complete Production Integration Script
# ============================================================================
# This script sets up the entire production C2 detection system:
# 1. System dependencies (requires sudo)
# 2. PostgreSQL database setup (requires sudo)
# 3. Python environment and packages
# 4. Real-time monitoring system
# 5. Dashboard and continuous detection

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_section() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_step() {
    echo -e "\n${YELLOW}[$(date +%H:%M:%S)] $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        echo -e "${RED}Error: Do not run this script as root.${NC}"
        echo "Run with your normal user account (will use sudo when needed)."
        exit 1
    fi
}

# ============================================================================
# SECTION 1: SYSTEM DEPENDENCIES (requires sudo)
# ============================================================================
install_system_dependencies() {
    print_section "1. Installing System Dependencies"
    
    print_step "Updating package list..."
    sudo apt-get update
    
    print_step "Installing PostgreSQL and dependencies..."
    sudo apt-get install -y postgresql postgresql-contrib libpq-dev python3-venv python3-pip
    
    print_step "Installing Python system packages..."
    sudo apt-get install -y python3-dev build-essential
    
    print_success "System dependencies installed"
}

# ============================================================================
# SECTION 2: POSTGRESQL SETUP (requires sudo)
# ============================================================================
setup_postgresql() {
    print_section "2. Setting Up PostgreSQL Database"
    
    print_step "Starting PostgreSQL service..."
    sudo systemctl start postgresql
    sudo systemctl enable postgresql
    
    print_step "Creating database user and database..."
    read -sp "Enter password for c2user: " DB_PASSWORD
    echo
    
    # Execute as postgres user
    sudo -u postgres psql <<EOF
CREATE USER c2user WITH PASSWORD '${DB_PASSWORD}';
CREATE DATABASE c2db OWNER c2user;
GRANT ALL PRIVILEGES ON DATABASE c2db TO c2user;
\c c2db
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
EOF
    
    # Save password to config file (secured)
    mkdir -p config
    cat > config/database.conf <<CONF
[postgresql]
host = localhost
database = c2db
user = c2user
password = ${DB_PASSWORD}
port = 5432
CONF
    chmod 600 config/database.conf
    
    print_step "Creating conn_log table for Zeek data..."
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
    
    print_success "PostgreSQL database configured"
}

# ============================================================================
# SECTION 3: PYTHON ENVIRONMENT
# ============================================================================
setup_python_environment() {
    print_section "3. Setting Up Python Environment"
    
    print_step "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    print_step "Installing Python packages..."
    cat > requirements.txt <<REQ
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
seaborn>=0.11.0
psycopg2-binary>=2.9.0
flask>=2.0.0
requests>=2.25.0
scikit-learn>=0.24.0
python-dateutil>=2.8.0
REQ
    
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Python environment ready"
}

# ============================================================================
# SECTION 4: CREATE INTEGRATION SCRIPTS
# ============================================================================
create_integration_scripts() {
    print_section "4. Creating Integration Scripts"
    
    print_step "Creating real-time analyzer..."
    cat > real_time_analyzer.py <<'PYEOF'
#!/usr/bin/env python3
"""
Real-time C2 Beacon Detector
Connects to PostgreSQL and analyzes Zeek logs for beaconing patterns
"""

import psycopg2
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
import configparser
import sys

class RealTimeC2Detector:
    def __init__(self, config_file='config/database.conf'):
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.conn = None
        
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(
                host=self.config['postgresql']['host'],
                database=self.config['postgresql']['database'],
                user=self.config['postgresql']['user'],
                password=self.config['postgresql']['password'],
                port=int(self.config['postgresql']['port'])
            )
            print("[+] Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False
    
    def get_recent_connections(self, minutes=5):
        """Get recent connection logs"""
        query = """
        SELECT ts, id_orig_h, id_resp_h, resp_bytes, orig_bytes
        FROM conn_log 
        WHERE ts >= NOW() - INTERVAL '%s minutes'
        ORDER BY ts
        """
        
        try:
            df = pd.read_sql_query(query, self.conn, params=(minutes,))
            return df
        except Exception as e:
            print(f"[-] Query failed: {e}")
            return pd.DataFrame()
    
    def analyze_host(self, host_ip, df):
        """Analyze a specific host for beaconing patterns"""
        host_df = df[df['id_orig_h'] == host_ip].copy()
        
        if len(host_df) < 10:
            return None
        
        # Create time series
        host_df['ts'] = pd.to_datetime(host_df['ts'])
        host_df.set_index('ts', inplace=True)
        
        # Resample to 1-second intervals
        time_series = host_df['resp_bytes'].resample('1S').sum().fillna(0)
        
        # Calculate metrics (simplified P-Score)
        if len(time_series) > 20:
            # FFT analysis
            N = len(time_series)
            yf = np.fft.rfft(time_series.values)
            xf = np.fft.rfftfreq(N, d=1)
            magnitude = 2.0/N * np.abs(yf)
            
            # Skip DC component
            if len(magnitude) > 1:
                fft_peak = np.max(magnitude[1:]) / np.max(magnitude)
            else:
                fft_peak = 0
            
            # Simple autocorrelation
            autocorr = np.correlate(time_series.values, time_series.values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr_max = np.max(autocorr[1:20]) / autocorr[0] if autocorr[0] > 0 else 0
            
            # Entropy
            hist, _ = np.histogram(time_series.values, bins=10, density=True)
            hist = hist[hist > 0]
            if len(hist) > 0:
                hist = hist / hist.sum()
                entropy = -np.sum(hist * np.log2(hist))
                entropy_norm = entropy / np.log2(len(hist))
            else:
                entropy_norm = 0
            
            # P-Score
            alpha, beta, gamma = 0.4, 0.4, 0.2
            p_score = (alpha * fft_peak + beta * autocorr_max + gamma * (1 - entropy_norm))
            
            return {
                'host': host_ip,
                'p_score': float(p_score),
                'fft_peak': float(fft_peak),
                'autocorr_max': float(autocorr_max),
                'entropy_norm': float(entropy_norm),
                'samples': len(time_series),
                'detected': p_score > 0.7
            }
        
        return None

def main():
    detector = RealTimeC2Detector()
    
    if not detector.connect():
        print("[-] Exiting")
        return
    
    print("[+] Starting real-time analysis...")
    print("[+] Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Get recent data
            df = detector.get_recent_connections(minutes=2)
            
            if not df.empty:
                # Get unique hosts
                hosts = df['id_orig_h'].unique()[:5]  # Analyze top 5 hosts
                
                print(f"\n{'='*50}")
                print(f"Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Total connections: {len(df)}")
                print('='*50)
                
                for host in hosts:
                    result = detector.analyze_host(host, df)
                    if result:
                        status = "⚠️ BEACON DETECTED" if result['detected'] else "✓ Normal"
                        print(f"{host}: P-Score={result['p_score']:.3f} - {status}")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n[+] Analysis stopped")

if __name__ == "__main__":
    main()
PYEOF
    chmod +x real_time_analyzer.py
    print_success "Created: real_time_analyzer.py"
    
    print_step "Creating continuous monitor..."
    cat > monitor_c2.py <<'PYEOF'
#!/usr/bin/env python3
"""
Continuous C2 Beacon Monitoring Service
"""

import time
import json
import logging
from datetime import datetime
from real_time_analyzer import RealTimeC2Detector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/c2_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class C2Monitor:
    def __init__(self):
        self.detector = RealTimeC2Detector()
        self.alerts = []
        
    def start(self, interval_seconds=60):
        """Start continuous monitoring"""
        if not self.detector.connect():
            logger.error("Failed to connect to database")
            return
        
        logger.info(f"Starting C2 beacon monitoring (interval: {interval_seconds}s)")
        
        while True:
            try:
                self.run_detection_cycle()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in detection cycle: {e}")
                time.sleep(10)
    
    def run_detection_cycle(self):
        """Run one detection cycle"""
        try:
            # Get recent data
            df = self.detector.get_recent_connections(minutes=5)
            
            if df.empty:
                logger.warning("No data available")
                return
            
            # Analyze each host
            for host in df['id_orig_h'].unique()[:10]:
                result = self.detector.analyze_host(host, df)
                
                if result and result['detected']:
                    # Create alert
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'host': result['host'],
                        'p_score': result['p_score'],
                        'details': result
                    }
                    
                    self.alerts.append(alert)
                    
                    # Log alert
                    logger.warning(
                        f"C2 BEACON DETECTED - Host: {result['host']}, "
                        f"P-Score: {result['p_score']:.3f}"
                    )
                    
                    # Save alert to file
                    with open('output/alerts.json', 'a') as f:
                        json.dump(alert, f)
                        f.write('\n')
        
        except Exception as e:
            logger.error(f"Detection error: {e}")

if __name__ == "__main__":
    monitor = C2Monitor()
    monitor.start(interval_seconds=30)
PYEOF
    chmod +x monitor_c2.py
    print_success "Created: monitor_c2.py"
    
    print_step "Creating test beacon generator..."
    cat > test_beacon.sh <<'SHEOF'
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
SHEOF
    chmod +x test_beacon.sh
    print_success "Created: test_beacon.sh"
    
    print_step "Creating dashboard..."
    cat > dashboard.py <<'PYEOF'
#!/usr/bin/env python3
"""
Simple Web Dashboard for C2 Detection
"""

from flask import Flask, render_template, jsonify, request
import json
import threading
import time
from datetime import datetime
from real_time_analyzer import RealTimeC2Detector

app = Flask(__name__)
detector = RealTimeC2Detector()

# Store latest results
system_status = {
    'last_update': datetime.now().isoformat(),
    'total_alerts': 0,
    'active_detections': [],
    'status': 'running'
}

def background_monitoring():
    """Background thread for continuous detection"""
    if not detector.connect():
        return
    
    while True:
        try:
            # Get recent alerts
            try:
                with open('output/alerts.json', 'r') as f:
                    alerts = [json.loads(line) for line in f.readlines()[-10:]]
            except:
                alerts = []
            
            # Update status
            system_status['last_update'] = datetime.now().isoformat()
            system_status['total_alerts'] = len(alerts)
            system_status['active_detections'] = alerts[-5:]  # Last 5 alerts
            
            time.sleep(10)
        except:
            time.sleep(5)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', status=system_status)

@app.route('/api/status')
def get_status():
    """API endpoint for system status"""
    return jsonify(system_status)

@app.route('/api/analyze')
def analyze_now():
    """Force immediate analysis"""
    if detector.connect():
        df = detector.get_recent_connections(minutes=2)
        results = []
        
        for host in df['id_orig_h'].unique()[:5]:
            result = detector.analyze_host(host, df)
            if result:
                results.append(result)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'total_connections': len(df)
        })
    
    return jsonify({'error': 'Database connection failed'})

@app.route('/api/alerts')
def get_alerts():
    """Get all alerts"""
    try:
        with open('output/alerts.json', 'r') as f:
            alerts = [json.loads(line) for line in f]
        return jsonify({'alerts': alerts[-20:]})  # Last 20 alerts
    except:
        return jsonify({'alerts': []})

@app.route('/api/clear_alerts', methods=['POST'])
def clear_alerts():
    """Clear all alerts"""
    try:
        open('output/alerts.json', 'w').close()
        return jsonify({'success': True})
    except:
        return jsonify({'success': False})

if __name__ == '__main__':
    # Start background monitoring
    monitor_thread = threading.Thread(target=background_monitoring, daemon=True)
    monitor_thread.start()
    
    # Create dashboard template
    import os
    os.makedirs('templates', exist_ok=True)
    
    with open('templates/dashboard.html', 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <title>C2 Beacon Detection Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #f8f9fa; }
        .card { margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .alert-critical { border-left: 5px solid #dc3545; }
        .alert-warning { border-left: 5px solid #ffc107; }
        .alert-normal { border-left: 5px solid #28a745; }
    </style>
</head>
<body>
    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4">C2 Beacon Detection Dashboard</h1>
            </div>
        </div>
        
        <div class="row">
            <!-- Status Panel -->
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0">System Status</h5>
                    </div>
                    <div class="card-body" id="statusPanel">
                        <p><strong>Last Update:</strong> <span id="lastUpdate">Loading...</span></p>
                        <p><strong>Total Alerts:</strong> <span id="totalAlerts">0</span></p>
                        <p><strong>Status:</strong> <span class="badge bg-success" id="systemStatus">Running</span></p>
                        <button class="btn btn-sm btn-primary mt-2" onclick="analyzeNow()">Analyze Now</button>
                    </div>
                </div>
            </div>
            
            <!-- Recent Alerts -->
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header bg-warning text-dark">
                        <h5 class="mb-0">Recent Alerts</h5>
                    </div>
                    <div class="card-body">
                        <div id="alertsContainer">
                            <p class="text-muted">No alerts detected</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Active Detections -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h5 class="mb-0">Active Detections</h5>
                    </div>
                    <div class="card-body">
                        <table class="table table-striped" id="detectionsTable">
                            <thead>
                                <tr>
                                    <th>Host</th>
                                    <th>P-Score</th>
                                    <th>FFT Peak</th>
                                    <th>Entropy</th>
                                    <th>Status</th>
                                    <th>Time</th>
                                </tr>
                            </thead>
                            <tbody id="detectionsBody">
                                <!-- Filled by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Update dashboard every 5 seconds
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('lastUpdate').textContent = new Date(data.last_update).toLocaleString();
                    document.getElementById('totalAlerts').textContent = data.total_alerts;
                    
                    // Update alerts
                    if (data.active_detections && data.active_detections.length > 0) {
                        let alertsHtml = '';
                        data.active_detections.forEach(alert => {
                            const time = new Date(alert.timestamp).toLocaleTimeString();
                            const badgeClass = alert.details.p_score > 0.8 ? 'danger' : 'warning';
                            alertsHtml += `
                                <div class="alert alert-${alert.details.detected ? 'warning' : 'light'}">
                                    <strong>${alert.host}</strong> - P-Score: ${alert.details.p_score.toFixed(3)}
                                    <span class="badge bg-${badgeClass} float-end">${time}</span>
                                </div>
                            `;
                        });
                        document.getElementById('alertsContainer').innerHTML = alertsHtml;
                    }
                });
            
            fetch('/api/alerts')
                .then(response => response.json())
                .then(data => {
                    const tableBody = document.getElementById('detectionsBody');
                    tableBody.innerHTML = '';
                    
                    if (data.alerts && data.alerts.length > 0) {
                        data.alerts.forEach(alert => {
                            const row = document.createElement('tr');
                            const statusClass = alert.details.p_score > 0.7 ? 
                                (alert.details.p_score > 0.9 ? 'bg-danger text-white' : 'bg-warning') : '';
                            
                            row.innerHTML = `
                                <td>${alert.host}</td>
                                <td>${alert.details.p_score.toFixed(3)}</td>
                                <td>${alert.details.fft_peak.toFixed(3)}</td>
                                <td>${alert.details.entropy_norm.toFixed(3)}</td>
                                <td><span class="badge ${alert.details.detected ? 'bg-danger' : 'bg-success'}">
                                    ${alert.details.detected ? 'BEACON' : 'Normal'}
                                </span></td>
                                <td>${new Date(alert.timestamp).toLocaleTimeString()}</td>
                            `;
                            if (statusClass) row.className = statusClass;
                            tableBody.appendChild(row);
                        });
                    }
                });
        }
        
        function analyzeNow() {
            fetch('/api/analyze')
                .then(response => response.json())
                .then(data => {
                    alert('Analysis complete! Check console for details.');
                    updateDashboard();
                });
        }
        
        // Initial load and periodic updates
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>''')
    
    print_success "Created: dashboard.py"
}

# ============================================================================
# SECTION 5: CREATE CONFIGURATION FILES
# ============================================================================
create_configuration() {
    print_section "5. Creating Configuration Files"
    
    print_step "Creating project structure..."
    mkdir -p {config,logs,output,test_data,templates,reports}
    
    print_step "Creating main configuration..."
    cat > config/project.conf <<'CONF'
[project]
name = C2 Beacon Detection System
version = 1.0.0
author = Security Research Team

[detection]
p_score_threshold = 0.7
analysis_interval = 30
monitoring_duration = 300

[network]
interface = enp0s8
beacon_ip = 192.168.56.20
monitored_subnet = 192.168.56.0/24

[paths]
logs_dir = logs/
output_dir = output/
reports_dir = reports/
CONF
    
    print_step "Creating Zeek integration script..."
    cat > import_zeek_logs.py <<'PYEOF'
#!/usr/bin/env python3
"""
Import Zeek conn.log files into PostgreSQL
"""

import pandas as pd
import psycopg2
import sys
import os
from datetime import datetime

def import_zeek_log(log_file):
    """Import a Zeek conn.log file into PostgreSQL"""
    
    # Read the log file
    # Zeek logs are tab-separated with # as comment lines
    try:
        df = pd.read_csv(
            log_file,
            sep='\t',
            comment='#',
            header=None,
            usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            names=[
                'ts', 'uid', 'id_orig_h', 'id_orig_p', 'id_resp_h', 'id_resp_p',
                'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
                'conn_state', 'local_orig', 'local_resp'
            ]
        )
        
        # Convert timestamp
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
        
        # Fill missing values
        df.fillna({
            'service': '',
            'duration': 0,
            'orig_bytes': 0,
            'resp_bytes': 0,
            'conn_state': ''
        }, inplace=True)
        
        print(f"[+] Loaded {len(df)} records from {log_file}")
        return df
        
    except Exception as e:
        print(f"[-] Failed to read {log_file}: {e}")
        return None

def save_to_database(df, config_file='config/database.conf'):
    """Save DataFrame to PostgreSQL"""
    import configparser
    config = configparser.ConfigParser()
    config.read(config_file)
    
    try:
        conn = psycopg2.connect(
            host=config['postgresql']['host'],
            database=config['postgresql']['database'],
            user=config['postgresql']['user'],
            password=config['postgresql']['password']
        )
        
        # Use pandas to_sql with chunksize for large files
        df.to_sql('conn_log', conn, if_exists='append', index=False, method='multi', chunksize=1000)
        
        print(f"[+] Successfully imported {len(df)} records to database")
        conn.close()
        return True
        
    except Exception as e:
        print(f"[-] Database import failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 import_zeek_logs.py <conn.log file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not os.path.exists(log_file):
        print(f"[-] File not found: {log_file}")
        sys.exit(1)
    
    print(f"[*] Importing {log_file}...")
    df = import_zeek_log(log_file)
    
    if df is not None and not df.empty:
        save_to_database(df)
PYEOF
    chmod +x import_zeek_logs.py
    print_success "Created: import_zeek_logs.py"
    
    print_step "Creating startup script..."
    cat > start_system.sh <<'SHEOF'
#!/bin/bash
# Start the complete C2 detection system

echo "Starting C2 Detection System..."
echo "================================"

# Start PostgreSQL if not running
sudo systemctl start postgresql

# Activate virtual environment
source venv/bin/activate

# Start the dashboard in background
echo "[1] Starting web dashboard..."
python3 dashboard.py &
DASHBOARD_PID=$!
echo "    Dashboard PID: $DASHBOARD_PID"
echo "    Access at: http://localhost:5000"

# Start the monitor in background
echo "[2] Starting C2 monitor..."
python3 monitor_c2.py &
MONITOR_PID=$!
echo "    Monitor PID: $MONITOR_PID"

echo ""
echo "[✓] System started successfully!"
echo ""
echo "To stop the system, run:"
echo "    kill $DASHBOARD_PID $MONITOR_PID"
echo ""
echo "Or run: ./stop_system.sh"
echo ""
echo "To test beacon detection, run in another terminal:"
echo "    ./test_beacon.sh"
echo ""
SHEOF
    chmod +x start_system.sh
    
    cat > stop_system.sh <<'SHEOF'
#!/bin/bash
# Stop the C2 detection system

echo "Stopping C2 Detection System..."
echo "================================"

# Find and kill dashboard and monitor
pkill -f "dashboard.py" 2>/dev/null
pkill -f "monitor_c2.py" 2>/dev/null

echo "[✓] System stopped"
SHEOF
    chmod +x stop_system.sh
    
    print_success "Created startup/shutdown scripts"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================
main() {
    print_section "C2 BEACON DETECTION - PRODUCTION INTEGRATION"
    echo "This script will set up the complete C2 detection system."
    echo ""
    echo "Requirements:"
    echo "1. Sudo privileges (for PostgreSQL and system packages)"
    echo "2. Internet connection"
    echo "3. At least 2GB free disk space"
    echo ""
    echo "The script will:"
    echo "  • Install system dependencies (sudo required)"
    echo "  • Set up PostgreSQL database (sudo required)"
    echo "  • Create Python virtual environment"
    echo "  • Install required Python packages"
    echo "  • Create all necessary scripts"
    echo ""
    
    read -p "Do you want to continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    # Check not running as root
    check_root
    
    # Track execution time
    START_TIME=$(date +%s)
    
    # Execute sections
    install_system_dependencies
    setup_postgresql
    setup_python_environment
    create_integration_scripts
    create_configuration
    
    # Calculate execution time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    print_section "SETUP COMPLETE"
    echo "Setup completed in ${DURATION} seconds."
    echo ""
    echo "What to do next:"
    echo ""
    echo "1. Start the system:"
    echo "   ./start_system.sh"
    echo ""
    echo "2. Access the dashboard:"
    echo "   Open browser to: http://localhost:5000"
    echo ""
    echo "3. Test beacon detection:"
    echo "   In another terminal, run: ./test_beacon.sh"
    echo ""
    echo "4. Import existing Zeek logs:"
    echo "   python3 import_zeek_logs.py /path/to/conn.log"
    echo ""
    echo "5. View logs:"
    echo "   tail -f logs/c2_monitor.log"
    echo ""
    echo "System files created:"
    echo "  • real_time_analyzer.py - Main detection logic"
    echo "  • monitor_c2.py - Continuous monitoring service"
    echo "  • dashboard.py - Web dashboard"
    echo "  • test_beacon.sh - Test beacon generator"
    echo "  • start_system.sh / stop_system.sh - Control scripts"
    echo ""
    echo "For more information, check the README.md file."
}

# Run main function
main