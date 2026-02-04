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
                safe_result = {k: (bool(v) if k=="detected" else float(v) if isinstance(v, (int, float)) else v) for k, v in result.items()}
                results.append(safe_result)
        
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
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)
