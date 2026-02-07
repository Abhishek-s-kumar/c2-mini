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
