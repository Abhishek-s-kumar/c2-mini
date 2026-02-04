#!/bin/bash

echo "═══════════════════════════════════════"
echo "  C2 Detection System Status Check"
echo "═══════════════════════════════════════"
echo ""

# Check if dashboard is running
if pgrep -f "dashboard.py" > /dev/null; then
    DASH_PID=$(pgrep -f "dashboard.py")
    echo "✓ Dashboard running (PID: $DASH_PID)"
    echo "  URL: http://localhost:5000"
else
    echo "✗ Dashboard NOT running"
    echo ""
    echo "Starting dashboard now..."
    cd ~/Desktop/c2/c2-mini
    source venv/bin/activate
    python3 dashboard.py &
    sleep 2
    echo "✓ Dashboard started on http://localhost:5000"
fi

echo ""

# Check if monitor is running  
if pgrep -f "monitor_c2.py" > /dev/null; then
    MON_PID=$(pgrep -f "monitor_c2.py")
    echo "✓ Monitor running (PID: $MON_PID)"
else
    echo "✗ Monitor NOT running"
    echo ""
    echo "Starting monitor now..."
    cd ~/Desktop/c2/c2-mini
    source venv/bin/activate
    python3 monitor_c2.py &
    sleep 1
    echo "✓ Monitor started"
fi

echo ""
echo "═══════════════════════════════════════"
echo ""
echo "Fresh data generated! Now:"
echo "1. Open: http://localhost:5000"
echo "2. Click 'Analyze Now'"
echo "3. See the detections!"
echo ""
