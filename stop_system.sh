#!/bin/bash
# Stop the C2 detection system

echo "Stopping C2 Detection System..."
echo "================================"

# Find and kill dashboard and monitor
pkill -f "dashboard.py" 2>/dev/null
pkill -f "monitor_c2.py" 2>/dev/null

echo "[✓] System stopped"
