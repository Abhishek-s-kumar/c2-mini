import time
import requests
import socket
from datetime import datetime

print("Generating test network patterns...")
for i in range(10):
    # Simulate regular intervals (C2 beaconing pattern)
    try:
        requests.get(f"http://localhost:5000/api/status?test={i}", timeout=1)
    except:
        pass
    
    # Simulate external connections
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=1)
    except:
        pass
    
    print(f"Batch {i+1} at {datetime.now()}")
    time.sleep(30)  # Every 30 seconds - typical beacon interval
