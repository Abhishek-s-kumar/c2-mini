#!/usr/bin/env python3
"""
Create a true C2 beacon signal with LOW ENTROPY
Real beacons: same packet size, regular intervals, low randomness
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("CREATING TRUE C2 BEACON SIGNAL (LOW ENTROPY)")
print("="*60)

# Create 300 seconds of data
t = np.arange(0, 300, 1)

# True C2 beacon characteristics:
# 1. Same packet size every time
# 2. Regular intervals (every 10 seconds)
# 3. Low background traffic (just the beacon)

bytes_per_second = np.zeros_like(t, dtype=float)

# Add beacon: 500 bytes every 10 seconds, starting at 0
for i in range(0, 300, 10):
    bytes_per_second[i] = 500  # Beacon packet
    
# Add very small random background (0-5 bytes) to other seconds
background = np.random.randint(0, 6, len(t))
bytes_per_second = bytes_per_second + background

# Make it even more realistic: beacon sends at exactly the same time
# and has the same response size
response_delay = 1  # 1-second response
for i in range(0, 300, 10):
    if i + response_delay < 300:
        bytes_per_second[i + response_delay] = 100  # C2 server response

df = pd.DataFrame({
    'seconds': t,
    'bytes_per_second': bytes_per_second
})

df.to_csv('true_c2_beacon.csv', index=False)
print(f"Created true C2 beacon with {len(df)} samples")
print("Characteristics:")
print("  • Beacon every 10 seconds (0.1 Hz)")
print("  • Beacon packet: 500 bytes")
print("  • Server response: 100 bytes (1 second later)")
print("  • Background noise: 0-5 bytes")
print(f"  • Saved to: true_c2_beacon.csv")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t[:60], bytes_per_second[:60], 'b-', linewidth=2, marker='o', markersize=4)
plt.title('First 60 Seconds - True C2 Beacon', fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Bytes/sec')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t, bytes_per_second, 'g-', linewidth=1, alpha=0.7)
plt.title('Full 5-Minute Time Series', fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Bytes/sec')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('true_c2_beacon.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to: true_c2_beacon.png")

# Calculate entropy
from scipy.stats import entropy
hist, _ = np.histogram(bytes_per_second, bins=10, density=True)
hist = hist + 1e-10
hist = hist / hist.sum()
H = -np.sum(hist * np.log2(hist))
H_norm = H / np.log2(len(hist))

print(f"\nEntropy Analysis:")
print(f"  Raw entropy (H): {H:.3f}")
print(f"  Normalized entropy (H_norm): {H_norm:.3f}")
print(f"  Inverse entropy (1 - H_norm): {1 - H_norm:.3f}")
print(f"  Interpretation: {'LOW entropy (good for beacon)' if H_norm < 0.3 else 'HIGH entropy (bad for beacon)'}")

plt.show()
