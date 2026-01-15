#!/usr/bin/env python3
"""
Create an ideal time series for demonstration
This shows what happens with clearer periodic traffic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("CREATING IDEAL DEMONSTRATION TIME SERIES")
print("="*60)

# Create a clear periodic signal: 10-second period
t = np.arange(0, 300, 1)  # 300 seconds = 5 minutes
frequency = 0.1  # 0.1 Hz = 10 second period

# Create a clear sine wave (plus some noise to make it realistic)
base_traffic = 50 + 30 * np.random.randn(len(t))  # Background traffic
periodic_component = 100 * np.sin(2 * np.pi * frequency * t)  # Strong periodic signal
signal = base_traffic + periodic_component

# Make all values positive (bytes can't be negative)
signal = np.abs(signal)

# Create DataFrame
df = pd.DataFrame({
    'seconds': t,
    'bytes_per_second': signal.astype(int)
})

# Save
df.to_csv('ideal_time_series.csv', index=False)
print(f"Created ideal time series with {len(df)} samples")
print(f"Clear periodic component: {frequency} Hz ({1/frequency} second period)")
print(f"Saved to: ideal_time_series.csv")

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t[:60], signal[:60], 'b-', linewidth=2)  # First 60 seconds
plt.title('First 60 Seconds (Clear 10s Period)', fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Bytes/sec')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(t, signal, 'g-', linewidth=1, alpha=0.7)
plt.title('Full 5-Minute Time Series', fontweight='bold')
plt.xlabel('Time (seconds)')
plt.ylabel('Bytes/sec')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ideal_time_series.png', dpi=150, bbox_inches='tight')
print(f"Visualization saved to: ideal_time_series.png")
plt.show()
