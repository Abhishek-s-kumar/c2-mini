#!/usr/bin/env python3
"""
Analyze the TRUE C2 beacon with LOW ENTROPY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import json

print("="*60)
print("ANALYSIS OF TRUE C2 BEACON (LOW ENTROPY)")
print("="*60)

# Load the true C2 beacon
df = pd.read_csv('true_c2_beacon.csv')
time_series = df['bytes_per_second'].values
N = len(time_series)

print(f"Time series length: {N} samples")
print(f"Duration: {N} seconds ({N/60:.1f} minutes)")

# Apply FFT
yf = fft(time_series)
xf = fftfreq(N, 1)[:N//2]
magnitude = 2.0/N * np.abs(yf[:N//2])

# Find dominant frequency
dominant_idx = np.argmax(magnitude[1:]) + 1
dominant_freq = xf[dominant_idx]
dominant_period = 1/dominant_freq if dominant_freq > 0 else 0

print(f"\n1. FFT Results:")
print(f"   Dominant frequency: {dominant_freq:.3f} Hz")
print(f"   Corresponding period: {dominant_period:.1f} seconds")
print(f"   Expected: 0.1 Hz (10-second period)")

# Calculate P-score components
fft_peak = np.max(magnitude[1:]) / np.max(magnitude)

# Autocorrelation
autocorr = signal.correlate(time_series, time_series, mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr_max = np.max(autocorr[1:100]) / autocorr[0]

# Entropy (should be LOW for true beacon)
hist, _ = np.histogram(time_series, bins=10, density=True)
hist = hist + 1e-10
hist = hist / hist.sum()
entropy_val = -np.sum(hist * np.log2(hist))
entropy_norm = entropy_val / np.log2(len(hist))

# P-score formula
alpha, beta, gamma = 0.4, 0.4, 0.2
p_score = (alpha * fft_peak + 
           beta * autocorr_max + 
           gamma * (1 - entropy_norm))

print(f"\n2. P-Score Components:")
print(f"   FFT Peak (α={alpha}): {fft_peak:.3f}")
print(f"   Autocorrelation Max (β={beta}): {autocorr_max:.3f}")
print(f"   Normalized Entropy (H_norm): {entropy_norm:.3f}")
print(f"   Inverse Entropy (1-H_norm): {1 - entropy_norm:.3f}")
print(f"   Entropy Contribution (γ={gamma}): {gamma * (1 - entropy_norm):.3f}")

print(f"\n3. P-Score Calculation:")
print(f"   {alpha} × {fft_peak:.3f} + {beta} × {autocorr_max:.3f} + {gamma} × {1 - entropy_norm:.3f}")
print(f"   = {p_score:.3f}")

beacon_detected = p_score > 0.6
print(f"\n4. Detection Result:")
print(f"   P-Score: {p_score:.3f}")
print(f"   Threshold: 0.6")
print(f"   Beacon Detected: {'YES ✓' if beacon_detected else 'NO ✗'}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('True C2 Beacon Analysis - Low Entropy Signal', fontsize=16, fontweight='bold')

# Time series
ax = axes[0, 0]
ax.plot(df['seconds'][:60], time_series[:60], 'b-', linewidth=2, marker='o', markersize=4)
ax.set_title('Time Series (First 60s)\nClear 10-Second Beacon', fontweight='bold')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Bytes/sec')
ax.grid(True, alpha=0.3)
for i in range(0, 60, 10):
    ax.axvline(x=i, color='r', linestyle=':', alpha=0.5)

# FFT
ax = axes[0, 1]
ax.plot(xf[1:30], magnitude[1:30], 'r-', linewidth=2)
ax.set_title('FFT Magnitude Spectrum', fontweight='bold')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.grid(True, alpha=0.3)
ax.axvline(x=0.1, color='g', linestyle='--', label='Expected: 0.1 Hz')
ax.axvline(x=dominant_freq, color='orange', linestyle='--', label=f'Detected: {dominant_freq:.3f} Hz')
ax.legend()

# Autocorrelation
ax = axes[1, 0]
lags = np.arange(100)
ax.plot(lags, autocorr[:100]/autocorr[0], 'g-', linewidth=2)
ax.set_title('Autocorrelation (First 100 lags)', fontweight='bold')
ax.set_xlabel('Lag (seconds)')
ax.set_ylabel('Correlation')
ax.grid(True, alpha=0.3)
for i in range(10, 100, 10):
    ax.axvline(x=i, color='r', linestyle=':', alpha=0.5, label=f'{i}s' if i==10 else "")

# Component contributions
ax = axes[1, 1]
components = ['FFT Peak', 'Autocorr', 'Inverse Entropy']
raw_scores = [fft_peak, autocorr_max, 1 - entropy_norm]
weights = [alpha, beta, gamma]
contributions = [w * s for w, s in zip(weights, raw_scores)]

x = np.arange(len(components))
width = 0.35

bars1 = ax.bar(x - width/2, raw_scores, width, label='Raw Score', color='lightblue')
bars2 = ax.bar(x + width/2, contributions, width, label='Weighted Contribution', color='orange')

ax.set_title('P-Score Component Breakdown', fontweight='bold')
ax.set_xlabel('Component')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels([f'{c}\nw={w}' for c, w in zip(components, weights)])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 1)

# Add P-score text
ax.text(1.5, 0.85, f'P-Score = {p_score:.3f}', 
        ha='center', va='center', fontsize=14, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('true_c2_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: true_c2_analysis.png")

# Save results
results = {
    'p_score': float(p_score),
    'fft_peak': float(fft_peak),
    'autocorr_max': float(autocorr_max),
    'entropy_norm': float(entropy_norm),
    'dominant_frequency': float(dominant_freq),
    'dominant_period': float(dominant_period),
    'beacon_detected': str(beacon_detected),
    'explanation': 'True C2 beacon with low entropy (automated traffic)'
}

with open('true_c2_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: true_c2_results.json")

print(f"\n" + "="*60)
print("KEY INSIGHT FOR PROFESSOR:")
print("="*60)
print("Real C2 beacons have LOW ENTROPY because:")
print("1. Same packet sizes repeated")
print("2. Regular timing intervals")
print("3. Automated, predictable patterns")
print("")
print(f"This signal has:")
print(f"  • Entropy (H_norm): {entropy_norm:.3f} (LOW = good for beacon detection)")
print(f"  • Inverse entropy (1-H_norm): {1 - entropy_norm:.3f} (HIGH contribution to P-score)")
print(f"  • P-Score: {p_score:.3f} (Should be > 0.6 for detection)")

plt.show()
