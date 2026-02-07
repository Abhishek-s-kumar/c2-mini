#!/usr/bin/env python3
"""
Analyze the ideal time series with the P-score formula
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal
import json

print("="*60)
print("ANALYSIS OF IDEAL PERIODIC TIME SERIES")
print("="*60)

# Load the ideal time series
df = pd.read_csv('ideal_time_series.csv')
time_series = df['bytes_per_second'].values
N = len(time_series)

print(f"Time series length: {N} samples")
print(f"Duration: {N} seconds ({N/60:.1f} minutes)")

# Apply FFT
yf = fft(time_series)
xf = fftfreq(N, 1)[:N//2]
magnitude = 2.0/N * np.abs(yf[:N//2])

# Find dominant frequency
dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
dominant_freq = xf[dominant_idx]
dominant_period = 1/dominant_freq

print(f"\n1. FFT Results:")
print(f"   Dominant frequency: {dominant_freq:.3f} Hz")
print(f"   Corresponding period: {dominant_period:.1f} seconds")
print(f"   Expected: 0.1 Hz (10-second period)")

# Calculate P-score components
fft_peak = np.max(magnitude[1:]) / np.max(magnitude)

# Autocorrelation
autocorr = signal.correlate(time_series, time_series, mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr_max = np.max(autocorr[1:100]) / autocorr[0]  # Check first 100 lags

# Entropy
hist, _ = np.histogram(time_series, bins=20, density=True)
hist = hist + 1e-10
hist = hist / hist.sum()
entropy = -np.sum(hist * np.log2(hist))
entropy_norm = entropy / np.log2(len(hist))

# P-score formula (from research paper)
alpha, beta, gamma = 0.4, 0.4, 0.2
p_score = (alpha * fft_peak + 
           beta * autocorr_max + 
           gamma * (1 - entropy_norm))

print(f"\n2. P-Score Components:")
print(f"   FFT Peak (α={alpha}): {fft_peak:.3f}")
print(f"   Autocorrelation Max (β={beta}): {autocorr_max:.3f}")
print(f"   Normalized Entropy: {entropy_norm:.3f}")
print(f"   Inverse Entropy (1-H): {1 - entropy_norm:.3f}")
print(f"   Entropy Contribution (γ={gamma}): {gamma * (1 - entropy_norm):.3f}")

print(f"\n3. P-Score Calculation:")
print(f"   {alpha} × {fft_peak:.3f} + {beta} × {autocorr_max:.3f} + {gamma} × {1 - entropy_norm:.3f}")
print(f"   = {p_score:.3f}")

beacon_detected = p_score > 0.6
print(f"\n4. Detection Result:")
print(f"   P-Score: {p_score:.3f}")
print(f"   Threshold: 0.6")
print(f"   Beacon Detected: {'YES ✓' if beacon_detected else 'NO ✗'}")

# Create comprehensive visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('Ideal Periodic Signal Analysis - High P-Score Example', fontsize=16, fontweight='bold')

# 1. Time series
ax = axes[0, 0]
ax.plot(df['seconds'][:120], time_series[:120], 'b-', linewidth=2)
ax.set_title('Time Series (First 2 Minutes)', fontweight='bold')
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Bytes/sec')
ax.grid(True, alpha=0.3)
ax.axvline(x=10, color='r', linestyle=':', alpha=0.5, label='10s')
ax.axvline(x=20, color='r', linestyle=':', alpha=0.5)
ax.axvline(x=30, color='r', linestyle=':', alpha=0.5)
ax.legend()

# 2. FFT
ax = axes[0, 1]
ax.plot(xf[1:50], magnitude[1:50], 'r-', linewidth=2)
ax.set_title('FFT Magnitude Spectrum', fontweight='bold')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.grid(True, alpha=0.3)
ax.axvline(x=0.1, color='g', linestyle='--', label='Expected: 0.1 Hz')
ax.axvline(x=dominant_freq, color='orange', linestyle='--', label=f'Detected: {dominant_freq:.3f} Hz')
ax.legend()

# 3. Autocorrelation
ax = axes[1, 0]
lags = np.arange(150)
ax.plot(lags, autocorr[:150]/autocorr[0], 'g-', linewidth=2)
ax.set_title('Autocorrelation (First 150 lags)', fontweight='bold')
ax.set_xlabel('Lag (seconds)')
ax.set_ylabel('Correlation')
ax.grid(True, alpha=0.3)
for i in range(1, 6):
    ax.axvline(x=i*10, color='r', linestyle=':', alpha=0.5, label=f'{i*10}s' if i==1 else "")
ax.legend()

# 4. Histogram (for entropy)
ax = axes[1, 1]
ax.hist(time_series, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax.set_title(f'Histogram (Entropy: {entropy_norm:.3f})', fontweight='bold')
ax.set_xlabel('Bytes/sec')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3)

# 5. Component contributions
ax = axes[2, 0]
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

# Add values on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 6. Final P-score
ax = axes[2, 1]
ax.bar(['P-Score'], [p_score], color='purple', alpha=0.7, width=0.5)
ax.axhline(y=0.6, color='r', linestyle='--', linewidth=2, label='Detection Threshold (0.6)')
ax.set_title(f'Final P-Score: {p_score:.3f}', fontweight='bold')
ax.set_ylabel('Score')
ax.set_ylim(0, 1)
if p_score > 0.6:
    ax.text(0, p_score/2, 'BEACON\nDETECTED', 
            ha='center', va='center', fontsize=14, fontweight='bold', 
            color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
else:
    ax.text(0, p_score/2, 'NO BEACON\nDETECTED', 
            ha='center', va='center', fontsize=14, fontweight='bold', 
            color='white', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ideal_analysis_high_pscore.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: ideal_analysis_high_pscore.png")

# Save results
results = {
    'p_score': float(p_score),
    'fft_peak': float(fft_peak),
    'autocorr_max': float(autocorr_max),
    'entropy_norm': float(entropy_norm),
    'dominant_frequency': float(dominant_freq),
    'dominant_period': float(dominant_period),
    'beacon_detected': str(beacon_detected),
    'explanation': 'Ideal periodic signal with clear 10-second period'
}

with open('ideal_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: ideal_results.json")

plt.show()

print(f"\n" + "="*60)
print("COMPARISON WITH SPARSE SIGNAL")
print("="*60)
print("Sparse Signal (previous demo):")
print("  • P-Score: 0.508 (Beacon NOT detected)")
print("  • Problem: Too sparse, FFT confused")
print("")
print("Ideal Periodic Signal (this demo):")
print(f"  • P-Score: {p_score:.3f} (Beacon {'DETECTED' if beacon_detected else 'NOT detected'})")
print("  • Solution: Clear periodic pattern in traffic")
