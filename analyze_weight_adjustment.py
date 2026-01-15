#!/usr/bin/env python3
"""
Show how adjusting weights affects P-Score
This demonstrates the importance of proper weight tuning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("="*60)
print("P-SCORE WEIGHT ADJUSTMENT ANALYSIS")
print("="*60)

# Load the true C2 beacon
df = pd.read_csv('true_c2_beacon.csv')
time_series = df['bytes_per_second'].values

# Calculate base components
N = len(time_series)
yf = np.fft.fft(time_series)
xf = np.fft.fftfreq(N, 1)[:N//2]
magnitude = 2.0/N * np.abs(yf[:N//2])
fft_peak = np.max(magnitude[1:]) / np.max(magnitude)

# Autocorrelation
autocorr = np.correlate(time_series, time_series, mode='full')
autocorr = autocorr[len(autocorr)//2:]
autocorr_max = np.max(autocorr[1:100]) / autocorr[0]

# Entropy
hist, _ = np.histogram(time_series, bins=10, density=True)
hist = hist + 1e-10
hist = hist / hist.sum()
entropy_val = -np.sum(hist * np.log2(hist))
entropy_norm = entropy_val / np.log2(len(hist))
inverse_entropy = 1 - entropy_norm

print(f"Component values:")
print(f"  FFT Peak: {fft_peak:.3f}")
print(f"  Autocorrelation Max: {autocorr_max:.3f}")
print(f"  Inverse Entropy (1-H_norm): {inverse_entropy:.3f}")

# Test different weight combinations
weight_combinations = [
    ("Research Paper", 0.4, 0.4, 0.2),
    ("FFT Heavy", 0.6, 0.3, 0.1),
    ("Autocorr Heavy", 0.3, 0.6, 0.1),
    ("Entropy Heavy", 0.2, 0.2, 0.6),
    ("Balanced", 0.33, 0.33, 0.34),
]

results = []
for name, alpha, beta, gamma in weight_combinations:
    p_score = alpha * fft_peak + beta * autocorr_max + gamma * inverse_entropy
    results.append({
        'name': name,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'p_score': p_score,
        'detected': p_score > 0.6
    })
    print(f"\n{name}:")
    print(f"  Weights: α={alpha}, β={beta}, γ={gamma}")
    print(f"  Calculation: {alpha}×{fft_peak:.3f} + {beta}×{autocorr_max:.3f} + {gamma}×{inverse_entropy:.3f}")
    print(f"  P-Score: {p_score:.3f}")
    print(f"  Beacon Detected: {'YES' if p_score > 0.6 else 'NO'}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart of P-scores with different weights
ax = axes[0]
names = [r['name'] for r in results]
p_scores = [r['p_score'] for r in results]
colors = ['green' if s > 0.6 else 'red' for s in p_scores]

bars = ax.bar(names, p_scores, color=colors, alpha=0.7)
ax.axhline(y=0.6, color='r', linestyle='--', linewidth=2, label='Detection Threshold (0.6)')
ax.set_title('P-Score with Different Weight Combinations', fontweight='bold')
ax.set_ylabel('P-Score')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, axis='y')
ax.legend()

# Add value labels
for bar, score in zip(bars, p_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{score:.3f}', ha='center', va='bottom')

# Radar chart of weights
ax = axes[1]
categories = ['FFT Weight (α)', 'Autocorr Weight (β)', 'Entropy Weight (γ)']

for result in results:
    values = [result['alpha'], result['beta'], result['gamma']]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]  # Close the circle
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=result['name'])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('Weight Distributions', fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
ax.grid(True)

plt.tight_layout()
plt.savefig('weight_adjustment_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: weight_adjustment_analysis.png")

print(f"\n" + "="*60)
print("TEACHING POINT FOR PROFESSOR:")
print("="*60)
print("The research paper's weight choice (α=0.4, β=0.4, γ=0.2) might need adjustment")
print("for specific network environments. This shows the importance of:")
print("1. Tuning weights based on your specific traffic")
print("2. Validating the formula with real C2 beacon data")
print("3. Understanding that different detection scenarios may require different weights")

plt.show()
