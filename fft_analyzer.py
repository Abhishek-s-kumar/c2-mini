#!/usr/bin/env python3
"""
Simple FFT Analyzer
Demonstrates: Time series → FFT → Frequency analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def analyze_with_fft():
    print("="*60)
    print("FFT ANALYSIS DEMO")
    print("="*60)
    
    # Create or load time series
    try:
        df = pd.read_csv('time_series_demo.csv')
        time_series = df['bytes_per_second'].values
        print(f"Loaded time series: {len(time_series)} samples")
    except:
        print("Creating sample time series...")
        # Create a synthetic signal: 10-second period
        t = np.linspace(0, 120, 120)  # 120 seconds, 1 sample/sec
        signal = 50 + 40 * np.sin(2 * np.pi * t / 10)  # 10-second period
        noise = np.random.normal(0, 5, len(t))
        time_series = np.abs(signal + noise)
        df = pd.DataFrame({'seconds': t, 'bytes_per_second': time_series})
    
    # Apply FFT
    print("\n1. Applying FFT...")
    N = len(time_series)
    yf = fft(time_series)
    xf = fftfreq(N, 1)[:N//2]  # 1 sample per second
    
    # Get magnitude
    magnitude = 2.0/N * np.abs(yf[:N//2])
    
    # Find dominant frequency
    dominant_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
    dominant_freq = xf[dominant_idx]
    dominant_period = 1/dominant_freq if dominant_freq > 0 else 0
    
    print(f"   Samples: {N}")
    print(f"   Dominant frequency: {dominant_freq:.3f} Hz")
    print(f"   Corresponding period: {dominant_period:.1f} seconds")
    
    # Calculate P-score (simplified)
    print("\n2. Calculating Periodicity Score...")
    fft_peak = np.max(magnitude[1:]) / np.max(magnitude)  # Normalized
    autocorr = np.correlate(time_series, time_series, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr_max = np.max(autocorr[1:10]) / autocorr[0]  # First 10 lags
    
    # Simple entropy
    hist, _ = np.histogram(time_series, bins=10, density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist))
    entropy_norm = entropy / np.log2(len(hist))
    
    # P-score formula: α·FFT_peak + β·Autocorr_max + γ·(1 - H_norm)
    alpha, beta, gamma = 0.4, 0.4, 0.2
    p_score = (alpha * fft_peak + 
               beta * autocorr_max + 
               gamma * (1 - entropy_norm))
    
    print(f"   FFT Peak: {fft_peak:.3f}")
    print(f"   Autocorrelation Max: {autocorr_max:.3f}")
    print(f"   Normalized Entropy: {entropy_norm:.3f}")
    print(f"   P-Score: {p_score:.3f}")
    beacon_detected = p_score > 0.6
    print(f"   Beacon Detected: {'YES' if beacon_detected else 'NO'}")
    
    # Create visualization
    print("\n3. Creating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original time series
    ax = axes[0, 0]
    ax.plot(df['seconds'], time_series, 'b-', linewidth=2)
    ax.set_title('Original Time Series', fontweight='bold')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Bytes/sec')
    ax.grid(True, alpha=0.3)
    
    # FFT magnitude spectrum
    ax = axes[0, 1]
    ax.plot(xf[1:], magnitude[1:], 'r-', linewidth=2)
    ax.set_title('FFT Magnitude Spectrum', fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=dominant_freq, color='green', linestyle='--', 
               label=f'Peak: {dominant_freq:.3f} Hz')
    ax.legend()
    
    # Autocorrelation
    ax = axes[1, 0]
    lags = np.arange(len(autocorr[:30]))
    ax.plot(lags, autocorr[:30]/autocorr[0], 'g-', linewidth=2)
    ax.set_title('Autocorrelation (First 30 lags)', fontweight='bold')
    ax.set_xlabel('Lag (seconds)')
    ax.set_ylabel('Correlation')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=10, color='orange', linestyle='--', label='Expected: 10s')
    ax.legend()
    
    # Component contributions
    ax = axes[1, 1]
    components = ['FFT Peak', 'Autocorr', 'Inverse Entropy']
    contributions = [alpha * fft_peak, beta * autocorr_max, gamma * (1 - entropy_norm)]
    colors = ['blue', 'green', 'red']
    
    bars = ax.bar(components, contributions, color=colors, alpha=0.7)
    ax.set_title('P-Score Components', fontweight='bold')
    ax.set_ylabel('Contribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, contributions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Add total score
    ax.text(1.5, 0.8, f'Total P-Score: {p_score:.3f}', 
            fontsize=12, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('fft_analysis_demo.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: fft_analysis_demo.png")
    
    # Save results - FIXED: Convert boolean to string
    results = {
        'p_score': float(p_score),
        'fft_peak': float(fft_peak),
        'autocorr_max': float(autocorr_max),
        'entropy_norm': float(entropy_norm),
        'dominant_frequency': float(dominant_freq),
        'dominant_period': float(dominant_period),
        'beacon_detected': str(beacon_detected)  # Convert to string
    }
    
    with open('fft_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"• P-Score: {p_score:.3f}")
    print(f"• Dominant Period: {dominant_period:.1f} seconds")
    print(f"• Beacon Detected: {'YES' if beacon_detected else 'NO'}")
    print(f"• Files created:")
    print(f"   1. fft_analysis_demo.png")
    print(f"   2. fft_results.json")
    
    return results

if __name__ == "__main__":
    analyze_with_fft()
