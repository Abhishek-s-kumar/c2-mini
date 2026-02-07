#!/bin/bash

# ============================================================================
# C2 Beacon Detection - Clean Setup Script
# ============================================================================

set -e  # Exit on any error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}C2 Beacon Detection Project Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Step 1: Check if we're in the right directory
echo -e "\n${YELLOW}[1/7] Checking environment...${NC}"
pwd

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 not found!${NC}"
    exit 1
fi
echo -e "✓ Python3: $(python3 --version)"

# Step 2: Create virtual environment
echo -e "\n${YELLOW}[2/7] Creating virtual environment...${NC}"
python3 -m venv venv || {
    echo -e "${RED}Failed to create virtual environment${NC}"
    exit 1
}
echo -e "✓ Virtual environment created"

# Step 3: Activate and install packages
echo -e "\n${YELLOW}[3/7] Installing Python packages...${NC}"
source venv/bin/activate
pip install --upgrade pip

# Create requirements.txt
cat > requirements.txt << 'REQ'
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
seaborn>=0.11.0
REQ

pip install -r requirements.txt
echo -e "✓ Packages installed"

# Step 4: Create project structure
echo -e "\n${YELLOW}[4/7] Creating project structure...${NC}"
mkdir -p {logs,output,test_data,reports}
echo -e "✓ Directories created"

# Step 5: Create a simple time series extractor
echo -e "\n${YELLOW}[5/7] Creating time series extractor...${NC}"

cat > time_series_extractor.py << 'PYEOF'
#!/usr/bin/env python3
"""
Simple Time Series Extractor
Demonstrates: Raw logs → Time series conversion
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def demonstrate_extraction():
    print("="*60)
    print("DEMO: Converting Logs to Time Series")
    print("="*60)
    
    # Step 1: Create sample logs
    print("\n1. Creating sample beacon logs...")
    logs = []
    base_time = datetime.now()
    
    for i in range(120):  # 120 seconds of data
        timestamp = base_time.timestamp() + i
        dt = datetime.fromtimestamp(timestamp)
        
        # Beacon every 10 seconds
        if i % 10 == 0:
            bytes_val = 500  # Beacon packet
            is_beacon = True
        else:
            bytes_val = np.random.randint(1, 30)  # Background noise
            is_beacon = False
        
        log_entry = {
            "timestamp": dt.isoformat(),
            "src_ip": "192.168.1.100",
            "dst_ip": "10.0.0.1",
            "bytes_toserver": bytes_val,
            "bytes_toclient": 100 if is_beacon else 0,
            "is_beacon": is_beacon
        }
        logs.append(log_entry)
    
    print(f"   Created {len(logs)} log entries")
    print("   Beacon packets: every 10 seconds")
    
    # Step 2: Convert to DataFrame
    print("\n2. Converting to DataFrame...")
    df = pd.DataFrame(logs)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   DataFrame shape: {df.shape}")
    
    # Step 3: Create time series
    print("\n3. Creating time series (bytes per second)...")
    df.set_index('timestamp', inplace=True)
    
    # Resample to 1-second intervals
    time_series = df['bytes_toserver'].resample('1S').sum().fillna(0)
    
    # Create final DataFrame with seconds
    start_time = time_series.index[0]
    seconds_from_start = (time_series.index - start_time).total_seconds()
    
    result_df = pd.DataFrame({
        'seconds': seconds_from_start,
        'bytes_per_second': time_series.values,
        'timestamp': time_series.index
    })
    
    print(f"   Time series length: {len(result_df)} samples")
    print(f"   Duration: {seconds_from_start[-1]:.0f} seconds")
    print(f"   Total bytes: {result_df['bytes_per_second'].sum():.0f}")
    
    # Step 4: Save to CSV
    print("\n4. Saving results...")
    result_df.to_csv('time_series_demo.csv', index=False)
    print("   ✓ Saved: time_series_demo.csv")
    
    # Step 5: Plot
    print("\n5. Creating visualization...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(result_df['seconds'], result_df['bytes_per_second'], 'b-', linewidth=2)
    plt.title('Time Series: Bytes per Second', fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Bytes/sec')
    plt.grid(True, alpha=0.3)
    
    # Highlight beacon spikes
    beacon_indices = result_df[result_df['bytes_per_second'] > 100].index
    plt.scatter(result_df.loc[beacon_indices, 'seconds'], 
                result_df.loc[beacon_indices, 'bytes_per_second'], 
                color='red', s=100, zorder=5, label='Beacon packets')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    # Cumulative bytes
    cumulative = result_df['bytes_per_second'].cumsum()
    plt.plot(result_df['seconds'], cumulative, 'g-', linewidth=2)
    plt.title('Cumulative Bytes Over Time', fontweight='bold')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cumulative Bytes')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_series_extraction_demo.png', dpi=150, bbox_inches='tight')
    print("   ✓ Saved: time_series_extraction_demo.png")
    
    # Step 6: Show summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"• Input: {len(logs)} raw log entries")
    print(f"• Output: {len(result_df)} time series samples")
    print(f"• Beacon intervals: Every 10 seconds")
    print(f"• Beacon packets detected: {len(beacon_indices)}")
    print(f"• Files created:")
    print(f"   1. time_series_demo.csv")
    print(f"   2. time_series_extraction_demo.png")
    print("\n✓ Demonstration complete!")
    
    return result_df

if __name__ == "__main__":
    demonstrate_extraction()
PYEOF

chmod +x time_series_extractor.py
echo -e "✓ Created: time_series_extractor.py"

# Step 6: Create simple FFT analyzer
echo -e "\n${YELLOW}[6/7] Creating FFT analyzer...${NC}"

cat > fft_analyzer.py << 'PYEOF'
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
    print(f"   Beacon Detected: {'YES' if p_score > 0.6 else 'NO'}")
    
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
    
    # Save results
    results = {
        'p_score': float(p_score),
        'fft_peak': float(fft_peak),
        'autocorr_max': float(autocorr_max),
        'entropy_norm': float(entropy_norm),
        'dominant_frequency': float(dominant_freq),
        'dominant_period': float(dominant_period),
        'beacon_detected': p_score > 0.6
    }
    
    with open('fft_results.json', 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"• P-Score: {p_score:.3f}")
    print(f"• Dominant Period: {dominant_period:.1f} seconds")
    print(f"• Beacon Detected: {'YES' if p_score > 0.6 else 'NO'}")
    print(f"• Files created:")
    print(f"   1. fft_analysis_demo.png")
    print(f"   2. fft_results.json")
    
    return results

if __name__ == "__main__":
    analyze_with_fft()
PYEOF

chmod +x fft_analyzer.py
echo -e "✓ Created: fft_analyzer.py"

# Step 7: Create runner script
echo -e "\n${YELLOW}[7/7] Creating runner script...${NC}"

cat > run_demo.sh << 'RUNEOF'
#!/bin/bash

# C2 Beacon Detection - Complete Demo Runner

echo "========================================"
echo "C2 Beacon Detection Demo"
echo "========================================"

# Activate virtual environment
source venv/bin/activate

# Step 1: Time series extraction
echo ""
echo "STEP 1: Time Series Extraction"
echo "--------------------------------"
python3 time_series_extractor.py

# Step 2: FFT Analysis
echo ""
echo "STEP 2: FFT Analysis"
echo "--------------------------------"
python3 fft_analyzer.py

# Step 3: Show results
echo ""
echo "========================================"
echo "DEMO COMPLETE!"
echo "========================================"
echo ""
echo "Files generated:"
ls -la *.csv *.png *.json 2>/dev/null | awk '{print $NF}'
echo ""
echo "To show your professor:"
echo "1. time_series_extraction_demo.png - Shows log → time series"
echo "2. fft_analysis_demo.png - Shows FFT and P-score calculation"
echo "3. fft_results.json - Contains detection results"
echo ""
echo "Quick commands to run during meeting:"
echo "  python3 time_series_extractor.py"
echo "  python3 fft_analyzer.py"
echo "========================================"
RUNEOF

chmod +x run_demo.sh
echo -e "✓ Created: run_demo.sh"

# Final message
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}SETUP COMPLETE!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "\n${YELLOW}To run the complete demo:${NC}"
echo -e "  ./run_demo.sh"
echo -e "\n${YELLOW}Or run step by step:${NC}"
echo -e "  source venv/bin/activate"
echo -e "  python3 time_series_extractor.py"
echo -e "  python3 fft_analyzer.py"
echo -e "\n${YELLOW}Files created:${NC}"
find . -type f -name "*.py" -o -name "*.sh" | sort
