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
