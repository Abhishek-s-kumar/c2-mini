#!/usr/bin/env python3
"""
C2 Beacon Detection Analyzer
Parses Suricata logs, creates time series, applies FFT, and visualizes results
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class BeaconAnalyzer:
    def __init__(self, log_file='c2_beacon.log'):
        """
        Initialize Beacon Analyzer
        
        Args:
            log_file: Path to Suricata log file
        """
        self.log_file = log_file
        self.df = None
        self.beacon_data = None
        self.time_series = None
        
    def parse_logs(self):
        """Parse Suricata JSON logs"""
        print("Parsing log file...")
        
        records = []
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError:
                    continue
        
        self.df = pd.DataFrame(records)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"Parsed {len(self.df)} records")
        print(f"Time range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        # Extract beacon-related alerts
        self.beacon_data = self.df[self.df['alert'].apply(
            lambda x: 'Beacon' in str(x.get('signature', '')) if isinstance(x, dict) else False
        )].copy()
        
        print(f"Found {len(self.beacon_data)} beacon-related alerts")
        
        return self.df
    
    def create_time_series(self, time_resolution='1S'):
        """
        Create time series from beacon data
        
        Args:
            time_resolution: Time bin size (e.g., '1S' for 1 second)
        """
        if self.beacon_data is None:
            self.parse_logs()
        
        print("\nCreating time series...")
        
        # Group by time bins
        self.beacon_data.set_index('timestamp', inplace=True)
        
        # Create multiple time series
        time_series = {}
        
        # 1. Count of beacon alerts per time bin
        time_series['alert_count'] = self.beacon_data.resample(time_resolution).size()
        
        # 2. Total bytes sent to server per time bin
        time_series['bytes_toserver'] = self.beacon_data['bytes_toserver'].resample(time_resolution).sum()
        
        # 3. Total bytes received from server per time bin
        time_series['bytes_toclient'] = self.beacon_data['bytes_toclient'].resample(time_resolution).sum()
        
        # 4. Packet count (sum of packets to/from server)
        time_series['packets_total'] = self.beacon_data['flow'].apply(
            lambda x: x.get('pkts_toserver', 0) + x.get('pkts_toclient', 0) if isinstance(x, dict) else 0
        ).resample(time_resolution).sum()
        
        # Combine into DataFrame
        self.time_series = pd.DataFrame(time_series).fillna(0)
        
        # Add time in seconds from start
        start_time = self.time_series.index[0]
        self.time_series['seconds'] = (self.time_series.index - start_time).total_seconds()
        
        print(f"Time series created with {len(self.time_series)} time bins")
        print(f"Time resolution: {time_resolution}")
        
        return self.time_series
    
    def plot_time_series(self, save_path='time_series_plot.png'):
        """Plot the time series data"""
        if self.time_series is None:
            self.create_time_series()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('C2 Beacon Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Alert Count
        ax = axes[0, 0]
        ax.plot(self.time_series.index, self.time_series['alert_count'], 
                'b-', linewidth=1.5, marker='o', markersize=3)
        ax.set_title('Beacon Alert Count Over Time', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Alert Count')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 2: Bytes to Server
        ax = axes[0, 1]
        ax.plot(self.time_series.index, self.time_series['bytes_toserver'], 
                'r-', linewidth=1.5, marker='s', markersize=3)
        ax.set_title('Bytes to C2 Server Over Time', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Bytes')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 3: Total Packets
        ax = axes[1, 0]
        ax.plot(self.time_series.index, self.time_series['packets_total'], 
                'g-', linewidth=1.5, marker='^', markersize=3)
        ax.set_title('Total Packets Over Time', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Packet Count')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Plot 4: All metrics normalized
        ax = axes[1, 1]
        metrics = ['alert_count', 'bytes_toserver', 'packets_total']
        colors = ['b', 'r', 'g']
        
        for metric, color in zip(metrics, colors):
            normalized = (self.time_series[metric] - self.time_series[metric].min()) / \
                        (self.time_series[metric].max() - self.time_series[metric].min())
            ax.plot(self.time_series.index, normalized, color + '-', 
                   linewidth=1.5, label=metric)
        
        ax.set_title('Normalized Metrics Comparison', fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Normalized Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Time series plot saved to: {save_path}")
        plt.show()
        
        return fig
    
    def apply_fft(self, signal_data, sampling_rate=1):
        """
        Apply FFT to time series data
        
        Args:
            signal_data: Time series data
            sampling_rate: Samples per second
        
        Returns:
            frequencies, fft_magnitude
        """
        n = len(signal_data)
        
        # Apply window function to reduce spectral leakage
        window = signal.windows.hann(n)
        windowed_signal = signal_data * window
        
        # Apply FFT
        fft_result = fft(windowed_signal)
        fft_magnitude = np.abs(fft_result)
        
        # Calculate frequencies
        frequencies = fftfreq(n, d=1/sampling_rate)
        
        # Only return positive frequencies
        positive_freq_mask = frequencies >= 0
        frequencies = frequencies[positive_freq_mask]
        fft_magnitude = fft_magnitude[positive_freq_mask]
        
        return frequencies, fft_magnitude
    
    def detect_beacon_periodicity(self, frequencies, fft_magnitude, threshold=0.3):
        """
        Detect periodic beacon patterns from FFT
        
        Args:
            frequencies: Frequency array from FFT
            fft_magnitude: FFT magnitude array
            threshold: Relative threshold for peak detection
        
        Returns:
            dict with detection results
        """
        # Normalize magnitude
        normalized_magnitude = fft_magnitude / np.max(fft_magnitude)
        
        # Find peaks above threshold
        peaks_mask = normalized_magnitude > threshold
        peak_frequencies = frequencies[peaks_mask]
        peak_magnitudes = normalized_magnitude[peaks_mask]
        
        # Calculate periodicity score (research paper formula)
        # Simple scoring: weighted sum of significant peaks
        score = np.sum(peak_magnitudes ** 2) * 100
        
        # Calculate dominant period
        if len(peak_frequencies) > 0:
            dominant_idx = np.argmax(peak_magnitudes)
            dominant_freq = peak_frequencies[dominant_idx]
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else 0
        else:
            dominant_freq = 0
            dominant_period = 0
        
        results = {
            'periodicity_score': score,
            'dominant_frequency': dominant_freq,
            'dominant_period': dominant_period,
            'peak_frequencies': peak_frequencies,
            'peak_magnitudes': peak_magnitudes,
            'is_periodic': score > 10  # Threshold for beacon detection
        }
        
        return results
    
    def plot_fft_analysis(self, save_path='fft_analysis.png'):
        """Plot FFT analysis results"""
        if self.time_series is None:
            self.create_time_series()
        
        # Prepare data for FFT
        signal_data = self.time_series['alert_count'].values
        sampling_rate = 1  # 1 sample per second
        
        # Apply FFT
        frequencies, fft_magnitude = self.apply_fft(signal_data, sampling_rate)
        
        # Detect periodicity
        detection_results = self.detect_beacon_periodicity(frequencies, fft_magnitude)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('C2 Beacon FFT Frequency Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Original Time Series
        ax = axes[0, 0]
        ax.plot(self.time_series['seconds'], signal_data, 'b-', linewidth=2)
        ax.set_title('Original Time Series (Alert Count)', fontweight='bold')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Alert Count')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: FFT Magnitude Spectrum
        ax = axes[0, 1]
        ax.plot(frequencies, fft_magnitude, 'r-', linewidth=2)
        ax.set_title('FFT Magnitude Spectrum', fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.grid(True, alpha=0.3)
        
        # Highlight peaks above threshold
        threshold = 0.3 * np.max(fft_magnitude)
        peaks = frequencies[fft_magnitude > threshold]
        peak_mags = fft_magnitude[fft_magnitude > threshold]
        ax.plot(peaks, peak_mags, 'go', markersize=8, label='Significant Peaks')
        ax.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.legend()
        
        # Plot 3: Normalized FFT with Period Indication
        ax = axes[1, 0]
        normalized_magnitude = fft_magnitude / np.max(fft_magnitude)
        ax.plot(frequencies, normalized_magnitude, 'g-', linewidth=2)
        ax.set_title('Normalized FFT Spectrum', fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Normalized Magnitude')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add period labels on top axis
        ax2 = ax.twiny()
        significant_periods = []
        for freq in detection_results['peak_frequencies']:
            if freq > 0:
                period = 1 / freq
                significant_periods.append(period)
                ax2.axvline(x=freq, color='orange', linestyle=':', alpha=0.5)
        
        # Plot 4: Periodogram (Welch's method)
        ax = axes[1, 1]
        f_welch, Pxx_welch = signal.welch(signal_data, fs=sampling_rate, 
                                         nperseg=min(256, len(signal_data)))
        ax.semilogy(f_welch, Pxx_welch, 'm-', linewidth=2)
        ax.set_title('Power Spectral Density (Welch)', fontweight='bold')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power/Frequency (dB/Hz)')
        ax.grid(True, alpha=0.3)
        
        # Add detection results text
        detection_text = f"""
        Detection Results:
        -----------------
        Periodicity Score: {detection_results['periodicity_score']:.2f}
        Dominant Frequency: {detection_results['dominant_frequency']:.3f} Hz
        Dominant Period: {detection_results['dominant_period']:.1f} seconds
        Beacon Detected: {'YES' if detection_results['is_periodic'] else 'NO'}
        """
        
        plt.figtext(0.02, 0.02, detection_text, fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"FFT analysis plot saved to: {save_path}")
        
        # Print detection results
        print("\n" + "="*50)
        print("BEACON DETECTION RESULTS")
        print("="*50)
        print(f"Periodicity Score: {detection_results['periodicity_score']:.2f}")
        print(f"Dominant Frequency: {detection_results['dominant_frequency']:.3f} Hz")
        print(f"Dominant Period: {detection_results['dominant_period']:.1f} seconds")
        print(f"Beacon Detected: {'YES' if detection_results['is_periodic'] else 'NO'}")
        
        if detection_results['is_periodic']:
            print(f"\n⚠️  C2 BEACON DETECTED!")
            print(f"   Estimated beacon interval: {detection_results['dominant_period']:.1f} seconds")
        
        plt.show()
        
        return fig, detection_results
    
    def generate_report(self, output_file='beacon_analysis_report.txt'):
        """Generate comprehensive analysis report"""
        if self.time_series is None:
            self.create_time_series()
        
        # Perform FFT analysis
        signal_data = self.time_series['alert_count'].values
        frequencies, fft_magnitude = self.apply_fft(signal_data, sampling_rate=1)
        detection_results = self.detect_beacon_periodicity(frequencies, fft_magnitude)
        
        # Generate report
        report = f"""
        C2 BEACON DETECTION ANALYSIS REPORT
        ===================================
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        1. DATA SUMMARY
        ---------------
        Log File: {self.log_file}
        Total Records: {len(self.df)}
        Beacon Alerts: {len(self.beacon_data)}
        Time Range: {self.time_series.index[0]} to {self.time_series.index[-1]}
        Duration: {(self.time_series.index[-1] - self.time_series.index[0]).total_seconds():.1f} seconds
        
        2. TIME SERIES STATISTICS
        -------------------------
        Alert Count:
          - Mean: {self.time_series['alert_count'].mean():.2f}
          - Std Dev: {self.time_series['alert_count'].std():.2f}
          - Max: {self.time_series['alert_count'].max():.0f}
          - Min: {self.time_series['alert_count'].min():.0f}
        
        Bytes to Server:
          - Total: {self.time_series['bytes_toserver'].sum():.0f} bytes
          - Mean per interval: {self.time_series['bytes_toserver'].mean():.2f}
        
        3. FREQUENCY ANALYSIS
        ---------------------
        Dominant Frequency: {detection_results['dominant_frequency']:.4f} Hz
        Dominant Period: {detection_results['dominant_period']:.2f} seconds
        Periodicity Score: {detection_results['periodicity_score']:.2f}/100
        
        4. DETECTION RESULTS
        --------------------
        Beacon Pattern Detected: {'YES' if detection_results['is_periodic'] else 'NO'}
        Confidence: {'HIGH' if detection_results['periodicity_score'] > 50 else 
                    'MEDIUM' if detection_results['periodicity_score'] > 20 else 'LOW'}
        
        5. SIGNIFICANT FREQUENCIES
        ---------------------------
        """
        
        for i, (freq, mag) in enumerate(zip(detection_results['peak_frequencies'], 
                                           detection_results['peak_magnitudes'])):
            if freq > 0:
                report += f"        Peak {i+1}: {freq:.4f} Hz (Period: {1/freq:.2f}s, Magnitude: {mag:.2%})\n"
        
        report += f"""
        6. RECOMMENDATIONS
        ------------------
        """
        
        if detection_results['is_periodic']:
            report += """        ✅ TAKE ACTION: Periodic C2 beacon detected!
           - Investigate source IPs in beacon alerts
           - Check for malware persistence mechanisms
           - Review network firewall rules
           - Consider blocking C2 server IPs"""
        else:
            report += """        ✅ No clear periodic beacon pattern detected.
           - Continue monitoring for suspicious activity
           - Review alerts for other IOC patterns"""
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)
        
        print(f"\nAnalysis report saved to: {output_file}")
        
        return report

def main():
    """Main execution function"""
    print("C2 Beacon Detection Analyzer")
    print("="*50)
    
    # Initialize analyzer
    analyzer = BeaconAnalyzer('c2_beacon.log')
    
    # Parse logs
    analyzer.parse_logs()
    
    # Create time series
    analyzer.create_time_series(time_resolution='1S')
    
    # Plot time series
    analyzer.plot_time_series('time_series_analysis.png')
    
    # Plot FFT analysis
    analyzer.plot_fft_analysis('fft_frequency_analysis.png')
    
    # Generate report
    analyzer.generate_report('beacon_detection_report.txt')
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("Output files:")
    print("  - time_series_analysis.png")
    print("  - fft_frequency_analysis.png")
    print("  - beacon_detection_report.txt")
    print("="*50)

if __name__ == "__main__":
    main()
